from __future__ import annotations

import logging

from fin_agent.application.domain_specialists import build_domain_supplement
from fin_agent.application.planner import RetrievalPlan
from fin_agent.application.retrieval import DOMAIN_PROMPT_HINTS
from fin_agent.application.tracing import zero_token_usage
from fin_agent.domain.models import EvidenceSnippet, Question, RetrievalConfig, TokenUsage
from fin_agent.infrastructure.data_access import DocumentRepository
from fin_agent.infrastructure.llm.openai_compatible_client import ChatMessage, OpenAiCompatibleChatClient

logger = logging.getLogger(__name__)


def _normalize_inline_text(text: str) -> str:
    """将文本压缩为单行，避免上下文膨胀。"""
    return " ".join((text or "").split()).strip()


def _truncate(text: str, max_chars: int) -> str:
    """截断文本到指定长度。"""
    normalized = _normalize_inline_text(text)
    if max_chars <= 0:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 15)].rstrip() + " ...(truncated)"


def format_hit_content(text: str, max_chars: int, truncate: bool) -> str:
    """按配置决定命中文本是否需要截断。"""
    if truncate:
        return _truncate(text, max_chars=max_chars)
    return _normalize_inline_text(text)


def build_option_doc_coverage_context(
    q: Question,
    doc_ids: list[str],
    evidence: list[EvidenceSnippet],
    per_hit_max_chars: int,
    truncate_hit_content: bool,
    option_text_max_chars: int | None,
) -> str:
    """构造按 option×doc 覆盖展示的证据链上下文。"""
    sections: list[str] = []
    for option_key in sorted(q.options.keys()):
        option_text = q.options[option_key]
        if option_text_max_chars is not None:
            option_text = _truncate(option_text, max_chars=option_text_max_chars)

        sections.append(f"## 选项 {option_key}")
        sections.append(f"候选陈述：{option_text}")
        for doc_id in doc_ids:
            sections.append(f"### DocID: {doc_id}")
            matched = [
                item
                for item in evidence
                if item.option_key == option_key and item.doc_id == doc_id
            ]
            if not matched:
                sections.append("None")
                continue
            matched.sort(key=lambda item: item.score, reverse=True)
            item = matched[0]
            preview = format_hit_content(
                text=item.content,
                max_chars=per_hit_max_chars,
                truncate=truncate_hit_content,
            )
            sections.append(
                f"- [DocID: {item.doc_id} | Title: {item.title} | Score: {item.score:.3f}] {preview}"
            )
        sections.append("")
    return "\n".join(sections).strip()


def build_context(
    llm: OpenAiCompatibleChatClient,
    docs: DocumentRepository,
    retrieval: RetrievalConfig,
    q: Question,
    plan: RetrievalPlan,
    doc_ids: list[str],
    evidence: list[EvidenceSnippet],
) -> tuple[str, TokenUsage]:
    supplement = None
    if retrieval.enable_domain_supplement:
        supplement = build_domain_supplement(q=q, doc_ids=doc_ids, docs=docs, evidence=evidence)
    per_hit_max_chars = max(1, min(int(retrieval.per_hit_max_chars), int(retrieval.chunk_max_chars)))
    option_text_max_chars: int | None = None

    def build_main() -> str:
        return build_option_doc_coverage_context(
            q=q,
            doc_ids=doc_ids,
            evidence=evidence,
            per_hit_max_chars=per_hit_max_chars,
            truncate_hit_content=retrieval.truncate_hit_content_for_context,
            option_text_max_chars=option_text_max_chars,
        )

    main_context = build_main()
    if supplement is not None and supplement.content:
        full_context = f"{main_context}\n\n## {supplement.title}\n{supplement.content}".strip()
    else:
        full_context = main_context

    if len(full_context) <= retrieval.max_context_chars:
        return full_context, zero_token_usage()

    if supplement is not None and supplement.content and len(main_context) <= retrieval.max_context_chars:
        return main_context, zero_token_usage()

    for _ in range(12):
        if retrieval.truncate_hit_content_for_context and per_hit_max_chars > 90:
            per_hit_max_chars = max(90, per_hit_max_chars - 60)
        elif option_text_max_chars is None:
            option_text_max_chars = 240
        elif option_text_max_chars > 120:
            option_text_max_chars = max(120, option_text_max_chars - 40)
        else:
            break

        main_context = build_main()
        if len(main_context) <= retrieval.max_context_chars:
            return main_context, zero_token_usage()

    return main_context[: retrieval.max_context_chars], zero_token_usage()


def refine_context_with_llm(
    llm: OpenAiCompatibleChatClient,
    q: Question,
    context: str,
) -> tuple[str, TokenUsage]:
    options_text = "\n".join(f"{key}. {value}" for key, value in sorted(q.options.items()))
    messages = [
        ChatMessage(
            role="system",
            content=(
                "你是金融证据压缩器。"
                "只保留能够直接支撑或反驳选项的证据。"
                "不允许引入外部知识。"
                "若某条证据无关，直接丢弃。"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"领域：{q.domain}\n"
                f"题目：{q.question}\n"
                f"选项：\n{options_text}\n\n"
                f"请按 A/B/C/D 分组输出最关键证据，每条证据保留 DocID 与 Title。\n"
                f"如果某个选项没有直接证据，请输出 None。\n\n"
                f"候选证据：\n{context}\n\n"
                f"补充提示：{DOMAIN_PROMPT_HINTS.get(q.domain, '')}"
            ),
        ),
    ]
    try:
        response = llm.chat(messages)
        return response.content.strip(), response.usage
    except Exception as exc:
        logger.warning("上下文精筛失败：qid=%s error=%s", q.qid, repr(exc))
        return "", zero_token_usage()
