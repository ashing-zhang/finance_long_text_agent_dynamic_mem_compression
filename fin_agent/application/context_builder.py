from __future__ import annotations

import logging

from fin_agent.application.domain_specialists import build_domain_supplement
from fin_agent.application.planner import RetrievalPlan
from fin_agent.application.retrieval import DOMAIN_PROMPT_HINTS, build_grouped_context, trim_grouped_context
from fin_agent.application.tracing import zero_token_usage
from fin_agent.domain.models import EvidenceSnippet, Question, RetrievalConfig, TokenUsage
from fin_agent.infrastructure.data_access import DocumentRepository
from fin_agent.infrastructure.llm.openai_compatible_client import ChatMessage, OpenAiCompatibleChatClient

logger = logging.getLogger(__name__)


def build_context(
    llm: OpenAiCompatibleChatClient,
    docs: DocumentRepository,
    retrieval: RetrievalConfig,
    q: Question,
    plan: RetrievalPlan,
    doc_ids: list[str],
    evidence: list[EvidenceSnippet],
) -> tuple[str, TokenUsage]:
    context = build_grouped_context(
        q=q,
        doc_ids=doc_ids,
        evidence=evidence,
        docs=docs,
    )
    supplement = build_domain_supplement(q=q, doc_ids=doc_ids, docs=docs, evidence=evidence)
    if supplement is not None and supplement.content:
        context = f"{context}\n\n## {supplement.title}\n{supplement.content}".strip()
    if len(context) <= retrieval.max_context_chars:
        return context, zero_token_usage()

    if len(context) > retrieval.refine_context_chars:
        refined, usage = refine_context_with_llm(llm=llm, q=q, context=context)
        if refined:
            refined = trim_grouped_context(
                query=plan.global_query,
                text=refined,
                max_chars=retrieval.max_context_chars,
            )
            return refined, usage

    compressed = trim_grouped_context(
        query=plan.global_query,
        text=context,
        max_chars=retrieval.max_context_chars,
    )
    return compressed, zero_token_usage()


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

