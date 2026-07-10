"""
Agentic RAG 控制器。

该模块用于将“检索→构建上下文→生成答案”的单轮流水线，扩展为带闭环的多轮检索：
1) 首轮按 RetrievalPlan 检索证据；
2) 基于已检索证据，由 LLM 判断证据是否充分，并给出增量检索特征（features / feature_pairs）；
3) 将增量特征合并进 plan 后再次检索，直到充分或达到最大迭代次数。
"""

from __future__ import annotations

import logging

from fin_agent.application.context.context_builder import build_context, refine_context_with_llm
from fin_agent.application.processing.guardrails import extract_json_payload
from fin_agent.application.retrieval.planner import RetrievalPlan
from fin_agent.application.retrieval.retrieval import QueryFeatures
from fin_agent.application.retrieval.retrieval_pipeline import retrieve_evidence
from fin_agent.application.tracing.tracing import RetrievalTrace, sum_token_usage, zero_token_usage
from fin_agent.domain.models import EvidenceSnippet, Question, RetrievalConfig, TokenUsage
from fin_agent.infrastructure.data_access import DocumentRepository
from fin_agent.infrastructure.llm.openai_compatible_client import ChatMessage, OpenAiCompatibleChatClient

logger = logging.getLogger(__name__)


def serialize_query_features(features: QueryFeatures) -> dict[str, object]:
    """将 QueryFeatures 序列化为 JSON 友好结构。"""
    return {
        "years": list(features.years),
        "numbers": list(features.numbers),
        "clauses": list(features.clauses),
        "features": list(features.features),
        "feature_pairs": [{"key": key, "value": value} for key, value in features.feature_pairs],
    }


def _normalize_feature_key(text: str) -> str:
    """清洗检索 key，避免带入 value 或噪音。"""
    raw = str(text or "").strip()
    if not raw:
        return ""
    for sep in ("：", ":", "为"):
        if sep in raw:
            raw = raw.split(sep, 1)[0].strip()
    if "/" in raw:
        raw = raw.split("/", 1)[0].strip()
    raw = " ".join(raw.split()).strip()
    if len(raw) > 24:
        raw = raw[:24].strip()
    return raw


def _normalize_feature_value(text: str) -> str:
    """清洗检索 value，保持短且精确。"""
    raw = str(text or "").strip()
    raw = " ".join(raw.split()).strip()
    if len(raw) > 64:
        raw = raw[:64].strip()
    return raw


def merge_query_features(base: QueryFeatures, extra: QueryFeatures, *, max_features: int, max_pairs: int) -> QueryFeatures:
    """合并 QueryFeatures，并控制总量避免 BM25 噪音膨胀。"""
    years = tuple(sorted(set(base.years).union(extra.years)))
    numbers = tuple(sorted(set(base.numbers).union(extra.numbers)))
    clauses = tuple(sorted(set(base.clauses).union(extra.clauses)))

    merged_features: list[str] = []
    seen_features: set[str] = set()
    for item in list(base.features) + list(extra.features):
        key = _normalize_feature_key(item)
        if not key or key in seen_features:
            continue
        merged_features.append(key)
        seen_features.add(key)
        if len(merged_features) >= max(1, int(max_features)):
            break

    merged_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for key, value in list(base.feature_pairs) + list(extra.feature_pairs):
        cleaned_key = _normalize_feature_key(key)
        cleaned_value = _normalize_feature_value(value)
        if not cleaned_key or not cleaned_value:
            continue
        pair = (cleaned_key, cleaned_value)
        if pair in seen_pairs:
            continue
        merged_pairs.append(pair)
        seen_pairs.add(pair)
        if len(merged_pairs) >= max(1, int(max_pairs)):
            break

    return QueryFeatures(
        years=years,
        numbers=numbers,
        clauses=clauses,
        features=tuple(merged_features),
        feature_pairs=tuple(merged_pairs),
    )


def _build_evidence_brief(q: Question, evidence: list[EvidenceSnippet], *, max_per_option: int) -> str:
    """构造用于 judge 的紧凑证据摘要。"""
    sections: list[str] = []
    for option_key in sorted(q.options.keys()):
        sections.append(f"## 选项 {option_key}")
        items = [item for item in evidence if item.option_key == option_key]
        items.sort(key=lambda item: item.score, reverse=True)
        if not items:
            sections.append("None")
            continue
        for hit in items[: max(1, int(max_per_option))]:
            preview = " ".join((hit.content or "").split()).strip()
            if len(preview) > 240:
                preview = preview[:225].rstrip() + " ...(truncated)"
            sections.append(f"- [DocID: {hit.doc_id} | Title: {hit.title} | Chunk: {hit.chunk_id}] {preview}")
    return "\n".join(sections).strip()


def _parse_suggestion_features(raw: object) -> QueryFeatures:
    """解析 judge 返回的建议 features 与 feature_pairs。"""
    if not isinstance(raw, dict):
        return QueryFeatures(years=(), numbers=(), clauses=(), features=(), feature_pairs=())

    features_raw = raw.get("features")
    pairs_raw = raw.get("feature_pairs")

    features: list[str] = []
    seen: set[str] = set()
    if isinstance(features_raw, list):
        for item in features_raw:
            key = _normalize_feature_key(item)
            if not key or key in seen:
                continue
            features.append(key)
            seen.add(key)
            if len(features) >= 6:
                break

    pairs: list[tuple[str, str]] = []
    pair_seen: set[tuple[str, str]] = set()
    if isinstance(pairs_raw, list):
        for item in pairs_raw:
            if not isinstance(item, dict):
                continue
            key = _normalize_feature_key(item.get("key", ""))
            value = _normalize_feature_value(item.get("value", ""))
            if not key or not value:
                continue
            pair = (key, value)
            if pair in pair_seen:
                continue
            pairs.append(pair)
            pair_seen.add(pair)
            if len(pairs) >= 4:
                break

    return QueryFeatures(years=(), numbers=(), clauses=(), features=tuple(features), feature_pairs=tuple(pairs))


def _judge_and_suggest(
    llm: OpenAiCompatibleChatClient,
    q: Question,
    plan: RetrievalPlan,
    evidence: list[EvidenceSnippet],
    *,
    max_per_option: int,
) -> tuple[dict[str, object] | None, dict[str, QueryFeatures], TokenUsage]:
    """让 LLM 判断证据是否充分，并输出下一轮可用的增量检索特征。"""
    evidence_brief = _build_evidence_brief(q=q, evidence=evidence, max_per_option=max_per_option)
    options_text = "\n".join(f"{key}. {value}" for key, value in sorted(q.options.items()))
    plan_brief = "\n".join(
        [
            "global.features: " + " | ".join(plan.features.features[:12]),
            "global.feature_pairs: " + " | ".join(f"{k}={v}" for k, v in plan.features.feature_pairs[:8]),
        ]
        + [
            f"option.{key}.features: " + " | ".join((plan.option_features.get(key) or QueryFeatures((), (), (), (), ())).features[:6])
            for key in sorted(q.options.keys())
        ]
        + [
            f"option.{key}.feature_pairs: " + " | ".join(
                f"{k}={v}" for k, v in (plan.option_features.get(key) or QueryFeatures((), (), (), (), ())).feature_pairs[:4]
            )
            for key in sorted(q.options.keys())
        ]
    ).strip()
    messages = [
        ChatMessage(
            role="system",
            content=(
                "你是金融 RAG 检索路由器。"
                "你的任务是基于已检索到的证据摘要，判断是否需要继续检索，并给出下一轮更精确的检索增量特征。"
                "你必须严格基于证据摘要与题干/选项，不得编造文档中不存在的具体数值或实体全称。"
                "只输出 JSON，不要输出任何解释。"
            ),
        ),
        ChatMessage(
            role="user",
            content=(
                f"领域：{q.domain}\n"
                f"题目：{q.question}\n"
                f"选项：\n{options_text}\n\n"
                f"当前检索特征（摘要）：\n{plan_brief}\n\n"
                f"证据摘要：\n{evidence_brief}\n\n"
                "请输出 JSON：\n"
                "{\n"
                '  "sufficient": true,\n'
                '  "option_status": {"A": {"has_evidence": true, "missing": ""}},\n'
                '  "suggestions": {\n'
                '    "A": {"features": ["字段名/术语"], "feature_pairs": [{"key": "字段名", "value": "精确短串"}]}\n'
                "  }\n"
                "}\n"
                "输出要求：\n"
                "- sufficient：若证据足以判断所有选项则为 true，否则为 false。\n"
                "- option_status：逐项判断该选项是否存在直接证据；若无，missing 用一句话描述缺什么（不超过 30 字）。\n"
                "- suggestions：仅在 sufficient=false 时填写；每个选项最多 3 条 features、最多 2 条 feature_pairs。\n"
                "- features 必须是字段名/术语（名词），不要整句；feature_pairs 的 value 必须是可用于定位证据的精确短串。\n"
            ),
        ),
    ]
    try:
        resp = llm.chat(messages)
        payload = extract_json_payload(resp.content)
        suggestions: dict[str, QueryFeatures] = {}
        raw_suggestions = payload.get("suggestions") if isinstance(payload, dict) else None
        for option_key in sorted(q.options.keys()):
            raw = raw_suggestions.get(option_key) if isinstance(raw_suggestions, dict) else None
            suggestions[option_key] = _parse_suggestion_features(raw)
        return payload, suggestions, resp.usage
    except Exception as exc:
        logger.warning("agentic judge 失败：qid=%s error=%s", q.qid, repr(exc))
        return None, {k: QueryFeatures(years=(), numbers=(), clauses=(), features=(), feature_pairs=()) for k in q.options.keys()}, zero_token_usage()


def run_agentic_rag(
    llm: OpenAiCompatibleChatClient,
    docs: DocumentRepository,
    retrieval: RetrievalConfig,
    q: Question,
    plan: RetrievalPlan,
    doc_ids: list[str],
) -> tuple[RetrievalPlan, list[EvidenceSnippet], RetrievalTrace, str, TokenUsage, list[dict[str, object]]]:
    """执行带闭环的多轮检索，并返回最终证据、上下文与追踪信息。"""
    max_iters = max(1, int(retrieval.agentic_max_iterations))
    agentic_traces: list[dict[str, object]] = []
    total_usage = zero_token_usage()
    current_plan = plan
    final_context = ""

    for iteration in range(1, max_iters + 1):
        evidence, retrieval_trace = retrieve_evidence(
            docs=docs,
            retrieval=retrieval,
            q=q,
            doc_ids=doc_ids,
            plan=current_plan,
        )
        context, _ = build_context(
            llm=llm,
            docs=docs,
            retrieval=retrieval,
            q=q,
            plan=current_plan,
            doc_ids=doc_ids,
            evidence=evidence,
        )
        refined_context = context
        refine_usage = zero_token_usage()
        if retrieval.agentic_enable_context_refine and context and len(context) > int(retrieval.refine_context_chars):
            refined_context, refine_usage = refine_context_with_llm(llm=llm, q=q, context=context)
            if refined_context:
                refined_context = refined_context[: int(retrieval.max_context_chars)]
        total_usage = sum_token_usage(total_usage, refine_usage)

        option_hit_counts = {
            option_key: len([item for item in evidence if item.option_key == option_key])
            for option_key in sorted(q.options.keys())
        }
        heuristic_sufficient = all(
            count >= max(0, int(retrieval.agentic_min_hits_per_option)) for count in option_hit_counts.values()
        )
        judge_payload = None
        applied_delta = False

        suggestions: dict[str, QueryFeatures] = {}
        judge_usage = zero_token_usage()
        if retrieval.agentic_enable_judge:
            judge_payload, suggestions, judge_usage = _judge_and_suggest(
                llm=llm,
                q=q,
                plan=current_plan,
                evidence=evidence,
                max_per_option=max(2, int(retrieval.per_doc_top_k)),
            )
            total_usage = sum_token_usage(total_usage, judge_usage)

        sufficient = bool(judge_payload.get("sufficient")) if isinstance(judge_payload, dict) else heuristic_sufficient
        if not sufficient and suggestions:
            updated_option_features: dict[str, QueryFeatures] = {}
            deltas: dict[str, dict[str, object]] = {}
            for option_key in sorted(q.options.keys()):
                base = current_plan.option_features.get(option_key) or QueryFeatures(years=(), numbers=(), clauses=(), features=(), feature_pairs=())
                extra = suggestions.get(option_key) or QueryFeatures(years=(), numbers=(), clauses=(), features=(), feature_pairs=())
                merged = merge_query_features(base, extra, max_features=8, max_pairs=4)
                updated_option_features[option_key] = merged
                if (merged != base) and (extra.features or extra.feature_pairs):
                    applied_delta = True
                deltas[option_key] = serialize_query_features(extra)

            updated_global = current_plan.features
            current_plan = RetrievalPlan(
                global_query=current_plan.global_query,
                option_queries=dict(current_plan.option_queries),
                features=updated_global,
                option_features=updated_option_features,
            )
            merged_snapshot = {k: serialize_query_features(v) for k, v in sorted(updated_option_features.items())}
        else:
            deltas = {}
            merged_snapshot = {}

        agentic_traces.append(
            {
                "iteration": iteration,
                "doc_ids": list(doc_ids),
                "option_hit_counts": dict(option_hit_counts),
                "context_chars": len(context),
                "refined_context_chars": len(refined_context),
                "heuristic_sufficient": heuristic_sufficient,
                "judge_payload": judge_payload if isinstance(judge_payload, dict) else None,
                "delta_option_features": deltas,
                "merged_option_features": merged_snapshot,
                "applied_delta": applied_delta,
            }
        )
        final_context = refined_context or context

        if sufficient or not applied_delta:
            return current_plan, evidence, retrieval_trace, final_context, total_usage, agentic_traces

    evidence, retrieval_trace = retrieve_evidence(
        docs=docs,
        retrieval=retrieval,
        q=q,
        doc_ids=doc_ids,
        plan=current_plan,
    )
    context, _ = build_context(
        llm=llm,
        docs=docs,
        retrieval=retrieval,
        q=q,
        plan=current_plan,
        doc_ids=doc_ids,
        evidence=evidence,
    )
    final_context = context[: int(retrieval.max_context_chars)]
    agentic_traces.append(
        {
            "iteration": max_iters + 1,
            "note": "forced_finalize",
            "context_chars": len(context),
        }
    )
    return current_plan, evidence, retrieval_trace, final_context, total_usage, agentic_traces
