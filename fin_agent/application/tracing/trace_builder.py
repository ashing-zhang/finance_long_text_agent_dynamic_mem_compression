from __future__ import annotations

from fin_agent.application.retrieval.retrieval import DOMAIN_PROMPT_HINTS
from fin_agent.application.tracing.tracing import summarize_evidence_hits, token_usage_to_dict
from fin_agent.domain.models import EvidenceSnippet, Question, TokenUsage
from fin_agent.infrastructure.llm.openai_compatible_client import ChatMessage
from fin_agent.application.tracing.tracing import QuestionTrace, RetrievalTrace


def serialize_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """将发送给 LLM 的完整 message 列表转换为可序列化结构。"""
    return [{"role": message.role, "content": message.content} for message in messages]


def serialize_retrieval_plan(plan: "RetrievalPlan") -> dict[str, object]:
    from fin_agent.application.retrieval.planner import RetrievalPlan
    return {
        "global_query": plan.global_query,
        "option_queries": dict(plan.option_queries),
        "features": {
            "years": list(plan.features.years),
            "numbers": list(plan.features.numbers),
            "clauses": list(plan.features.clauses),
            "features": list(plan.features.features),
            "expanded_terms": list(plan.features.expanded_terms),
            "feature_pairs": [{"key": key, "value": value} for key, value in plan.features.feature_pairs],
        },
        "option_features": {
            option_key: {
                "years": list(features.years),
                "numbers": list(features.numbers),
                "clauses": list(features.clauses),
                "features": list(features.features),
                "expanded_terms": list(features.expanded_terms),
                "feature_pairs": [{"key": key, "value": value} for key, value in features.feature_pairs],
            }
            for option_key, features in sorted(plan.option_features.items())
        },
    }


def build_question_trace(
    q: Question,
    plan: "RetrievalPlan",
    doc_ids: list[str],
    retrieval_trace: RetrievalTrace,
    evidence: list[EvidenceSnippet],
    context: str,
    refine_usage: TokenUsage,
    agentic_trace: list[dict[str, object]] | None,
    messages: list[ChatMessage],
    model_output: str,
    answer: str,
    answer_usage: TokenUsage,
    total_usage: TokenUsage,
    feature_usage: TokenUsage,
) -> QuestionTrace:
    thought_trace = {
        "retrieval_plan": serialize_retrieval_plan(plan),
        "global_query": plan.global_query,
        "option_queries": dict(plan.option_queries),
        "query_features": {
            "years": list(plan.features.years),
            "numbers": list(plan.features.numbers),
            "clauses": list(plan.features.clauses),
            "features": list(plan.features.features),
            "expanded_terms": list(plan.features.expanded_terms),
            "feature_pairs": [{"key": key, "value": value} for key, value in plan.features.feature_pairs],
        },
        "domain_hint": DOMAIN_PROMPT_HINTS.get(q.domain, ""),
        "candidate_doc_ids": list(doc_ids),
    }
    search_trace = {
        "candidate_doc_ids": list(retrieval_trace.candidate_doc_ids),
        "used_fallback": retrieval_trace.used_fallback,
        "retrieval_rounds": [
            {
                "round_index": item.round_index,
                "query_mode": item.query_mode,
                "option_queries": item.option_queries,
                "literal_terms": item.literal_terms,
                "hit_count": item.hit_count,
            }
            for item in retrieval_trace.rounds
        ],
        "evidence": summarize_evidence_hits(evidence, limit=len(evidence)),
    }
    if agentic_trace:
        search_trace["agentic_iterations"] = list(agentic_trace)
    answer_trace = {
        "context_chars": len(context),
        "messages": serialize_messages(messages),
        "feature_usage": token_usage_to_dict(feature_usage),
        "refine_usage": token_usage_to_dict(refine_usage),
        "answer_usage": token_usage_to_dict(answer_usage),
        "total_usage": token_usage_to_dict(total_usage),
        "model_output": model_output,
        "normalized_answer": answer,
    }
    return QuestionTrace(
        qid=q.qid,
        domain=q.domain,
        question=q.question,
        options=dict(q.options),
        thought_trace=thought_trace,
        search_trace=search_trace,
        answer_trace=answer_trace,
    )
