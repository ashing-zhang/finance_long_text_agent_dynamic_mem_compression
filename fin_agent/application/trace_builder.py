from __future__ import annotations

from fin_agent.application.planner import RetrievalPlan
from fin_agent.application.retrieval import DOMAIN_PROMPT_HINTS
from fin_agent.application.tracing import summarize_evidence_hits, token_usage_to_dict, truncate_text
from fin_agent.domain.models import EvidenceSnippet, Question, TokenUsage
from fin_agent.infrastructure.llm.openai_compatible_client import ChatMessage
from fin_agent.application.tracing import QuestionTrace, RetrievalTrace


def build_question_trace(
    q: Question,
    plan: RetrievalPlan,
    doc_ids: list[str],
    retrieval_trace: RetrievalTrace,
    evidence: list[EvidenceSnippet],
    context: str,
    refine_usage: TokenUsage,
    messages: list[ChatMessage],
    model_output: str,
    answer: str,
    answer_usage: TokenUsage,
    total_usage: TokenUsage,
    feature_usage: TokenUsage,
) -> QuestionTrace:
    thought_trace = {
        "global_query": plan.global_query,
        "option_queries": dict(plan.option_queries),
        "query_features": {
            "years": list(plan.features.years),
            "numbers": list(plan.features.numbers),
            "clauses": list(plan.features.clauses),
            "keywords": list(plan.features.keywords),
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
                "hit_count": item.hit_count,
                "top_hits": item.top_hits,
                "option_doc_hits": item.option_doc_hits,
            }
            for item in retrieval_trace.rounds
        ],
        "evidence": summarize_evidence_hits(evidence, limit=len(evidence)),
    }
    answer_trace = {
        "context_chars": len(context),
        "context_preview": truncate_text(context, max_chars=1500),
        "messages": [{"role": m.role, "content": truncate_text(m.content, max_chars=4000)} for m in messages],
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
