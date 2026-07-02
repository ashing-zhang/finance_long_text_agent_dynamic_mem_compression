from __future__ import annotations

from fin_agent.compat import dataclass

from fin_agent.domain.models import AnswerRecord, EvidenceSnippet, TokenUsage


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    answers: list[AnswerRecord]
    total_usage: TokenUsage
    traces: list["QuestionTrace"]


@dataclass(frozen=True, slots=True)
class RetrievalRoundTrace:
    round_index: int
    query_mode: str
    option_queries: dict[str, str]
    hit_count: int
    top_hits: list[dict[str, object]]
    option_doc_hits: dict[str, dict[str, dict[str, object]]]


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    candidate_doc_ids: list[str]
    rounds: list[RetrievalRoundTrace]
    used_fallback: bool


@dataclass(frozen=True, slots=True)
class QuestionTrace:
    qid: str
    domain: str
    question: str
    options: dict[str, str]
    thought_trace: dict[str, object]
    search_trace: dict[str, object]
    answer_trace: dict[str, object]


def zero_token_usage() -> TokenUsage:
    return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def sum_token_usage(*items: TokenUsage) -> TokenUsage:
    prompt_tokens = sum(item.prompt_tokens for item in items)
    completion_tokens = sum(item.completion_tokens for item in items)
    total_tokens = sum(item.total_tokens for item in items)
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def token_usage_to_dict(usage: TokenUsage) -> dict[str, int]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def truncate_text(text: str, max_chars: int) -> str:
    normalized = (text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 15].rstrip() + "\n...(truncated)"


def summarize_evidence_hits(items: list[EvidenceSnippet], limit: int) -> list[dict[str, object]]:
    return [
        {
            "doc_id": item.doc_id,
            "title": item.title,
            "chunk_id": item.chunk_id,
            "score": round(float(item.score), 4),
            "option_key": item.option_key,
            "content_preview": truncate_text(item.content, max_chars=240),
        }
        for item in items[:limit]
    ]


def summarize_hits_by_option_doc(
    items: list[EvidenceSnippet],
    option_keys: list[str],
    doc_ids: list[str],
    preview_limit: int = 5,
) -> dict[str, dict[str, dict[str, object]]]:
    grouped: dict[str, dict[str, dict[str, object]]] = {}
    for option_key in option_keys:
        option_group: dict[str, dict[str, object]] = {}
        for doc_id in doc_ids:
            matched = [
                item
                for item in items
                if item.option_key == option_key and item.doc_id == doc_id
            ]
            option_group[doc_id] = {
                "hit_count": len(matched),
                "hits_preview": summarize_evidence_hits(matched, limit=preview_limit),
            }
        grouped[option_key] = option_group
    return grouped


def summarize_hits_by_option_doc_flat(
    items: list[EvidenceSnippet],
    option_keys: list[str],
    doc_ids: list[str],
    per_doc_limit: int,
) -> list[dict[str, object]]:
    flattened: list[dict[str, object]] = []
    for option_key in option_keys:
        for doc_id in doc_ids:
            matched = [
                item
                for item in items
                if item.option_key == option_key and item.doc_id == doc_id
            ]
            if not matched:
                continue
            flattened.extend(summarize_evidence_hits(matched, limit=per_doc_limit))
    return flattened
