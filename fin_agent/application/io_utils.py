from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from fin_agent.domain.models import AnswerRecord, EvidenceSnippet, Question
from fin_agent.application.tracing import QuestionTrace

logger = logging.getLogger(__name__)


def write_answer_csv(path: Path, records: list[AnswerRecord]) -> None:
    import csv

    total_prompt = sum(r.usage.prompt_tokens for r in records)
    total_completion = sum(r.usage.completion_tokens for r in records)
    total_total = sum(r.usage.total_tokens for r in records)

    path.parent.mkdir(parents=True, exist_ok=True)
    output_path = path
    try:
        f = output_path.open("w", encoding="utf-8", newline="")
    except PermissionError:
        output_path = _pick_alternate_output_path(path)
        logger.warning("写入失败（文件可能被占用），回退写入：%s -> %s", path, output_path)
        f = output_path.open("w", encoding="utf-8", newline="")

    with f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        writer.writerow(["summary", "", total_prompt, total_completion, total_total])
        for r in records:
            writer.writerow([r.qid, r.answer, r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.total_tokens])


def write_logs_csv(path: Path, traces: list[QuestionTrace]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "qid",
                "domain",
                "question",
                "options_json",
                "thought_trace_json",
                "search_trace_json",
                "answer_trace_json",
            ]
        )
        for trace in traces:
            writer.writerow(
                [
                    trace.qid,
                    trace.domain,
                    trace.question,
                    json.dumps(trace.options, ensure_ascii=False),
                    json.dumps(trace.thought_trace, ensure_ascii=False),
                    json.dumps(trace.search_trace, ensure_ascii=False),
                    json.dumps(trace.answer_trace, ensure_ascii=False),
                ]
            )


def _pick_alternate_output_path(path: Path) -> Path:
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    for i in range(1, 1000):
        candidate = parent / f"{stem}{i}{suffix}"
        if not candidate.exists():
            return candidate
    return parent / f"{stem}{int(math.floor(1000 * (1 + math.fabs(math.sin(len(stem))))))}{suffix}"


def write_evidence_jsonl(path: Path, q: Question, evidence: list[EvidenceSnippet]) -> None:
    payload = {
        "qid": q.qid,
        "domain": q.domain,
        "answer_format": q.answer_format.value,
        "evidence_retrieval": [
            {
                "doc_id": e.doc_id,
                "title": e.title,
                "chunk_id": e.chunk_id,
                "score": e.score,
                "option_key": e.option_key,
                "quoted_clause": e.content,
            }
            for e in evidence
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

