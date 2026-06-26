from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

from fin_agent.domain.models import (
    AnswerFormat,
    AnswerRecord,
    EvidenceSnippet,
    LlmConfig,
    Question,
    RetrievalConfig,
    RunConfig,
    TokenUsage,
)
from fin_agent.infrastructure.data_access import DocumentRepository, QuestionRepository
from fin_agent.infrastructure.llm.openai_compatible_client import (
    ChatMessage,
    OpenAiCompatibleChatClient,
)

logger = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """配置 logging。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """整批评测结果。"""

    answers: list[AnswerRecord]
    total_usage: TokenUsage


class FinanceLongTextAgent:
    """金融长文本问答 Agent（检索 + 压缩 + 受控输出）。"""

    def __init__(
        self,
        llm: OpenAiCompatibleChatClient,
        docs: DocumentRepository,
        retrieval: RetrievalConfig,
    ) -> None:
        """初始化 Agent。"""
        self._llm = llm
        self._docs = docs
        self._retrieval = retrieval

    def answer(self, q: Question) -> tuple[AnswerRecord, list[EvidenceSnippet]]:
        """回答单题，返回 answer.csv 行与证据片段。"""
        doc_ids = q.doc_ids or self._docs.list_doc_ids(q.domain)
        if not doc_ids:
            logger.warning("题目无 doc_ids 且无法枚举领域文档：%s", q.qid)

        query = self._build_query(q)
        evidence = self._retrieve_evidence(domain=q.domain, doc_ids=doc_ids, query=query)
        context = self._build_context(query=query, evidence=evidence)
        messages = self._build_messages(q=q, context=context)

        resp = self._llm.chat(messages)
        answer = self._normalize_answer(q.answer_format, resp.content)
        record = AnswerRecord(qid=q.qid, answer=answer, usage=resp.usage)
        return record, evidence

    def _build_query(self, q: Question) -> str:
        """构造检索 query。"""
        parts = [q.question.strip(), "选项："]
        for k in sorted(q.options.keys()):
            parts.append(f"{k}. {q.options[k].strip()}")
        return "\n".join(parts).strip()

    def _build_messages(self, q: Question, context: str) -> list[ChatMessage]:
        """构造 Chat messages。"""
        system = (
            "你是金融长文档问答助手。"
            "你必须严格基于提供的证据片段作答。"
            "不要输出解释、不要输出除答案字母以外的内容。"
        )
        user = self._format_user_prompt(q=q, context=context)
        return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]

    def _format_user_prompt(self, q: Question, context: str) -> str:
        """构造用户 prompt。"""
        fmt = q.answer_format
        format_hint = {
            AnswerFormat.MCQ: "单选题：只输出一个大写字母（A/B/C/D）。",
            AnswerFormat.TF: "判断题：只输出一个大写字母（A 或 B）。",
            AnswerFormat.MULTI: "多选题：输出多个大写字母，按字母顺序排列，不使用分隔符（例如 ABC）。",
        }[fmt]

        options_text = "\n".join(f"{k}. {v}" for k, v in sorted(q.options.items()))
        return (
            f"{format_hint}\n\n"
            f"题目：{q.question}\n\n"
            f"选项：\n{options_text}\n\n"
            f"证据片段：\n{context}\n\n"
            "请给出最终答案："
        )

    def _normalize_answer(self, fmt: AnswerFormat, text: str) -> str:
        """从模型输出中抽取并规范化答案。"""
        cleaned = (text or "").strip().upper()
        letters = re.findall(r"[A-D]", cleaned)

        if fmt in {AnswerFormat.MCQ, AnswerFormat.TF}:
            return letters[0] if letters else "A"

        if fmt == AnswerFormat.MULTI:
            unique = sorted(set(letters))
            return "".join(unique) if unique else "A"

        return "A"

    def _retrieve_evidence(self, domain: str, doc_ids: list[str], query: str) -> list[EvidenceSnippet]:
        """对候选文档做无向量检索，返回 top-k 证据片段。"""
        top_k = self._retrieval.top_k_chunks
        per_doc_top_k = max(1, self._retrieval.per_doc_top_k)

        all_hits: list[EvidenceSnippet] = []
        for doc_id in doc_ids:
            try:
                text = self._docs.load_text(domain=domain, doc_id=doc_id)
            except Exception as exc:
                logger.warning("文档读取失败：%s/%s: %s", domain, doc_id, repr(exc))
                continue

            chunks = chunk_text(text, max_chars=self._retrieval.chunk_max_chars)
            scores = bm25_rank(query=query, chunks=chunks)
            for idx, score in scores[:per_doc_top_k]:
                all_hits.append(
                    EvidenceSnippet(
                        doc_id=doc_id,
                        content=chunks[idx].strip(),
                        score=float(score),
                        chunk_id=f"{doc_id}::chunk{idx}",
                    )
                )

        all_hits.sort(key=lambda x: x.score, reverse=True)
        return all_hits[:top_k]

    def _build_context(self, query: str, evidence: list[EvidenceSnippet]) -> str:
        """拼接证据并做长度压缩。"""
        parts: list[str] = []
        for e in evidence:
            parts.append(f"[{e.doc_id}] {e.content}")
        combined = "\n\n".join(parts).strip()

        if len(combined) <= self._retrieval.max_context_chars:
            return combined

        return compress_text_by_overlap(
            query=query,
            text=combined,
            max_chars=self._retrieval.max_context_chars,
        )


def chunk_text(text: str, max_chars: int) -> list[str]:
    """将长文本切分为 chunks。"""
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    paras = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    if not paras:
        return [normalized.strip()]

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in paras:
        add_len = len(p) + (2 if buf else 0)
        if buf and buf_len + add_len > max_chars:
            chunks.append("\n\n".join(buf).strip())
            buf = [p]
            buf_len = len(p)
        else:
            buf.append(p)
            buf_len += add_len
    if buf:
        chunks.append("\n\n".join(buf).strip())
    return chunks


def tokenize(text: str) -> list[str]:
    """对中英混合文本做轻量 tokenization（非 embedding）。"""
    text = (text or "").lower()
    tokens: list[str] = []
    for part in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            tokens.extend(list(part))
        else:
            tokens.append(part)
    return tokens


def bm25_rank(query: str, chunks: list[str]) -> list[tuple[int, float]]:
    """对 chunks 使用 BM25 打分并排序。"""
    q_terms = tokenize(query)
    if not q_terms:
        return [(i, 0.0) for i in range(len(chunks))]

    docs_terms = [tokenize(c) for c in chunks]
    doc_lens = [len(t) for t in docs_terms]
    avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0

    df: dict[str, int] = {}
    for terms in docs_terms:
        for t in set(terms):
            df[t] = df.get(t, 0) + 1

    n = len(chunks)
    k1 = 1.5
    b = 0.75

    def idf(t: str) -> float:
        dft = df.get(t, 0)
        return math.log(1 + (n - dft + 0.5) / (dft + 0.5))

    scores: list[tuple[int, float]] = []
    q_tf: dict[str, int] = {}
    for t in q_terms:
        q_tf[t] = q_tf.get(t, 0) + 1

    for i, terms in enumerate(docs_terms):
        tf: dict[str, int] = {}
        for t in terms:
            tf[t] = tf.get(t, 0) + 1
        dl = doc_lens[i] or 1
        denom_base = k1 * (1 - b + b * (dl / (avgdl or 1.0)))
        score = 0.0
        for t in q_tf.keys():
            f = tf.get(t, 0)
            if f <= 0:
                continue
            score += idf(t) * (f * (k1 + 1)) / (f + denom_base)
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def compress_text_by_overlap(query: str, text: str, max_chars: int) -> str:
    """按 query 的 token overlap 做抽取式压缩。"""
    q_terms = set(tokenize(query))
    if not q_terms:
        return (text or "")[:max_chars]

    sentences = re.split(r"(?<=[。！？!?\n])\s*", text)
    scored: list[tuple[float, str]] = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        t = tokenize(s)
        if not t:
            continue
        overlap = sum(1 for x in t if x in q_terms)
        scored.append((float(overlap) / (len(t) or 1), s))

    scored.sort(key=lambda x: x[0], reverse=True)
    buf: list[str] = []
    total = 0
    for _, s in scored:
        if total + len(s) + 1 > max_chars:
            continue
        buf.append(s)
        total += len(s) + 1
        if total >= max_chars:
            break
    return "\n".join(buf).strip() or (text or "")[:max_chars]


def write_answer_csv(path: Path, records: list[AnswerRecord]) -> None:
    """写入符合提交格式的 answer.csv。"""
    import csv

    total_prompt = sum(r.usage.prompt_tokens for r in records)
    total_completion = sum(r.usage.completion_tokens for r in records)
    total_total = sum(r.usage.total_tokens for r in records)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])
        writer.writerow(["summary", "", total_prompt, total_completion, total_total])
        for r in records:
            writer.writerow([r.qid, r.answer, r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.total_tokens])


def write_evidence_jsonl(path: Path, q: Question, evidence: list[EvidenceSnippet]) -> None:
    """追加写入单题证据到 jsonl（可选调试）。"""
    payload = {
        "qid": q.qid,
        "domain": q.domain,
        "answer_format": q.answer_format.value,
        "evidence_retrieval": [
            {
                "doc_id": e.doc_id,
                "chunk_id": e.chunk_id,
                "score": e.score,
                "quoted_clause": e.content,
            }
            for e in evidence
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_evaluation(run: RunConfig, llm: LlmConfig, retrieval: RetrievalConfig) -> EvaluationResult:
    """按配置执行评测并产出 answer.csv。"""
    questions_dir = run.dataset_root / run.questions_subdir
    raw_docs_root = run.dataset_root / run.raw_docs_subdir

    q_repo = QuestionRepository(questions_dir=questions_dir)
    d_repo = DocumentRepository(raw_root=raw_docs_root)

    client = OpenAiCompatibleChatClient(llm)
    agent = FinanceLongTextAgent(llm=client, docs=d_repo, retrieval=retrieval)

    questions = q_repo.load_questions(split=run.split)
    logger.info("加载题目：%s（split=%s）", len(questions), run.split)

    records: list[AnswerRecord] = []
    total_prompt = 0
    total_completion = 0

    for idx, q in enumerate(questions, start=1):
        logger.info("作答中：%s (%s/%s)", q.qid, idx, len(questions))
        record, evidence = agent.answer(q)
        records.append(record)
        total_prompt += record.usage.prompt_tokens
        total_completion += record.usage.completion_tokens

        if run.output_evidence_jsonl is not None:
            write_evidence_jsonl(run.output_evidence_jsonl, q=q, evidence=evidence)

    write_answer_csv(run.output_csv, records=records)
    total_usage = TokenUsage(
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        total_tokens=total_prompt + total_completion,
    )
    return EvaluationResult(answers=records, total_usage=total_usage)
