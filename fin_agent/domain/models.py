from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class AnswerFormat(StrEnum):
    """题型枚举。"""

    MCQ = "mcq"
    MULTI = "multi"
    TF = "tf"


@dataclass(frozen=True, slots=True)
class Question:
    """评测题目领域模型。"""

    qid: str
    domain: str
    split: str
    question: str
    options: dict[str, str]
    answer_format: AnswerFormat
    type: str | None = None
    doc_ids: list[str] | None = None


@dataclass(frozen=True, slots=True)
class EvidenceSnippet:
    """用于可追溯的证据片段。"""

    doc_id: str
    content: str
    score: float
    chunk_id: str


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """单次调用的 token 消耗。"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class AnswerRecord:
    """单题输出记录，用于写入 answer.csv。"""

    qid: str
    answer: str
    usage: TokenUsage


@dataclass(frozen=True, slots=True)
class LlmConfig:
    """大模型调用配置（OpenAI-compatible）。"""

    base_url: str
    model: str
    api_key_env: str
    timeout_seconds: float = 120.0
    max_retries: int = 2


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """检索与压缩相关配置。"""

    chunk_max_chars: int = 1600
    top_k_chunks: int = 6
    max_context_chars: int = 8000
    per_doc_top_k: int = 3


@dataclass(frozen=True, slots=True)
class RunConfig:
    """运行配置。"""

    dataset_root: Path
    questions_subdir: Path
    raw_docs_subdir: Path
    split: str = "A"
    output_csv: Path = Path("answer.csv")
    output_evidence_jsonl: Path | None = None
