from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from fin_agent.domain.models import Question

logger = logging.getLogger(__name__)

HEADING_PATTERNS = (
    r"^第[一二三四五六七八九十百千万0-9]+[编章节条款]\s*.*$",
    r"^[一二三四五六七八九十]+[、.]\s*.*$",
    r"^[(（]?[一二三四五六七八九十0-9]+[)）][、.]?\s*.*$",
    r"^[A-D][、.]\s*.*$",
    r"^\d+(\.\d+){0,3}\s+.*$",
)


@dataclass(frozen=True, slots=True)
class DocumentRef:
    """文档引用：domain + doc_id -> file path。"""

    domain: str
    doc_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class StructuredChunk:
    """结构化分块结果。"""

    doc_id: str
    title: str
    content: str
    chunk_id: str
    order: int

    def to_index_text(self) -> str:
        """返回适合索引与检索的文本。"""
        return f"[DocID: {self.doc_id} | Title: {self.title}] {self.content}".strip()


class QuestionRepository:
    """从 questions 目录加载题目。"""

    def __init__(self, questions_dir: Path) -> None:
        """初始化仓库。"""
        self._questions_dir = questions_dir

    def load_questions(self, split: str) -> list[Question]:
        """加载指定 split（A/B）的全部题目。"""
        if not self._questions_dir.exists():
            raise FileNotFoundError(f"questions 目录不存在：{self._questions_dir}")

        questions: list[Question] = []
        for file_path in sorted(self._questions_dir.glob("*.json")):
            items = json.loads(file_path.read_text(encoding="utf-8"))
            for item in items:
                if (item.get("split") or "").upper() != split.upper():
                    continue
                questions.append(self._to_question(item))
        return questions

    def _to_question(self, item: dict) -> Question:
        """将原始 dict 转为领域模型。"""
        answer_format = AnswerFormat(str(item["answer_format"]))
        options = {str(k): str(v) for k, v in (item.get("options") or {}).items()}
        doc_ids = item.get("doc_ids")
        if doc_ids is not None:
            doc_ids = [str(d) for d in doc_ids]
        return Question(
            qid=str(item["qid"]),
            domain=str(item["domain"]),
            split=str(item["split"]),
            question=str(item["question"]),
            options=options,
            answer_format=answer_format,
            type=(str(item["type"]) if item.get("type") is not None else None),
            doc_ids=doc_ids,
        )


class DocumentRepository:
    """从 raw 目录按 doc_id 加载文档文本与结构化 chunks。"""

    def __init__(self, raw_root: Path) -> None:
        """初始化仓库。"""
        self._raw_root = raw_root
        self._text_cache: dict[tuple[str, str], str] = {}
        self._chunk_cache: dict[tuple[str, str, int], list[StructuredChunk]] = {}

    def resolve(self, domain: str, doc_id: str) -> DocumentRef:
        """定位 doc_id 对应的文件路径。"""
        domain_dir = self._raw_root / domain
        if domain == "regulatory":
            txt_path = domain_dir / "txt" / f"{doc_id}.txt"
            if txt_path.exists():
                return DocumentRef(domain=domain, doc_id=doc_id, path=txt_path)

        candidates = list(domain_dir.glob(f"{doc_id}.*"))
        if not candidates and domain == "regulatory":
            candidates = list((domain_dir / "html").glob(f"{doc_id}.html"))
        if not candidates:
            raise FileNotFoundError(f"找不到文档：domain={domain} doc_id={doc_id}")

        candidates.sort(key=lambda p: p.suffix.lower())
        return DocumentRef(domain=domain, doc_id=doc_id, path=candidates[0])

    def load_text(self, domain: str, doc_id: str) -> str:
        """读取并返回文本内容（带缓存）。"""
        cache_key = (domain, doc_id)
        cached = self._text_cache.get(cache_key)
        if cached is not None:
            return cached

        ref = self.resolve(domain=domain, doc_id=doc_id)
        text = self._read_document(ref.path)
        self._text_cache[cache_key] = text
        return text

    def load_chunks(self, domain: str, doc_id: str, max_chars: int) -> list[StructuredChunk]:
        """读取并返回结构化 chunks（带缓存）。"""
        cache_key = (domain, doc_id, max_chars)
        cached = self._chunk_cache.get(cache_key)
        if cached is not None:
            return cached

        text = self.load_text(domain=domain, doc_id=doc_id)
        chunks = self._build_structured_chunks(doc_id=doc_id, text=text, max_chars=max_chars)
        self._chunk_cache[cache_key] = chunks
        return chunks

    def load_doc_profile(self, domain: str, doc_id: str, window_chars: int) -> str:
        """加载用于文档级粗筛的 profile 文本。"""
        text = self.load_text(domain=domain, doc_id=doc_id)
        title = self.infer_doc_title(domain=domain, doc_id=doc_id, text=text)
        outline = self.build_outline(domain=domain, doc_id=doc_id, max_items=6)
        body_head = re.sub(r"\s+", " ", text[:window_chars]).strip()
        return f"{doc_id}\n{title}\n{outline}\n{body_head}".strip()

    def infer_doc_title(self, domain: str, doc_id: str, text: str | None = None) -> str:
        """推断文档标题。"""
        if text is None:
            text = self.load_text(domain=domain, doc_id=doc_id)
        for line in text.splitlines():
            stripped = line.strip()
            if 4 <= len(stripped) <= 120:
                return stripped
        return doc_id

    def build_outline(self, domain: str, doc_id: str, max_items: int = 8) -> str:
        """提取文档核心大纲，供回退路由使用。"""
        text = self.load_text(domain=domain, doc_id=doc_id)
        headings: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and is_heading(stripped):
                headings.append(stripped)
            if len(headings) >= max_items:
                break
        return "\n".join(headings)

    def list_doc_ids(self, domain: str) -> list[str]:
        """列出某领域下可用的 doc_id（用于 B 组未知 doc_ids 的回退策略）。"""
        domain_dir = self._raw_root / domain
        if not domain_dir.exists():
            return []

        doc_ids: set[str] = set()
        if domain == "regulatory" and (domain_dir / "txt").exists():
            for p in (domain_dir / "txt").glob("*.txt"):
                doc_ids.add(p.stem)
            return sorted(doc_ids)

        for pattern in ("*.pdf", "*.PDF", "*.txt", "*.TXT"):
            for p in domain_dir.glob(pattern):
                doc_ids.add(p.stem)
        return sorted(doc_ids)

    def _build_structured_chunks(self, doc_id: str, text: str, max_chars: int) -> list[StructuredChunk]:
        """按标题/条款优先的策略构建结构化 chunks。"""
        normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in normalized.splitlines()]
        blocks: list[tuple[str, list[str]]] = []
        current_title = doc_id
        current_lines: list[str] = []

        for line in lines:
            if not line:
                if current_lines and current_lines[-1] != "":
                    current_lines.append("")
                continue
            if is_heading(line):
                if current_lines:
                    blocks.append((current_title, current_lines))
                    current_lines = []
                current_title = line[:120]
                current_lines = [line]
                continue
            current_lines.append(line)

        if current_lines:
            blocks.append((current_title, current_lines))

        chunks: list[StructuredChunk] = []
        order = 0
        for title, block_lines in blocks:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", "\n".join(block_lines)) if p.strip()]
            if not paragraphs:
                continue

            buffer: list[str] = []
            buffer_len = 0
            for para in paragraphs:
                para_len = len(para) + (2 if buffer else 0)
                if buffer and buffer_len + para_len > max_chars:
                    content = "\n\n".join(buffer).strip()
                    chunks.append(
                        StructuredChunk(
                            doc_id=doc_id,
                            title=title,
                            content=content,
                            chunk_id=f"{doc_id}::chunk{order}",
                            order=order,
                        )
                    )
                    order += 1
                    buffer = [para]
                    buffer_len = len(para)
                else:
                    buffer.append(para)
                    buffer_len += para_len
            if buffer:
                content = "\n\n".join(buffer).strip()
                chunks.append(
                    StructuredChunk(
                        doc_id=doc_id,
                        title=title,
                        content=content,
                        chunk_id=f"{doc_id}::chunk{order}",
                        order=order,
                    )
                )
                order += 1

        if not chunks:
            chunks.append(
                StructuredChunk(
                    doc_id=doc_id,
                    title=doc_id,
                    content=normalized[:max_chars].strip(),
                    chunk_id=f"{doc_id}::chunk0",
                    order=0,
                )
            )
        return chunks

    def _read_document(self, path: Path) -> str:
        """按文件类型读取文档并返回文本。"""
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".html":
            html = path.read_text(encoding="utf-8", errors="ignore")
            return self._strip_html(html)
        if suffix == ".pdf":
            return self._extract_pdf_text(path)
        raise ValueError(f"不支持的文档类型：{path}")

    def _strip_html(self, html: str) -> str:
        """将 HTML 粗略转为纯文本。"""
        html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"<[^>]+>", "\n", html)
        html = re.sub(r"[ \t\r\f\v]+", " ", html)
        html = re.sub(r"\n{2,}", "\n\n", html)
        return html.strip()

    def _extract_pdf_text(self, path: Path) -> str:
        """从 PDF 提取文本（可选依赖 pypdf 或 PyPDF2）。"""
        try:
            from pypdf import PdfReader
        except Exception:
            try:
                from PyPDF2 import PdfReader
            except Exception as exc:
                raise RuntimeError(
                    "PDF 文本抽取需要安装 pypdf 或 PyPDF2。"
                    "建议：pip install pypdf"
                ) from exc

        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        return "\n".join(parts).strip()


def is_heading(line: str) -> bool:
    """判断某行是否可视为标题或条款起始。"""
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > 120:
        return False
    return any(re.match(pattern, stripped) for pattern in HEADING_PATTERNS)
