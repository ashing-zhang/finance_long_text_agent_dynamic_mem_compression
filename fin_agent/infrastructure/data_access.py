from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from fin_agent.domain.models import AnswerFormat, Question

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DocumentRef:
    """文档引用：domain + doc_id -> file path。"""

    domain: str
    doc_id: str
    path: Path


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
    """从 raw 目录按 doc_id 加载文档文本。"""

    def __init__(self, raw_root: Path) -> None:
        """初始化仓库。"""
        self._raw_root = raw_root
        self._cache: dict[tuple[str, str], str] = {}

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
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        ref = self.resolve(domain=domain, doc_id=doc_id)
        text = self._read_document(ref.path)
        self._cache[cache_key] = text
        return text

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

        for p in domain_dir.glob("*.pdf"):
            doc_ids.add(p.stem)
        for p in domain_dir.glob("*.PDF"):
            doc_ids.add(p.stem)
        return sorted(doc_ids)

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
        html = re.sub(r"<[^>]+>", " ", html)
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
