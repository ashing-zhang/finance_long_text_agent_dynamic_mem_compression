from __future__ import annotations

import logging
import tempfile
from fin_agent.compat import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MarkdownSection:
    title: str
    content: str
    metadata: dict[str, str]


class MineruMarkdownConverter:
    """使用 MinerU 将 PDF 转为 Markdown。"""

    def convert_pdf_to_markdown(self, pdf_path: Path) -> str:
        do_parse, read_fn = _import_mineru()
        pdf_bytes = read_fn(pdf_path)
        with tempfile.TemporaryDirectory(prefix="mineru_md_") as temp_dir:
            output_dir = Path(temp_dir)
            do_parse(
                output_dir=str(output_dir),
                pdf_file_names=[pdf_path.stem],
                pdf_bytes_list=[pdf_bytes],
                p_lang_list=["ch"],
                backend="pipeline",
                parse_method="auto",
                formula_enable=True,
                table_enable=True,
                start_page_id=0,
                end_page_id=None,
            )
            md_path = _find_markdown_output(output_dir=output_dir, stem=pdf_path.stem)
            if md_path is None:
                raise RuntimeError(f"MinerU 未产出 Markdown：{pdf_path}")
            return md_path.read_text(encoding="utf-8", errors="ignore")


def split_markdown_by_headers(markdown_text: str) -> list[MarkdownSection]:
    MarkdownHeaderTextSplitter = _import_markdown_splitter()
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
            ("######", "h6"),
        ],
        strip_headers=False,
    )
    documents = splitter.split_text(markdown_text or "")
    sections: list[MarkdownSection] = []
    for doc in documents:
        content = (getattr(doc, "page_content", "") or "").strip()
        metadata = {str(k): str(v) for k, v in (getattr(doc, "metadata", {}) or {}).items()}
        title = _build_section_title(metadata=metadata, content=content)
        if not content:
            continue
        sections.append(MarkdownSection(title=title, content=content, metadata=metadata))
    return sections


def extract_markdown_headings(markdown_text: str, max_items: int) -> list[str]:
    headings: list[str] = []
    for raw_line in (markdown_text or "").splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            headings.append(line)
        if len(headings) >= max_items:
            break
    return headings


def _build_section_title(metadata: dict[str, str], content: str) -> str:
    header_values = [metadata.get(f"h{i}", "").strip() for i in range(1, 7)]
    header_values = [item for item in header_values if item]
    if header_values:
        return " / ".join(header_values)[:120]
    for raw_line in content.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#"):
            return stripped[:120]
    return "markdown_section"


def _find_markdown_output(output_dir: Path, stem: str) -> Path | None:
    preferred = list(output_dir.glob(f"**/{stem}.md"))
    if preferred:
        return preferred[0]
    candidates = list(output_dir.glob("**/*.md"))
    return candidates[0] if candidates else None


def _import_mineru():
    try:
        from mineru.cli.common import do_parse, read_fn
    except Exception as exc:
        raise RuntimeError(
            "需要安装 MinerU 才能将 PDF 转为 Markdown。建议安装：pip install mineru"
        ) from exc
    return do_parse, read_fn


def _import_markdown_splitter():
    try:
        from langchain_text_splitters import MarkdownHeaderTextSplitter
        return MarkdownHeaderTextSplitter
    except Exception:
        try:
            from langchain.text_splitter import MarkdownHeaderTextSplitter
            return MarkdownHeaderTextSplitter
        except Exception as exc:
            raise RuntimeError(
                "需要安装 LangChain Markdown 标题切分器。建议安装：pip install langchain-text-splitters"
            ) from exc
