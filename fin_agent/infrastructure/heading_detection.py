from __future__ import annotations

import logging
import tempfile
from fin_agent.compat import dataclass
from pathlib import Path
import re

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
    """按 Markdown 标题（#..######）切分为 sections。"""
    lines = (markdown_text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
    heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    stack: list[tuple[int, str]] = []
    buffer: list[str] = []
    sections: list[MarkdownSection] = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if not content:
            return
        metadata = {f"h{level}": title for level, title in stack}
        title = " / ".join([t for _, t in stack])[:120] if stack else "markdown_section"
        sections.append(MarkdownSection(title=title or "markdown_section", content=content, metadata=metadata))

    for raw_line in lines:
        line = raw_line.rstrip()
        match = heading_re.match(line.strip())
        if match:
            flush()
            buffer = [line]
            level = len(match.group(1))
            title_text = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title_text))
            continue
        buffer.append(line)

    flush()
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
