"""
运行指南（模块化）：

1) （推荐）准备配置文件并指定：
   - FIN_AGENT_CONFIG=./configs/agent.toml
   - FIN_AGENT_DOTENV=./.env（可选）

2) 执行预处理：
   - python -m fin_agent.preprocess_data

说明：
- 该模块会扫描 dataset_root/raw_docs_subdir 下的所有 PDF，并尝试使用 MinerU 转为 Markdown；
- 对 `regulatory` domain，还会额外扫描 `html/` 与 `txt/` 目录，并统一转换为 `md/` 下的 Markdown 文件；
- Markdown 会统一输出到 raw/<domain>/md/（可配置），文件名与 doc_id 相同，例如：
  - raw/financial_contracts/text01.pdf -> raw/financial_contracts/md/text01.md
  - raw/regulatory/attachments/xxx.pdf -> raw/regulatory/md/xxx.md
- 若转换失败（缺依赖或解析异常），会记录告警并跳过该文件。
"""

from __future__ import annotations

import html
import logging
import os
from pathlib import Path
import re

from fin_agent.compat import dataclass
from fin_agent.infrastructure.heading_detection import MineruMarkdownConverter
from fin_agent.run import load_app_config, load_dotenv, load_toml

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """预处理配置。"""

    dataset_root: Path
    raw_docs_subdir: Path
    output_subdir: str = "md"
    overwrite: bool = False
    max_files: int = 0


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """待转换的原始文档描述。"""

    domain: str
    path: Path
    source_type: str
    output_priority: int


def main() -> None:
    """模块化入口：批量将 data 目录下 PDF 转为 Markdown。"""
    load_dotenv(Path(os.getenv("FIN_AGENT_DOTENV", ".env")))
    config_path = Path(os.getenv("FIN_AGENT_CONFIG", "configs/agent.toml"))
    app = load_app_config(config_path)

    raw = load_toml(config_path) if config_path.exists() else {}
    preprocess = raw.get("preprocess", {}) if isinstance(raw, dict) else {}
    output_subdir = _sanitize_output_subdir(str(preprocess.get("output_subdir", "md")))
    overwrite = bool(preprocess.get("overwrite", False))
    max_files = int(preprocess.get("max_files", 0))
    cfg = PreprocessConfig(
        dataset_root=app.run.dataset_root,
        raw_docs_subdir=app.run.raw_docs_subdir,
        output_subdir=output_subdir,
        overwrite=overwrite,
        max_files=max_files,
    )

    configure_logging(level=os.getenv("FIN_AGENT_PREPROCESS_LOG_LEVEL", "INFO"))

    raw_root = cfg.dataset_root / cfg.raw_docs_subdir
    if not raw_root.exists():
        logger.warning("raw 目录不存在：%s", raw_root)
        return

    converter = MineruMarkdownConverter()
    source_docs = find_source_documents(raw_root)
    logger.info("发现待转换文档：%s（root=%s）", len(source_docs), raw_root)

    processed = 0
    skipped = 0
    failed = 0

    for idx, source_doc in enumerate(source_docs, start=1):
        if cfg.max_files > 0 and processed + skipped + failed >= cfg.max_files:
            break
        out_path = build_output_path(raw_root=raw_root, source_path=source_doc.path, output_subdir=cfg.output_subdir)
        if out_path.exists() and not cfg.overwrite:
            skipped += 1
            continue
        try:
            markdown = convert_source_to_markdown(converter=converter, source_doc=source_doc)
            if not markdown.strip():
                raise RuntimeError("Markdown 输出为空")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(markdown, encoding="utf-8", errors="ignore")
            processed += 1
            if processed % 10 == 0 or idx == len(source_docs):
                logger.info(
                    "进度：%s/%s processed=%s skipped=%s failed=%s",
                    idx,
                    len(source_docs),
                    processed,
                    skipped,
                    failed,
                )
        except ModuleNotFoundError as exc:
            failed += 1
            missing = getattr(exc, "name", None) or str(exc)
            logger.warning("转换失败（缺少依赖）：source=%s missing=%s", source_doc.path, missing)
        except Exception as exc:
            failed += 1
            logger.warning("转换失败：source=%s type=%s error=%s", source_doc.path, source_doc.source_type, repr(exc))

    logger.info("完成：processed=%s skipped=%s failed=%s", processed, skipped, failed)


def configure_logging(level: str) -> None:
    """初始化日志输出格式。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def find_source_documents(root: Path) -> list[SourceDocument]:
    """收集所有待转换文档，并保证同名输出时处理顺序稳定。"""
    docs: list[SourceDocument] = []
    for pdf_path in _glob_case_insensitive(root, "pdf"):
        docs.append(
            SourceDocument(
                domain=_infer_domain(raw_root=root, source_path=pdf_path),
                path=pdf_path,
                source_type="pdf",
                output_priority=2,
            )
        )

    regulatory_root = root / "regulatory"
    if regulatory_root.exists():
        for txt_path in _glob_case_insensitive(regulatory_root / "txt", "txt"):
            docs.append(
                SourceDocument(
                    domain="regulatory",
                    path=txt_path,
                    source_type="txt",
                    output_priority=0,
                )
            )
        for html_path in _glob_case_insensitive(regulatory_root / "html", "html"):
            docs.append(
                SourceDocument(
                    domain="regulatory",
                    path=html_path,
                    source_type="html",
                    output_priority=1,
                )
            )

    docs.sort(
        key=lambda item: (
            str(build_output_path(raw_root=root, source_path=item.path, output_subdir="md")),
            item.output_priority,
            str(item.path),
        )
    )
    return docs


def convert_source_to_markdown(converter: MineruMarkdownConverter, source_doc: SourceDocument) -> str:
    """按源文件类型执行 Markdown 转换。"""
    if source_doc.source_type == "pdf":
        return converter.convert_pdf_to_markdown(source_doc.path)
    if source_doc.source_type == "txt":
        return convert_txt_to_markdown(source_doc.path)
    if source_doc.source_type == "html":
        return convert_html_to_markdown(source_doc.path)
    raise ValueError(f"不支持的 source_type: {source_doc.source_type}")


def convert_txt_to_markdown(txt_path: Path) -> str:
    """将 txt 文本规范化为 Markdown 文本。"""
    text = txt_path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.splitlines()]
    body = "\n".join(lines).strip()
    if not body:
        return ""
    return f"# {txt_path.stem}\n\n{body}\n"


def convert_html_to_markdown(html_path: Path) -> str:
    """将 html 文本做轻量结构化清洗后转为 Markdown。"""
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"(?is)<(script|style)\b.*?>.*?</\1>", "\n", raw)
    replacements = {
        r"(?i)<br\s*/?>": "\n",
        r"(?i)</p>": "\n\n",
        r"(?i)</div>": "\n",
        r"(?i)</tr>": "\n",
        r"(?i)</li>": "\n",
        r"(?i)<li\b[^>]*>": "- ",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    for level in range(1, 7):
        text = re.sub(
            rf"(?is)<h{level}\b[^>]*>(.*?)</h{level}>",
            lambda match: "\n" + ("#" * level) + " " + _strip_html_fragment(match.group(1)) + "\n\n",
            text,
        )
    text = re.sub(r"(?is)<[^>]+>", "", text)
    text = html.unescape(text)
    lines = [_normalize_text_line(line) for line in text.splitlines()]
    compact_lines = _compact_lines(lines)
    body = "\n".join(compact_lines).strip()
    if not body:
        return ""
    if not body.startswith("#"):
        body = f"# {html_path.stem}\n\n{body}"
    return body.rstrip() + "\n"


def build_output_path(raw_root: Path, source_path: Path, output_subdir: str) -> Path:
    """根据源文件路径推断 domain，并构造输出 md 目标路径。"""
    domain = _infer_domain(raw_root=raw_root, source_path=source_path)
    return raw_root / domain / output_subdir / f"{source_path.stem}.md"


def _infer_domain(raw_root: Path, source_path: Path) -> str:
    """从源文件相对路径中提取 domain。"""
    try:
        rel = source_path.relative_to(raw_root)
        return rel.parts[0] if rel.parts else "unknown"
    except Exception:
        return "unknown"


def _glob_case_insensitive(root: Path, suffix: str) -> list[Path]:
    """收集同一后缀的大小写文件。"""
    if not root.exists():
        return []
    results: list[Path] = []
    for pattern in (f"*.{suffix.lower()}", f"*.{suffix.upper()}"):
        results.extend(root.rglob(pattern))
    deduped = sorted({path.resolve(): path for path in results}.values(), key=lambda item: str(item))
    return deduped


def _strip_html_fragment(text: str) -> str:
    """移除 HTML 片段中的标签并反转义。"""
    without_tags = re.sub(r"(?is)<[^>]+>", "", text or "")
    return _normalize_text_line(html.unescape(without_tags))


def _normalize_text_line(text: str) -> str:
    """压缩单行文本中的多余空白。"""
    return re.sub(r"\s+", " ", text or "").strip()


def _compact_lines(lines: list[str]) -> list[str]:
    """压缩空行，保留 Markdown 基本段落结构。"""
    compacted: list[str] = []
    prev_blank = False
    for line in lines:
        if not line:
            if compacted and not prev_blank:
                compacted.append("")
            prev_blank = True
            continue
        compacted.append(line)
        prev_blank = False
    while compacted and not compacted[-1]:
        compacted.pop()
    return compacted


def _sanitize_output_subdir(value: str) -> str:
    """清洗输出子目录名，避免非法路径片段。"""
    stripped = (value or "").strip().strip("/").strip("\\")
    if not stripped:
        return "md"
    if stripped in {".", ".."}:
        return "md"
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,32}", stripped):
        return "md"
    return stripped


if __name__ == "__main__":
    main()
