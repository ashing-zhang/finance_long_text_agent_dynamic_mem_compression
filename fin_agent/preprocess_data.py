"""
运行指南（模块化）：

1) （推荐）准备配置文件并指定：
   - FIN_AGENT_CONFIG=./configs/agent.toml
   - FIN_AGENT_DOTENV=./.env（可选）

2) 执行预处理：
   - python -m fin_agent.preprocess_data

说明：
- 该模块会扫描 dataset_root/raw_docs_subdir 下的所有 PDF，并尝试使用 MinerU 转为 Markdown；
- Markdown 会统一输出到 raw/<domain>/md/（可配置），文件名与 doc_id 相同，例如：
  - raw/financial_contracts/text01.pdf -> raw/financial_contracts/md/text01.md
  - raw/regulatory/attachments/xxx.pdf -> raw/regulatory/md/xxx.md
- 若转换失败（缺依赖或解析异常），会记录告警并跳过该文件。
"""

from __future__ import annotations

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
    pdf_paths = find_pdfs(raw_root)
    logger.info("发现 PDF：%s（root=%s）", len(pdf_paths), raw_root)

    processed = 0
    skipped = 0
    failed = 0

    for idx, pdf_path in enumerate(pdf_paths, start=1):
        if cfg.max_files > 0 and processed + skipped + failed >= cfg.max_files:
            break
        out_path = build_output_path(raw_root=raw_root, pdf_path=pdf_path, output_subdir=cfg.output_subdir)
        if out_path.exists() and not cfg.overwrite:
            skipped += 1
            continue
        try:
            markdown = converter.convert_pdf_to_markdown(pdf_path)
            if not markdown.strip():
                raise RuntimeError("MinerU 输出为空")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(markdown, encoding="utf-8", errors="ignore")
            processed += 1
            if processed % 10 == 0 or idx == len(pdf_paths):
                logger.info("进度：%s/%s processed=%s skipped=%s failed=%s", idx, len(pdf_paths), processed, skipped, failed)
        except ModuleNotFoundError as exc:
            failed += 1
            missing = getattr(exc, "name", None) or str(exc)
            logger.warning("转换失败（缺少依赖）：pdf=%s missing=%s", pdf_path, missing)
        except Exception as exc:
            failed += 1
            logger.warning("转换失败：pdf=%s error=%s", pdf_path, repr(exc))

    logger.info("完成：processed=%s skipped=%s failed=%s", processed, skipped, failed)


def configure_logging(level: str) -> None:
    """初始化日志输出格式。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def find_pdfs(root: Path) -> list[Path]:
    """递归扫描目录并返回所有 PDF 路径。"""
    pdfs: list[Path] = []
    for pattern in ("*.pdf", "*.PDF"):
        pdfs.extend(root.rglob(pattern))
    pdfs.sort(key=lambda p: str(p))
    return pdfs


def build_output_path(raw_root: Path, pdf_path: Path, output_subdir: str) -> Path:
    """根据 pdf_path 推断 domain，并构造输出 md 的目标路径。"""
    try:
        rel = pdf_path.relative_to(raw_root)
        domain = rel.parts[0] if rel.parts else "unknown"
    except Exception:
        domain = "unknown"
    return raw_root / domain / output_subdir / f"{pdf_path.stem}.md"


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
