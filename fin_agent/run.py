"""
运行指南（模块化）：

1) 配置环境变量（建议）
   - DASHSCOPE_API_KEY：阿里云 DashScope / Model Studio 的 API Key

2) （可选）创建配置文件（TOML），并通过环境变量指定
   - FIN_AGENT_CONFIG=./configs/agent.toml

3) 运行
   - python -m fin_agent.run

说明：
- 默认使用 OpenAI-compatible Chat Completions 接口；
- 默认 base_url 为 https://dashscope.aliyuncs.com/compatible-mode/v1
  可在配置文件中覆盖该值。
"""

from __future__ import annotations

import ast
import logging
import os
from pathlib import Path

from fin_agent.compat import dataclass
from fin_agent.application.agent import configure_logging, run_evaluation
from fin_agent.domain.models import LlmConfig, RetrievalConfig, RunConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AppConfig:
    """聚合应用配置。"""

    run: RunConfig
    llm: LlmConfig
    retrieval: RetrievalConfig
    log_level: str = "INFO"


def main() -> None:
    """模块化入口。"""
    load_dotenv(Path(os.getenv("FIN_AGENT_DOTENV", ".env")))
    config_path = Path(os.getenv("FIN_AGENT_CONFIG", "configs/agent.toml"))
    app = load_app_config(config_path)
    configure_logging(app.log_level)

    logger.info("开始评测：output_csv=%s", app.run.output_csv)
    result = run_evaluation(run=app.run, llm=app.llm, retrieval=app.retrieval)
    logger.info(
        "完成：题数=%s total_tokens=%s (prompt=%s completion=%s)",
        len(result.answers),
        result.total_usage.total_tokens,
        result.total_usage.prompt_tokens,
        result.total_usage.completion_tokens,
    )


def load_app_config(path: Path) -> AppConfig:
    """从 TOML 配置文件加载配置，不存在则使用默认值。"""
    data = {}
    if path.exists():
        data = load_toml(path)
    else:
        data = {}

    dataset_root = Path(data.get("run", {}).get("dataset_root", "data/public_dataset_a/public_dataset_upload"))
    questions_subdir = Path(data.get("run", {}).get("questions_subdir", "questions/group_a"))
    raw_docs_subdir = Path(data.get("run", {}).get("raw_docs_subdir", "raw"))
    split = str(data.get("run", {}).get("split", "A"))
    output_csv = Path(data.get("run", {}).get("output_csv", "outputs/answer.csv"))
    evidence_jsonl = data.get("run", {}).get("output_evidence_jsonl")
    output_evidence_jsonl = Path(evidence_jsonl) if evidence_jsonl else None

    base_url = str(data.get("llm", {}).get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    model = str(data.get("llm", {}).get("model", "qwen-plus"))
    api_key_env = str(data.get("llm", {}).get("api_key_env", "DASHSCOPE_API_KEY"))
    timeout_seconds = float(data.get("llm", {}).get("timeout_seconds", 120.0))
    max_retries = int(data.get("llm", {}).get("max_retries", 2))

    chunk_max_chars = int(data.get("retrieval", {}).get("chunk_max_chars", 1600))
    doc_top_k = int(data.get("retrieval", {}).get("doc_top_k", 3))
    per_doc_top_k = int(data.get("retrieval", {}).get("per_doc_top_k", 3))
    per_option_top_k = int(data.get("retrieval", {}).get("per_option_top_k", 5))
    top_k_chunks = int(data.get("retrieval", {}).get("top_k_chunks", 12))
    coarse_doc_window_chars = int(data.get("retrieval", {}).get("coarse_doc_window_chars", 2400))
    refine_context_chars = int(data.get("retrieval", {}).get("refine_context_chars", 12000))
    max_context_chars = int(data.get("retrieval", {}).get("max_context_chars", 9000))
    max_routing_rounds = int(data.get("retrieval", {}).get("max_routing_rounds", 2))

    log_level = str(data.get("log_level", "INFO"))

    run = RunConfig(
        dataset_root=dataset_root,
        questions_subdir=questions_subdir,
        raw_docs_subdir=raw_docs_subdir,
        split=split,
        output_csv=output_csv,
        output_evidence_jsonl=output_evidence_jsonl,
    )
    llm = LlmConfig(
        base_url=base_url,
        model=model,
        api_key_env=api_key_env,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    retrieval = RetrievalConfig(
        chunk_max_chars=chunk_max_chars,
        doc_top_k=doc_top_k,
        per_doc_top_k=per_doc_top_k,
        per_option_top_k=per_option_top_k,
        top_k_chunks=top_k_chunks,
        coarse_doc_window_chars=coarse_doc_window_chars,
        refine_context_chars=refine_context_chars,
        max_context_chars=max_context_chars,
        max_routing_rounds=max_routing_rounds,
    )
    return AppConfig(run=run, llm=llm, retrieval=retrieval, log_level=log_level)


def load_toml(path: Path) -> dict:
    """读取 TOML 文件并返回 dict。"""
    try:
        import tomllib
    except Exception:
        try:
            import tomli as tomllib
        except Exception:
            return parse_simple_toml(path.read_text(encoding="utf-8"))

    raw = path.read_bytes()
    return tomllib.loads(raw.decode("utf-8"))


def parse_simple_toml(text: str) -> dict:
    """Parse the limited TOML subset used by this project."""
    result: dict[str, object] = {}
    current = result

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("[") and line.endswith("]"):
            section_name = line[1:-1].strip()
            if not section_name:
                continue
            current = result.setdefault(section_name, {})
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        current[key] = parse_simple_toml_value(value)

    return result


def parse_simple_toml_value(value: str) -> object:
    """Parse strings, ints, floats and booleans from simple TOML values."""
    if not value:
        return ""

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if value[0] in {'"', "'"}:
        try:
            return ast.literal_eval(value)
        except Exception:
            return value.strip('"').strip("'")

    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_dotenv(path: Path) -> None:
    """加载 .env 文件到环境变量（不覆盖已存在的变量）。"""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        os.environ.setdefault(key, value)


if __name__ == "__main__":
    main()
