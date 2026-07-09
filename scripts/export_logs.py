from __future__ import annotations

"""
运行指南（模块化）：

1) 默认导出（全量）
   - python -m scripts.export_logs
   - 输入：outputs/logs.csv
   - 输出：outputs/logs_export.json

2) 指定输入/输出
   - python -m scripts.export_logs --input outputs/logs.csv --output outputs/logs_all.json

说明：
- 该脚本会读取 logs.csv 并将所有记录写入 JSON；
- 每条记录仅保留 retrieval plan 与 evidence，便于全量导出；
- 以 *_json 结尾的列会尝试按 JSON 解析（解析失败则保留原字符串）。
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    root_dir = Path(__file__).resolve().parent.parent
    default_input = root_dir / "outputs" / "logs.csv"
    default_output = root_dir / "outputs" / "logs_export.json"

    parser = argparse.ArgumentParser(
        description="Export all records from outputs/logs.csv to a compact JSON file.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to logs.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Path to the output JSON file",
    )
    return parser.parse_args()


def maybe_parse_json(value: str) -> object:
    """尝试把字符串解析为 JSON 对象。"""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except Exception:
        return value


def _build_retrieval_plan(thought: object) -> dict[str, object]:
    """从 thought_trace 中提取/构造可导出的 retrieval plan。"""
    if not isinstance(thought, dict):
        return {"global_query": "", "option_queries": {}, "features": {}, "option_features": {}}
    plan = thought.get("retrieval_plan")
    if isinstance(plan, dict):
        return plan
    return {
        "global_query": thought.get("global_query", ""),
        "option_queries": thought.get("option_queries", {}),
        "features": thought.get("query_features", {}),
        "option_features": {},
    }


def _extract_evidence(search_trace: object) -> list[object]:
    """从 search_trace 中提取 evidence。"""
    if not isinstance(search_trace, dict):
        return []
    evidence = search_trace.get("evidence")
    return evidence if isinstance(evidence, list) else []


def normalize_export_row(row: dict[str, object]) -> dict[str, object]:
    """将单条日志记录压缩为只包含 retrieval plan 与 evidence。"""
    qid = str(row.get("qid") or "")
    domain = str(row.get("domain") or "")
    retrieval_plan = _build_retrieval_plan(row.get("thought_trace_json"))
    evidence = _extract_evidence(row.get("search_trace_json"))
    return {
        "qid": qid,
        "domain": domain,
        "retrieval_plan": retrieval_plan,
        "evidence": evidence,
    }


def iter_rows(path: Path):
    """逐行读取 CSV 并产出压缩后的导出结构。"""
    if not path.exists():
        raise FileNotFoundError(f"logs.csv not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row: dict[str, object] = {}
            for key, value in row.items():
                if key.endswith("_json"):
                    parsed_row[key] = maybe_parse_json(value or "")
                else:
                    parsed_row[key] = value or ""
            yield normalize_export_row(parsed_row)


def write_json(path: Path, rows) -> int:
    """把导出结果写入 JSON 文件（流式写入）。"""
    count_placeholder_width = 12
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write('{\n  "count": ')
        count_pos = f.tell()
        f.write("0" * count_placeholder_width)
        f.write(',\n  "records": [\n')
        first = True
        for item in rows:
            if not first:
                f.write(",\n")
            first = False
            f.write(json.dumps(item, ensure_ascii=False))
            count += 1
        f.write("\n  ]\n}\n")
        f.flush()
        if count < 10**count_placeholder_width:
            f.seek(count_pos)
            f.write(f"{count:0{count_placeholder_width}d}")
    return count


def main() -> None:
    """模块化入口。"""
    args = parse_args()
    count = write_json(path=args.output, rows=iter_rows(path=args.input))
    print(f"Exported {count} records to {args.output}")


if __name__ == "__main__":
    main()

