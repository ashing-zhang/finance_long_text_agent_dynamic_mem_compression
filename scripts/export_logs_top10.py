from __future__ import annotations

"""
运行指南（模块化）：

1) 默认导出
   - python -m scripts.export_logs_top10
   - 输入：outputs/logs.csv
   - 输出：outputs/logs_top10.json
   - 行数：10

2) 指定输入/输出与导出行数
   - python -m scripts.export_logs_top10 --input outputs/logs.csv --output outputs/logs_top50.json --limit 50

说明：
- 该脚本会读取 logs.csv 的前 N 行并写入 JSON；
- 以 *_json 结尾的列会尝试按 JSON 解析（解析失败则保留原字符串）。
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent.parent
    default_input = root_dir / "outputs" / "logs.csv"
    default_output = root_dir / "outputs" / "logs_top10.json"

    parser = argparse.ArgumentParser(
        description="Export the first N records from outputs/logs.csv to a JSON file."
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
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of rows to export",
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


def prune_search_trace(trace: object) -> object:
    """移除导出 JSON 中体积较大的检索明细字段。"""
    if not isinstance(trace, dict):
        return trace
    retrieval_rounds = trace.get("retrieval_rounds")
    if not isinstance(retrieval_rounds, list):
        return trace

    compact_rounds: list[object] = []
    for item in retrieval_rounds:
        if not isinstance(item, dict):
            compact_rounds.append(item)
            continue
        compact_round = dict(item)
        compact_round.pop("top_hits", None)
        compact_round.pop("option_doc_hits", None)
        compact_rounds.append(compact_round)

    compact_trace = dict(trace)
    compact_trace["retrieval_rounds"] = compact_rounds
    return compact_trace


def normalize_export_row(row: dict[str, object]) -> dict[str, object]:
    """按导出需求清理单条日志记录。"""
    normalized = dict(row)
    normalized["search_trace_json"] = prune_search_trace(normalized.get("search_trace_json"))
    thought = normalized.get("thought_trace_json")
    if isinstance(thought, dict):
        plan = thought.get("retrieval_plan")
        if not isinstance(plan, dict):
            plan = {
                "global_query": thought.get("global_query", ""),
                "option_queries": thought.get("option_queries", {}),
                "features": thought.get("query_features", {}),
                "option_features": {},
            }
        normalized["retrieval_plan_json"] = plan
    return normalized


def load_rows(path: Path, limit: int) -> list[dict[str, object]]:
    """读取 CSV 并转换为适合导出的结构。"""
    if limit < 0:
        raise ValueError("--limit must be >= 0")
    if not path.exists():
        raise FileNotFoundError(f"logs.csv not found: {path}")

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            if index >= limit:
                break
            parsed_row: dict[str, object] = {}
            for key, value in row.items():
                if key.endswith("_json"):
                    parsed_row[key] = maybe_parse_json(value or "")
                else:
                    parsed_row[key] = value or ""
            rows.append(normalize_export_row(parsed_row))
    return rows


def write_json(path: Path, rows: list[dict[str, object]]) -> None:
    """把导出结果写入 JSON 文件。"""
    payload = {
        "count": len(rows),
        "records": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(path=args.input, limit=args.limit)
    write_json(path=args.output, rows=rows)
    print(f"Exported {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()
