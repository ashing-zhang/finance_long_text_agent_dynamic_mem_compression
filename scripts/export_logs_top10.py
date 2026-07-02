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
    if value is None:
        return None
    text = value.strip()
    if not text:
        return ""
    try:
        return json.loads(text)
    except Exception:
        return value


def load_rows(path: Path, limit: int) -> list[dict[str, object]]:
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
            rows.append(parsed_row)
    return rows


def write_json(path: Path, rows: list[dict[str, object]]) -> None:
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
