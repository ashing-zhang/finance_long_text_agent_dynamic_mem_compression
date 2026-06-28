from __future__ import annotations

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
