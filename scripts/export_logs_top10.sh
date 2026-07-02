#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_PATH="${EXPORT_LOGS_INPUT:-}"
OUTPUT_PATH="${EXPORT_LOGS_OUTPUT:-}"
LIMIT="${EXPORT_LOGS_LIMIT:-}"

ARGS=()
if [ -n "$INPUT_PATH" ]; then
  ARGS+=(--input "$INPUT_PATH")
fi
if [ -n "$OUTPUT_PATH" ]; then
  ARGS+=(--output "$OUTPUT_PATH")
fi
if [ -n "$LIMIT" ]; then
  ARGS+=(--limit "$LIMIT")
fi

python -m scripts.export_logs_top10 "${ARGS[@]}" "$@"
