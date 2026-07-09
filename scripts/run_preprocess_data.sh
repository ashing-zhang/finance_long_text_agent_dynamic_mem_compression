#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${FIN_AGENT_CONFIG:-configs/agent.toml}"
DOTENV_PATH="${FIN_AGENT_DOTENV:-.env}"
PREPROCESS_LOG_LEVEL="${FIN_AGENT_PREPROCESS_LOG_LEVEL:-INFO}"

FIN_AGENT_CONFIG="$CONFIG_PATH" \
FIN_AGENT_DOTENV="$DOTENV_PATH" \
FIN_AGENT_PREPROCESS_LOG_LEVEL="$PREPROCESS_LOG_LEVEL" \
python -m fin_agent.preprocess_data "$@"
