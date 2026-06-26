#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_PATH="${FIN_AGENT_CONFIG:-configs/agent.toml}"
DOTENV_PATH="${FIN_AGENT_DOTENV:-.env}"

FIN_AGENT_CONFIG="$CONFIG_PATH" FIN_AGENT_DOTENV="$DOTENV_PATH" python -m fin_agent.run
