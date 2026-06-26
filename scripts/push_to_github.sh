#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

REMOTE_URL="${GITHUB_REMOTE_URL:-${1:-}}"
COMMIT_MESSAGE="${GITHUB_COMMIT_MESSAGE:-${2:-chore: initial commit}}"
BRANCH="${GITHUB_BRANCH:-main}"

if ! command -v git >/dev/null 2>&1; then
  echo "git 未安装或不在 PATH 中" >&2
  exit 1
fi

if [ ! -d ".git" ]; then
  git init
fi

git branch -M "$BRANCH" >/dev/null 2>&1 || true

if git remote get-url origin >/dev/null 2>&1; then
  :
else
  if [ -z "$REMOTE_URL" ]; then
    echo "未设置远程地址。请提供参数或设置环境变量 GITHUB_REMOTE_URL" >&2
    echo "用法：" >&2
    echo "  GITHUB_REMOTE_URL=https://github.com/<user>/<repo>.git ./scripts/push_to_github.sh" >&2
    echo "  ./scripts/push_to_github.sh https://github.com/<user>/<repo>.git \"feat: init\"" >&2
    exit 2
  fi
  git remote add origin "$REMOTE_URL"
fi

git add -A

if git diff --cached --quiet; then
  echo "没有需要提交的变更"
  exit 0
fi

git commit -m "$COMMIT_MESSAGE" >/dev/null
git push -u origin "$BRANCH"
