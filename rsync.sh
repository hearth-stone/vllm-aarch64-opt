#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="Arm-codex"
REMOTE_DIR="/home/zhangxu/codex/vllm-aarch64-v0.22.0-dsv4"

RSYNC_DELETE=""
if [[ "${1:-}" == "--delete" ]]; then
  RSYNC_DELETE="--delete"
  shift
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_DIR}'"

rsync -azvh --progress \
  ${RSYNC_DELETE:+${RSYNC_DELETE}} \
  --exclude ".codegraph/" \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude ".cache/" \
  --exclude ".mypy_cache/" \
  --exclude ".pytest_cache/" \
  --exclude ".ruff_cache/" \
  --exclude "__pycache__/" \
  --exclude "build/" \
  --exclude "dist/" \
  --exclude "*.egg-info/" \
  --exclude "*.pyc" \
  --exclude "*.so" \
  "$@" \
  "${SCRIPT_DIR}/" \
  "${REMOTE_HOST}:${REMOTE_DIR}/"
