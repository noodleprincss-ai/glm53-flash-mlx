#!/bin/bash
set -euo pipefail

resolve_dir() { (cd "$1" && pwd -P); }

ROOT="$(resolve_dir "${GLM53_CAMPAIGN_ROOT:-$HOME/experiments/glm53-flash-mlx-opt-v1}")"
RUNTIME="$(resolve_dir "${GLM53_RUNTIME_WORKTREE:-$ROOT/worktrees/runtime/baseline}")"
VENV="$(resolve_dir "${GLM53_STOCK_VENV:-$ROOT/venvs/stock-py312}")"
MODEL_ROOT="$(resolve_dir "${GLM53_MODEL_ROOT:-$HOME/models/GLM-5.3-Flash-MLX-mixed-4_8-bit}")"
NEUTRAL="$ROOT/run-neutral"

[[ -x "$VENV/bin/python" ]] || { echo "missing campaign Python: $VENV/bin/python" >&2; exit 2; }
[[ ! -L "${GLM53_STOCK_VENV:-$ROOT/venvs/stock-py312}" ]] || { echo "campaign venv may not be a symlink" >&2; exit 2; }
[[ -d "$RUNTIME/.git" || -f "$RUNTIME/.git" ]] || { echo "missing runtime worktree: $RUNTIME" >&2; exit 2; }
[[ "$MODEL_ROOT" != "$RUNTIME"/* ]] || { echo "model root may not be inside a worktree" >&2; exit 2; }
mkdir -p "$NEUTRAL"

# Resolve an existing relative script against the runtime before leaving its CWD.
if [[ $# -gt 0 && "$1" != -* && -f "$RUNTIME/$1" ]]; then
  FIRST="$RUNTIME/$1"
  shift
  set -- "$FIRST" "$@"
fi
cd "$NEUTRAL"

exec env -i \
  HOME="$HOME" \
  LOGNAME="${LOGNAME:-$USER}" \
  USER="$USER" \
  TMPDIR="${TMPDIR:-/tmp}" \
  PATH="$VENV/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  VIRTUAL_ENV="$VENV" \
  PYTHONNOUSERSITE=1 \
  PYTHONSAFEPATH=1 \
  PYTHONPATH="$RUNTIME" \
  GLM53_CAMPAIGN_ROOT="$ROOT" \
  GLM53_MODEL_ROOT="$MODEL_ROOT" \
  "$VENV/bin/python" -s -P "$@"
