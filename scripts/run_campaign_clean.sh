#!/bin/bash
set -euo pipefail

ROOT="${GLM53_CAMPAIGN_ROOT:-$HOME/experiments/glm53-flash-mlx-opt-v1}"
RUNTIME="${GLM53_RUNTIME_WORKTREE:-$ROOT/worktrees/runtime/baseline}"
VENV="${GLM53_STOCK_VENV:-$ROOT/venvs/stock-py312}"
MODEL_ROOT="${GLM53_MODEL_ROOT:-$HOME/models/GLM-5.3-Flash-MLX-mixed-4_8-bit}"

[[ -x "$VENV/bin/python" ]] || { echo "missing campaign Python: $VENV/bin/python" >&2; exit 2; }
[[ -d "$RUNTIME/.git" || -f "$RUNTIME/.git" ]] || { echo "missing runtime worktree: $RUNTIME" >&2; exit 2; }
[[ "$MODEL_ROOT" != "$RUNTIME"/* ]] || { echo "model root may not be inside a worktree" >&2; exit 2; }

exec env -i \
  HOME="$HOME" \
  LOGNAME="${LOGNAME:-$USER}" \
  USER="$USER" \
  TMPDIR="${TMPDIR:-/tmp}" \
  PATH="$VENV/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  VIRTUAL_ENV="$VENV" \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$RUNTIME" \
  GLM53_CAMPAIGN_ROOT="$ROOT" \
  GLM53_MODEL_ROOT="$MODEL_ROOT" \
  "$VENV/bin/python" -s "$@"
