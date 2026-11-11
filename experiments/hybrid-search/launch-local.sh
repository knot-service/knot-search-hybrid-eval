#!/bin/zsh
set -eu
cd -- "${0:A:h}"
if [[ -n "${KNOT_EVAL_ENV_FILE:-}" ]]; then
  while IFS='=' read -r name value || [[ -n "$name" ]]; do
    if [[ "$name" == LLM_* ]]; then
      export "$name=${(Q)value}"
    fi
  done < "$KNOT_EVAL_ENV_FILE"
fi
export HF_HUB_DISABLE_PROGRESS_BARS=1
exec "${KNOT_HYBRID_PYTHON:-.venv/bin/python}" -m uvicorn experiment:app --host 127.0.0.1 --port 18173
