#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ "${AUTO_DOWNLOAD_STAGE1_WEIGHTS:-1}" == "1" ]]; then
  echo "[backend] checking Stage 1 weights..."
  python scripts/download_stage1_weights.py
fi

exec uvicorn backend.api:app --host 0.0.0.0 --port 8000
