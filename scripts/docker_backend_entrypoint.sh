#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ "${AUTO_DOWNLOAD_MODEL_WEIGHTS:-1}" == "1" ]]; then
  echo "[backend] checking model weights..."
  python scripts/download_model_weights.py
fi

exec uvicorn backend.api:app --host 0.0.0.0 --port 8000
