# Pedestrian Sensing Project

This project provides a demo-ready two-stage intelligent sensing system for pedestrian behavior understanding and crossing risk prediction in driving scenarios.

The current version is designed for direct team use:

- Stage 1 focuses on perception and temporal behavior understanding (YOLO + CNN/LSTM integration point).
- Stage 2 focuses on crossing-intention and risk estimation (XGBoost integration point).
- The frontend dashboard presents pipeline status, key outputs, and operational guidance in English.
- The backend exposes clear API contracts for integration with data preprocessing, model inference, and evaluation modules.
- Docker-based startup is included for consistent team deployment across machines.

## Project Overview

The system targets safety-critical pedestrian interactions in urban traffic. Instead of only detecting pedestrian location, it models behavior over time and estimates crossing probability to support proactive driving decisions.

Input:
- Video sequence path
- Pedestrian ID
- Optional scene-context flag

Output:
- Behavior label and confidence
- Crossing probability and risk level
- Feature-level summary for quick interpretation

## Project Structure

```text
pedestrian_sensing/
  backend/
    api.py
    pipeline.py
    schemas.py
    stage1_behavior.py
    stage2_risk.py
  frontend/
    index.html
    styles.css
    app.js
  data/
    raw/
    processed/
  models/
    stage1/
    stage2/
  outputs/
    logs/
    predictions/
  scripts/
    run_backend.sh
    up.sh
    down.sh
  Dockerfile
  docker-compose.yml
  requirements.txt
```

## One-Command Startup (Recommended)

### Prerequisites

- Docker Desktop (or Docker Engine + Compose plugin)

### Run

```bash
cd pedestrian_sensing
./scripts/up.sh
```

Equivalent command:

```bash
docker compose up --build
```

### Access

- Frontend dashboard: [http://localhost:8080](http://localhost:8080)
- Backend health check: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
- Backend API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Stop

```bash
cd pedestrian_sensing
./scripts/down.sh
```

Equivalent command:

```bash
docker compose down
```

## API Endpoints

- `GET /api/v1/health`
- `POST /api/v1/analyze`

Request example:

```json
{
  "video_path": "/data/raw/sample_clip.mp4",
  "pedestrian_id": "ped_001",
  "include_context": true
}
```

## Integration Checklist

1. Replace `backend/stage1_behavior.py` with real YOLO + CNN/LSTM inference.
2. Replace `backend/stage2_risk.py` with trained XGBoost inference.
3. Upload your team dataset under `data/raw` and connect preprocessing outputs into `data/processed`.
4. Add experiment scripts for Accuracy, F1, AUC, and mAP.
5. Save trained artifacts into `models/stage1` and `models/stage2`.
