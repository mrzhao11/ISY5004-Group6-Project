# Pedestrian Sensing Project

A demo-ready baseline for Group 6:

- Stage 1: Pedestrian detection + temporal behavior understanding (YOLO + CNN/LSTM integration point)
- Stage 2: Crossing intention and risk prediction (XGBoost integration point)
- English frontend dashboard with pipeline status and risk output
- Dockerized environment for one-command startup

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

## Local Startup (Without Docker)

```bash
cd pedestrian_sensing
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.api:app --reload --port 8000
```

Open frontend (new terminal):

```bash
cd pedestrian_sensing/frontend
python3 -m http.server 8080
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
