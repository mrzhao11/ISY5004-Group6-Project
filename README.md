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

## Stage 1 Inference Artifacts

The current trained Stage 1 inference code has been migrated into:

```text
stage1_inference/
  config.py
  dataset.py
  frame_dataset.py
  infer_best.py
  models.py
models/stage1/
  action_sequence_swin3d_t.pt
  look_frame_swin_t.pt
  person_detector.pt
data/processed/stage1/
  action_manifest.csv
  frame_manifest.csv
  action_crops/
  frame_crops/
  action_tracks/
```

The migrated Stage 1 setup uses:

- Action classifier: 16-frame sequence input, Video Swin-T checkpoint.
- Look classifier: single-frame Swin-T checkpoint, frame probabilities averaged per pedestrian track.

Run inference from the project root:

```bash
python -m stage1_inference.infer_best \
  --split test \
  --device cuda \
  --output-path outputs/predictions/stage1_best_inference_test.csv
```

The command uses project-local default paths:

- `data/processed/stage1/action_manifest.csv`
- `data/processed/stage1/frame_manifest.csv`
- `models/stage1/action_sequence_swin3d_t.pt`
- `models/stage1/look_frame_swin_t.pt`

Use `--device cpu` if CUDA is not available. The output CSV contains action/look predictions, probabilities, and a 512-dimensional behavior embedding per action sequence window.

### Raw Video Stage 1 Inference

For real-scene inference from an original video, Stage 1 first needs pedestrian boxes. The project supports two modes:

1. Use an existing detection/tracking CSV.
2. Use a YOLO person detector if `ultralytics` and a YOLO weight file are installed locally.

Detection CSV format:

```csv
frame_id,pedestrian_id,x1,y1,x2,y2,score
0,ped_001,100,80,180,300,0.91
1,ped_001,102,82,182,302,0.93
```

Run with an existing detection CSV:

```bash
python -m stage1_inference.infer_video \
  --video-path data/raw/sample_clip.mp4 \
  --detections-csv data/processed/sample_detections.csv \
  --pedestrian-id ped_001 \
  --clip-length 16 \
  --window-stride 8 \
  --device cuda \
  --output-path outputs/predictions/stage1_video_inference.csv
```

Run with a local YOLO model:

```bash
python -m stage1_inference.infer_video \
  --video-path data/raw/sample_clip.mp4 \
  --yolo-model-path models/stage1/person_detector.pt \
  --clip-length 16 \
  --window-stride 8 \
  --device cuda \
  --output-path outputs/predictions/stage1_video_inference.csv
```

The default detector is `models/stage1/person_detector.pt`, which is the Ultralytics YOLOv8n COCO detector. It detects `person` boxes and the script applies a lightweight IoU tracker to form pedestrian tracks. The raw-video script creates temporary pedestrian crops internally, runs the action sequence model on sliding 16-frame crop windows, runs the look model on sampled single-frame crops from each window, and writes one prediction row per continuous window for the selected pedestrian.
