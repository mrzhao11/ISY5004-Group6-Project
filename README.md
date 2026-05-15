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

## Stage 2 Inference Artifacts

The current best Stage 2 inference code has been migrated into:

```text
stage2_inference/
  config.py
  dataset.py
  infer_best.py
  models.py
models/stage2/
  trajectory_motion_tcl.pt
  trajectory_motion_stage1_raw_logit_seed7_tcl.pt
data/processed/stage2/
  base_manifest.csv
  stage1_manifest_seed7.csv
  tracks/
```

The packaged Stage 2 predictor is the deployment-oriented `T42` no-context blend:

- `0.96`: trajectory sequence + track-derived motion model
- `0.04`: trajectory sequence + track-derived motion + Stage 1 `p_walking` / `p_looking`

Run Stage 2 inference from the project root:

```bash
python -m stage2_inference.infer_best \
  --split test \
  --device cuda \
  --output-path outputs/predictions/stage2_best_inference_test.csv
```

The output CSV contains each submodel logit/probability plus the fused crossing probability and binary prediction.

### What Stage 2 Needs

For each observed pedestrian window, Stage 2 needs:

- a pedestrian track sequence with bounding-box-derived fields such as `center_x`, `center_y`, `width`, and `height`
- track-derived motion statistics such as speed, displacement, bbox scale change, trajectory length, and jitter
- Stage 1 behavior probabilities: `p_walking` and `p_looking` for the T42 Stage 1 branch

The bundled offline manifests already contain these features for the migrated JAAD subset. The `tracks/` directory is required because the trajectory-sequence branch reconstructs the observed motion sequence from the per-window CSV files.

### From Raw Video To Stage 2

Starting from a raw video, the current code can already produce:

- person detections and lightweight tracks from YOLO
- pedestrian crops
- Stage 1 action/look probabilities
- trajectory sequence features and track-derived motion features from the tracked boxes

The full raw-video pipeline is technically feasible, but it is not yet fully wired into one command in this repository:

```text
raw video
  -> YOLO detection + tracking
  -> Stage 1 behavior inference
  -> sliding-window Stage 2 feature builder
  -> Stage 2 crossing inference
```

The packaged T42 predictor does not require scene-context annotation, so it is better aligned with real deployment from raw video than the higher-scoring offline T41 blend.

### Full Raw-Video Pipeline

The complete online path is now available:

```text
raw video
  -> YOLO person detection + lightweight tracking
  -> Stage 1 sliding-window action/look inference
  -> Stage 2 trajectory + motion feature construction
  -> T42 crossing inference
```

Run it from the project root:

```bash
python -m stage2_inference.infer_video_full \
  --video-path data/raw/sample_clip.mp4 \
  --yolo-model-path models/stage1/person_detector.pt \
  --clip-length 16 \
  --window-stride 8 \
  --device cuda \
  --output-path outputs/predictions/full_video_inference.csv
```

For each 16-frame sliding window, the script:

- runs Stage 1 over the cropped pedestrian sequence
- uses the first 8 tracked boxes as the Stage 2 observation sequence
- computes 16-frame track-derived motion statistics
- applies the deployment-oriented T42 blend

If `--pedestrian-id` is omitted, the script uses the longest detected track in the video.
