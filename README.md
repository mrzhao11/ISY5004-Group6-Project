# Pedestrian Behavior and Crossing Risk Prediction

This project implements a two-stage intelligent sensing demo for pedestrian behavior understanding and crossing-risk prediction from driving videos. The system accepts a raw video, detects and tracks pedestrians, extracts pedestrian crop sequences, predicts short-term behavior labels, and estimates whether the selected pedestrian is likely to cross.

The web demo is designed for direct evaluation: a teacher can open the dashboard, choose one of the provided demo videos or upload a local video, and view behavior labels, crop evidence, crossing probability, and risk level.

## System Workflow

```text
Raw driving video
  -> YOLO pedestrian detection
  -> lightweight IoU tracking
  -> pedestrian crop sequence extraction
  -> Stage 1 behavior recognition
  -> Stage 1 CSV with behavior probabilities and serialized track windows
  -> Stage 2 crossing-risk inference
  -> behavior labels, crossing probability, and risk level
```

## Main Features

- Raw video input through the frontend dashboard.
- Built-in demo video selector for quick testing.
- YOLO-based pedestrian detection and lightweight tracking.
- Pedestrian crop previews for visual verification.
- Stage 1 behavior recognition for action and looking behavior.
- Stage 2 crossing-risk prediction using trajectory, motion, and behavior features.
- Docker-based startup with automatic model-weight checking and download.
- Prediction CSV files saved under `outputs/predictions` for later analysis.

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
    nginx.conf
  stage1_inference/
    stage1.py
    infer_best.py
    models.py
  stage2_inference/
    stage2.py
    infer_best.py
    models.py
  data/
    demo/
    raw/
    processed/
  models/
    stage1/
    stage2/
  outputs/
    predictions/
  scripts/
    download_model_weights.py
    up.sh
    down.sh
  Dockerfile
  docker-compose.yml
  requirements.txt
```

## Quick Start

### Prerequisites

- Docker Desktop, or Docker Engine with the Compose plugin.
- Internet access on first startup so the model weights can be downloaded from the team Google Drive folders.

### Run The System

```bash
cd pedestrian_sensing
docker compose up --build
```

The backend entrypoint checks the required model weights in `models/stage1` and `models/stage2`. Existing files are reused. Missing files are downloaded automatically.

### Open The Dashboard

- Frontend: [http://localhost:8080](http://localhost:8080)
- Backend health check: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
- Backend API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Stop The System

```bash
docker compose down
```

## Web Demo Usage

1. Open [http://localhost:8080](http://localhost:8080).
2. Select a demo video from the dropdown, or upload a local video file.
3. Preview the raw video in the browser.
4. Click `Run Analysis`.
5. Review the detected pedestrian crops, behavior labels, crossing probability, and risk level.

The backend saves one Stage 1 CSV and one Stage 2 CSV for each request under `outputs/predictions`.

## Model Weights

Model weights are intentionally excluded from git. They are prepared by:

```bash
python scripts/download_model_weights.py
```

Useful options:

- `--stage stage1`: download only Stage 1 detector/action/look weights.
- `--stage stage2`: download only Stage 2 crossing-risk weights.
- `--force`: redownload files even if local copies already exist.

Expected files:

```text
models/stage1/
  action_sequence_swin3d_t.pt
  look_frame_swin_t.pt
  person_detector.pt
models/stage2/
  trajectory_motion_tcl.pt
  trajectory_motion_stage1_raw_logit_seed7_tcl.pt
```

## Stage 1

Stage 1 converts a raw video into behavior predictions and trajectory-window features.

Input:

- Raw video file, or a detection/tracking CSV.
- YOLO person detector weight if detections are not provided.

Output:

- One row per sliding pedestrian window.
- Action probability: standing / walking.
- Looking probability: looking / not-looking.
- Behavior embedding.
- `track_window_json`, which stores the bounding-box sequence used by Stage 2.

Run Stage 1 directly:

```bash
python -m stage1_inference.stage1 \
  --video-path data/raw/sample_clip.mp4 \
  --yolo-model-path models/stage1/person_detector.pt \
  --clip-length 16 \
  --window-stride 8 \
  --device cuda \
  --output-path outputs/predictions/stage1_video_inference.csv
```

Use `--device cpu` if CUDA is not available.

## Stage 2

Stage 2 reads the Stage 1 CSV and estimates crossing risk. It does not rerun detection or behavior recognition.

Input:

- Stage 1 CSV containing behavior probabilities and `track_window_json`.
- Stage 2 crossing-risk model weights.

Output:

- `base_prob_crossing`
- `stage1_aux_prob_crossing`
- `prob_crossing`
- `pred_crossing`

Run Stage 2 directly:

```bash
python -m stage2_inference.stage2 \
  --stage1-path outputs/predictions/stage1_video_inference.csv \
  --observation-length 8 \
  --device cuda \
  --output-path outputs/predictions/stage2_video_inference.csv
```

The final crossing probability is a deployment-oriented T42 blend:

```text
prob_crossing = 0.96 * base_prob_crossing + 0.04 * stage1_aux_prob_crossing
```

## API Endpoints

- `GET /api/v1/health`: backend health check.
- `GET /api/v1/demo-videos`: list available demo videos.
- `GET /api/v1/demo-videos/{filename}`: stream a demo video for browser preview.
- `POST /api/v1/analyze-demo`: run the full pipeline on a selected demo video.
- `POST /api/v1/analyze-video`: upload a video and run the full pipeline.
- `POST /api/v1/analyze`: run the full pipeline from a server-side video path.

Example request for `POST /api/v1/analyze`:

```json
{
  "video_path": "data/raw/sample_clip.mp4",
  "pedestrian_id": null,
  "yolo_model_path": "models/stage1/person_detector.pt",
  "clip_length": 16,
  "window_stride": 8,
  "look_max_frames": 15,
  "device": "cpu",
  "include_context": true
}
```

## Outputs

Runtime outputs are written to:

```text
outputs/predictions/
```

For each web request, the backend creates:

- `stage1_video_inference_<request_id>.csv`
- `stage2_video_inference_<request_id>.csv`
