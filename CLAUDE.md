# CLAUDE.md — ISY5004 Group 6 Project

This file lets Claude Code reconstruct the full project context on a new server.
Read it before doing anything in this repo.

---

## Project Overview

Two-stage intelligent sensing system based on the JAAD dataset.
Predicts whether a pedestrian will cross the road in the next 1–2 seconds.

**Pipeline:**
```
Video frames → Detection → Tracking → Stage 1 (Behavior) → Stage 2 (Crossing Prediction)
```

**Team split:**
| Member | Responsibility |
|--------|---------------|
| A (you) | Data preprocessing — JAAD parsing, pedestrian sequences, trajectory features |
| B | Stage 1 — CNN+LSTM / Video Swin Transformer behavior recognition |
| C | Stage 2 — TCL / XGBoost multimodal crossing intention prediction |
| D | System integration — pipeline, report |

---

## Repository Setup

```bash
git clone -b data-preprocessing https://github.com/mrzhao11/ISY5004-Group6-Project.git
cd ISY5004-Group6-Project
```

---

## Directory Structure

```
ISY5004-Group6-Project/
├── backend/
│   ├── pipeline.py          # TwoStagePedestrianPipeline (FastAPI app logic)
│   ├── stage1_behavior.py   # Stage 1 stub (MD5-based mock, ready for real model)
│   ├── stage2_risk.py       # Stage 2 stub (lookup-table mock, ready for real model)
│   ├── schemas.py           # Pydantic models: AnalyzeRequest/Response, BehaviorPrediction, RiskPrediction
│   └── api.py               # FastAPI routes
├── triton/
│   ├── model_repository/
│   │   ├── stage1_behavior/         # B's model slot
│   │   │   ├── config.pbtxt
│   │   │   └── 1/model.py           # Python backend (replace with real model)
│   │   ├── stage2_crossing/         # C's model slot
│   │   │   ├── config.pbtxt
│   │   │   └── 1/model.py           # Python backend (replace with real model)
│   │   └── pedestrian_pipeline/     # Ensemble: auto-chains Stage1 → Stage2
│   │       ├── config.pbtxt
│   │       └── 1/                   # Empty dir required by Triton for ensemble
│   └── client/
│       ├── test_inference.py        # Full inference test + latency benchmark
│       └── requirements.txt
├── docker-compose.yml       # Services: triton (8001), backend (8000), frontend (8080)
├── requirements.txt         # fastapi, uvicorn, pydantic, Pillow
└── scripts/
    ├── prepare_jaad.py
    ├── check_stage1_inputs.py
    └── check_stage2_features.py
```

---

## Triton Deployment

### What was built

Three Triton models using the **Python backend** (CPU-compatible, no GPU required for simulation):

#### stage1_behavior
- **Input:** `ped_clip` `[B, 16, 224, 224, 3]` float32
- **Outputs:** `behavior_probs` `[B, 4]` float32, `behavior_embedding` `[B, 256]` float32
- **Labels (index 0–3):** walking, standing, looking, waiting
- **Current logic:** deterministic simulated softmax
- **B's replacement point:** `triton/model_repository/stage1_behavior/1/model.py` — replace the body of `execute()` where comments say `# Replace with:`

#### stage2_crossing
- **Inputs:**
  - `traj_features` `[B, 16, 6]` float32 — `[cx, cy, w, h, Δcx, Δcy]` per frame from tracker
  - `behavior_probs` `[B, 4]` float32 — from Stage 1
  - `behavior_embedding` `[B, 256]` float32 — from Stage 1
  - `context_features` `[B, 6]` float32 — `[crosswalk, traffic_light, stop_sign, weather, time_of_day, crossing_loc]`
  - `vehicle_features` `[B, 4]` float32 — `[moving_slow, moving_fast, slowing_down, speeding_up]`
- **Outputs:** `crossing_prob` `[B, 1]` float32, `risk_level` `[B, 1]` int32 (0=Low, 1=Medium, 2=High)
- **C's replacement point:** `triton/model_repository/stage2_crossing/1/model.py` — replace the fusion block

#### pedestrian_pipeline (Ensemble)
- **Inputs:** `ped_clip`, `traj_features`, `context_features`, `vehicle_features`
- **Outputs:** `behavior_probs`, `crossing_prob`, `risk_level`
- **No code to change** — config-only ensemble that chains the two models above

### Start Triton

```bash
# First run pulls ~10 GB image (nvcr.io/nvidia/tritonserver:24.10-py3)
docker compose up triton

# Check server is live
curl http://localhost:8001/v2/health/live
```

### Run inference tests

```bash
pip install -r triton/client/requirements.txt

# Default: batch=1, 20 latency runs
python triton/client/test_inference.py --url localhost:8001

# Larger batch
python triton/client/test_inference.py --url localhost:8001 --batch 4 --runs 50
```

Test script runs four checks in order:
1. `stage1_behavior` alone
2. `stage2_crossing` alone (with manually constructed Stage 1 inputs)
3. `pedestrian_pipeline` full ensemble
4. Latency benchmark (p50 / p95)

### Port map

| Host port | Service |
|-----------|---------|
| 8000 | FastAPI backend |
| 8001 | Triton HTTP inference |
| 8002 | Triton gRPC inference |
| 8003 | Triton metrics |
| 8080 | Frontend (nginx) |

---

## How B and C Plug In Their Real Models

Both members only need to edit one file each.
Tensor shapes and the ensemble config stay the same.

**B replaces Stage 1:**
```python
# triton/model_repository/stage1_behavior/1/model.py
# In execute(), replace the simulation block with:
logits = your_video_swin_model(clip)          # or CNN+LSTM
behavior_probs = softmax(logits)
behavior_embedding = your_backbone.features   # 256-dim
```

**C replaces Stage 2:**
```python
# triton/model_repository/stage2_crossing/1/model.py
# In execute(), replace the fusion block with:
features = np.concatenate([traj.reshape(B,-1), b_prob, b_emb, ctx, veh], axis=1)
crossing_prob = your_xgboost_or_tcl_model.predict(features)
```

If C uses a saved XGBoost model file, load it in `initialize()`:
```python
def initialize(self, args):
    import xgboost as xgb, json
    model_dir = args["model_repository"] + "/" + args["model_version"]
    self.model = xgb.Booster()
    self.model.load_model(model_dir + "/model.ubj")
```

---

## Rebuild Checklist (new server)

```bash
# 1. Clone
git clone -b data-preprocessing https://github.com/mrzhao11/ISY5004-Group6-Project.git
cd ISY5004-Group6-Project

# 2. Start Triton
docker compose up triton -d

# 3. Wait ~30s for models to load, then test
pip install tritonclient[http] numpy
python triton/client/test_inference.py --url localhost:8001

# 4. (Optional) start full stack
docker compose up -d
```

Expected test output (simulation mode):
```
stage1_behavior              [READY]
stage2_crossing              [READY]
pedestrian_pipeline          [READY]

Test 1 — stage1_behavior
  sample [0]  behavior=walking (0.xxx)  ...

Test 3 — pedestrian_pipeline (full ensemble)
  sample [0]  Stage1=walking(0.xxx)  Stage2=crossing_prob=0.xxx  risk=Medium
```

---

## Key Design Decisions

- Stage 1 does **not** output crossing probability — avoids label leakage into Stage 2
- Stage 1 learns "current state"; Stage 2 learns "future intention"
- Ensemble wires `behavior_embedding` through without client needing to manage intermediate tensors
- Python backend chosen over ONNX/TensorRT so B and C can iterate in pure Python before optimising
