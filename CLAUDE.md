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

### Triton image version

The `docker-compose.yml` specifies the image tag. Use whatever tag is already pulled on the server
to avoid re-downloading 9 GB+. Check with:

```bash
docker images | grep tritonserver
```

If the tag in `docker-compose.yml` doesn't match, update it:

```bash
# Example: locally available tag is 24.08-py3
sed -i 's|nvcr.io/nvidia/tritonserver:.*|nvcr.io/nvidia/tritonserver:24.08-py3|' docker-compose.yml
```

### Start Triton

```bash
docker compose up triton -d

# Triton takes ~10–15s to load all three models. Tail logs to confirm:
docker logs -f pedestrian_triton 2>&1 | grep -E "READY|ERROR|Started HTTP"
# Look for: "Started HTTPService at 0.0.0.0:8000"
```

### Smoke test (run after every new server setup)

```bash
# Install client deps — use --break-system-packages on Ubuntu if pip refuses
pip install tritonclient[http] numpy --break-system-packages

# Run with python3 (python may not be aliased)
python3 triton/client/test_inference.py --url localhost:8001 --batch 2
```

Test script runs four checks in order:
1. `stage1_behavior` alone
2. `stage2_crossing` alone (with manually constructed Stage 1 inputs)
3. `pedestrian_pipeline` full ensemble
4. Latency benchmark (p50 / p95)

Expected output on a CPU-only server (simulation mode, batch=2):

```
Server is live.

  stage1_behavior                [READY]
  stage2_crossing                [READY]
  pedestrian_pipeline            [READY]

Test 1 — stage1_behavior
  sample [0]  behavior=walking (0.328)  probs=[...]  emb_norm=1.60
  sample [1]  behavior=walking (0.408)  probs=[...]  emb_norm=1.52
  latency: ~160 ms  (batch=2)

Test 2 — stage2_crossing
  sample [0]  behavior=waiting  crossing_prob=0.364  risk=Low
  sample [1]  behavior=standing crossing_prob=0.317  risk=Low
  latency: ~27 ms  (batch=2)

Test 3 — pedestrian_pipeline (full ensemble)
  sample [0]  Stage1=walking(0.328)  Stage2=crossing_prob=0.348  risk=Low
  sample [1]  Stage1=walking(0.408)  Stage2=crossing_prob=0.363  risk=Low
  latency: ~85 ms  (batch=2)

Latency benchmark — 20 runs (batch=2)
  mean=57ms  p50=55ms  p95=76ms  min=53ms  max=76ms
```

If any model shows `[NOT READY]`, check logs:
```bash
docker logs pedestrian_triton 2>&1 | grep -i error
```

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

Both `model.py` files **auto-detect** which model file is present and load accordingly.
No code changes needed — just drop the weight file into the right directory.

### Stage 1 (B) — priority order

| File to place | Format | How loaded |
|---|---|---|
| `stage1_behavior/1/model.onnx` | ONNX | onnxruntime — fastest, use for benchmarking |
| `stage1_behavior/1/model.pt`   | PyTorch | `torch.load` + `model.eval()` |
| (neither) | — | numpy simulation, smoke-test mode |

**Export from PyTorch training:**
```python
# .pt
torch.save(model.state_dict(), "model.pt")

# .onnx
dummy = torch.zeros(1, 16, 3, 224, 224)   # NCTHW
torch.onnx.export(model, dummy, "model.onnx",
    input_names=["ped_clip"],
    output_names=["behavior_probs", "behavior_embedding"],
    dynamic_axes={"ped_clip": {0: "batch"}})
```

ONNX model must output tensors named exactly `behavior_probs` and `behavior_embedding`.

### Stage 2 (C) — priority order

| File to place | Format | How loaded |
|---|---|---|
| `stage2_crossing/1/model.onnx` | ONNX | onnxruntime — fastest |
| `stage2_crossing/1/model.ubj`  | XGBoost binary | `xgb.Booster().load_model()` |
| `stage2_crossing/1/model.pt`   | PyTorch (TCL) | `torch.load` + `model.eval()` |
| (none) | — | numpy simulation |

**Export from XGBoost:**
```python
model.save_model("model.ubj")
```

**Export XGBoost → ONNX (for benchmarking):**
```python
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
onnx_model = convert_xgboost(model, initial_types=[("features", FloatTensorType([None, 366]))])
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

ONNX model must accept input named `features` with shape `[B, 366]`
(= traj 96 + behavior_probs 4 + behavior_embedding 256 + context 6 + vehicle 4).

### Verifying after dropping in a model file

Restart Triton and re-run the smoke test — the log line will confirm which mode loaded:
```
stage1_behavior: loaded ONNX model       ← or "loaded PyTorch model" / "simulation mode"
stage2_crossing: loaded XGBoost model
```

---

## Latency Benchmarking (for report)

Run the same test script against three configurations to produce comparable numbers:

```bash
# Config 1: simulation (baseline, already done)
python3 triton/client/test_inference.py --url localhost:8001 --batch 2 --runs 50

# Config 2: real models, Python backend (.pt / .ubj)
# Drop model files → restart Triton → run same command

# Config 3: real models, ONNX backend (.onnx)
# Drop .onnx files → restart Triton → run same command
```

The `--runs` flag controls how many iterations the latency benchmark averages over.
Compare p50 and p95 across configs in the report.

---

## Rebuild Checklist (new server)

```bash
# 1. Clone
git clone -b data-preprocessing https://github.com/mrzhao11/ISY5004-Group6-Project.git
cd ISY5004-Group6-Project

# 2. Match docker-compose image tag to whatever is already pulled
docker images | grep tritonserver
# Then update docker-compose.yml tag if needed (see "Triton image version" section above)

# 3. Start Triton and wait for models to load
docker compose up triton -d
docker logs -f pedestrian_triton 2>&1 | grep -E "READY|ERROR|Started HTTP"

# 4. Run smoke test
pip install tritonclient[http] numpy --break-system-packages
python3 triton/client/test_inference.py --url localhost:8001 --batch 2

# 5. (Optional) start full stack
docker compose up -d
```

---

## Key Design Decisions

- Stage 1 does **not** output crossing probability — avoids label leakage into Stage 2
- Stage 1 learns "current state"; Stage 2 learns "future intention"
- Ensemble wires `behavior_embedding` through without client needing to manage intermediate tensors
- Python backend chosen over ONNX/TensorRT so B and C can iterate in pure Python before optimising
