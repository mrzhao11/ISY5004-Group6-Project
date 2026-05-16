"""
Triton Inference Client — JAAD Two-Stage Pipeline Test

Tests three models:
  1. stage1_behavior      — behavior recognition alone
  2. stage2_crossing      — crossing prediction alone (manual Stage 1 inputs)
  3. pedestrian_pipeline  — full ensemble (Stage 1 → Stage 2)

Usage:
  pip install tritonclient[grpc] numpy
  python test_inference.py [--url localhost:8002] [--batch 1]
"""

import argparse
import time

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

BEHAVIOR_LABELS = ["walking", "standing", "looking", "waiting"]
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def make_ped_clip(batch: int, T: int = 16, H: int = 224, W: int = 224) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((batch, T, H, W, 3)).astype(np.float32)


def make_traj_features(batch: int, T: int = 16) -> np.ndarray:
    rng = np.random.default_rng(7)
    cx = np.linspace(0.3, 0.5, T)
    cy = np.linspace(0.4, 0.6, T)
    traj = np.stack([
        cx, cy,
        np.full(T, 0.08),
        np.full(T, 0.20),
        np.gradient(cx),
        np.gradient(cy),
    ], axis=-1).astype(np.float32)
    return np.tile(traj[np.newaxis], (batch, 1, 1))


def make_context_features(batch: int) -> np.ndarray:
    return np.array([[1.0, 1.0, 0.0, 0.8, 0.5, 1.0]] * batch, dtype=np.float32)


def make_vehicle_features(batch: int) -> np.ndarray:
    return np.array([[0.0, 0.0, 1.0, 0.0]] * batch, dtype=np.float32)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fp32(name: str, data: np.ndarray) -> grpcclient.InferInput:
    inp = grpcclient.InferInput(name, list(data.shape), "FP32")
    inp.set_data_from_numpy(data)
    return inp


def _out(name: str) -> grpcclient.InferRequestedOutput:
    return grpcclient.InferRequestedOutput(name)


def infer(client, model_name: str, inputs, output_names):
    outputs = [_out(n) for n in output_names]
    t0 = time.perf_counter()
    result = client.infer(model_name, inputs, outputs=outputs)
    latency_ms = (time.perf_counter() - t0) * 1000
    return result, latency_ms


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# ---------------------------------------------------------------------------
# Individual model tests
# ---------------------------------------------------------------------------

def test_stage1(client, batch: int):
    print_header("Test 1 — stage1_behavior")

    clip = make_ped_clip(batch)
    result, ms = infer(
        client, "stage1_behavior",
        [_fp32("ped_clip", clip)],
        ["behavior_probs", "behavior_embedding"],
    )

    b_probs = result.as_numpy("behavior_probs")      # [B, 4]
    emb     = result.as_numpy("behavior_embedding")  # [B, 256]

    for b in range(batch):
        top = int(b_probs[b].argmax())
        print(f"  sample [{b}]  behavior={BEHAVIOR_LABELS[top]}({b_probs[b][top]:.3f})  "
              f"probs={b_probs[b].round(3).tolist()}  "
              f"emb_norm={float(np.linalg.norm(emb[b])):.2f}")

    print(f"  latency: {ms:.1f} ms  (batch={batch})")
    return b_probs, emb


def test_stage2(client, batch: int):
    print_header("Test 2 — stage2_crossing  (manual Stage 1 inputs)")

    rng   = np.random.default_rng(0)
    b_prob = rng.dirichlet(alpha=[2, 1, 1, 0.5], size=batch).astype(np.float32)  # [B, 4]
    b_emb  = rng.standard_normal((batch, 256)).astype(np.float32) * 0.1           # [B, 256]
    traj   = make_traj_features(batch)
    ctx    = make_context_features(batch)
    veh    = make_vehicle_features(batch)

    inputs = [
        _fp32("traj_features",      traj),
        _fp32("behavior_probs",     b_prob),
        _fp32("behavior_embedding", b_emb),
        _fp32("context_features",   ctx),
        _fp32("vehicle_features",   veh),
    ]

    result, ms = infer(client, "stage2_crossing", inputs, ["crossing_prob", "risk_level"])

    probs = result.as_numpy("crossing_prob")  # [B, 1]
    risks = result.as_numpy("risk_level")     # [B, 1]

    for b in range(batch):
        top = int(b_prob[b].argmax())
        print(f"  sample [{b}]  behavior={BEHAVIOR_LABELS[top]}  "
              f"crossing_prob={probs[b, 0]:.3f}  "
              f"risk={RISK_LABELS[int(risks[b, 0])]}")

    print(f"  latency: {ms:.1f} ms  (batch={batch})")


def test_ensemble(client, batch: int):
    print_header("Test 3 — pedestrian_pipeline  (full ensemble)")

    clip = make_ped_clip(batch)
    traj = make_traj_features(batch)
    ctx  = make_context_features(batch)
    veh  = make_vehicle_features(batch)

    inputs = [
        _fp32("ped_clip",          clip),
        _fp32("traj_features",     traj),
        _fp32("context_features",  ctx),
        _fp32("vehicle_features",  veh),
    ]

    result, ms = infer(
        client, "pedestrian_pipeline", inputs,
        ["behavior_probs", "crossing_prob", "risk_level"],
    )

    b_probs = result.as_numpy("behavior_probs")  # [B, 4]
    c_probs = result.as_numpy("crossing_prob")   # [B, 1]
    risks   = result.as_numpy("risk_level")      # [B, 1]

    for b in range(batch):
        top = int(b_probs[b].argmax())
        print(f"  sample [{b}]  "
              f"Stage1={BEHAVIOR_LABELS[top]}({b_probs[b][top]:.3f})  "
              f"Stage2=crossing_prob={c_probs[b, 0]:.3f}  "
              f"risk={RISK_LABELS[int(risks[b, 0])]}")

    print(f"  latency: {ms:.1f} ms  (batch={batch})")


def test_latency(client, batch: int, n_runs: int = 20):
    print_header(f"Latency benchmark — {n_runs} runs (batch={batch})")

    clip = make_ped_clip(batch)
    traj = make_traj_features(batch)
    ctx  = make_context_features(batch)
    veh  = make_vehicle_features(batch)

    inputs = [
        _fp32("ped_clip",         clip),
        _fp32("traj_features",    traj),
        _fp32("context_features", ctx),
        _fp32("vehicle_features", veh),
    ]
    out_names = ["behavior_probs", "crossing_prob", "risk_level"]

    times = []
    for _ in range(n_runs):
        _, ms = infer(client, "pedestrian_pipeline", inputs, out_names)
        times.append(ms)

    times = np.array(times)
    print(f"  mean={times.mean():.1f}ms  "
          f"p50={np.percentile(times, 50):.1f}ms  "
          f"p95={np.percentile(times, 95):.1f}ms  "
          f"min={times.min():.1f}ms  max={times.max():.1f}ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",   default="localhost:8002", help="Triton gRPC endpoint")
    parser.add_argument("--batch", default=1,  type=int,    help="Batch size")
    parser.add_argument("--runs",  default=20, type=int,    help="Runs for latency benchmark")
    args = parser.parse_args()

    print(f"Connecting to Triton at {args.url} ...")
    try:
        client = grpcclient.InferenceServerClient(url=args.url, verbose=False)
        if not client.is_server_live():
            print("ERROR: Server is not live.")
            return
        print("Server is live.\n")
    except Exception as e:
        print(f"Connection failed: {e}")
        return

    for model in ("stage1_behavior", "stage2_crossing", "pedestrian_pipeline"):
        ready = client.is_model_ready(model)
        print(f"  {model:30s} [{'READY' if ready else 'NOT READY'}]")

    try:
        test_stage1(client, args.batch)
        test_stage2(client, args.batch)
        test_ensemble(client, args.batch)
        test_latency(client, args.batch, args.runs)
    except InferenceServerException as e:
        print(f"\nInference error: {e}")


if __name__ == "__main__":
    main()
