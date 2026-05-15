from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import DataConfig, ModelConfig
from .dataset import Stage2Dataset
from .models import Stage2CrossingModel


DEFAULT_BASE_CHECKPOINT = Path("models/stage2/trajectory_motion_tcl.pt")
DEFAULT_STAGE1_CHECKPOINT = Path("models/stage2/trajectory_motion_stage1_raw_logit_seed7_tcl.pt")

DEFAULT_BASE_MANIFEST = Path("data/processed/stage2/base_manifest.csv")
DEFAULT_STAGE1_MANIFEST = Path("data/processed/stage2/stage1_manifest_seed7.csv")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deployment-oriented Stage 2 T42 inference")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1_CHECKPOINT)
    parser.add_argument("--base-manifest-path", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--stage1-manifest-path", type=Path, default=DEFAULT_STAGE1_MANIFEST)
    parser.add_argument("--base-weight", type=float, default=0.96)
    parser.add_argument("--stage1-weight", type=float, default=0.04)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/predictions/stage2_best_inference_test.csv"))
    return parser


def _load_model(checkpoint_path: Path, manifest_path: Path, device: torch.device) -> tuple[Stage2CrossingModel, DataConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    data_cfg = DataConfig(**checkpoint["data_config"])
    data_cfg.manifest_path = manifest_path
    allowed = {field.name for field in fields(ModelConfig)}
    model_cfg = ModelConfig(**{key: value for key, value in checkpoint["model_config"].items() if key in allowed})
    probe_dataset = Stage2Dataset(manifest_path, split="train", config=data_cfg)
    model = Stage2CrossingModel(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        trajectory_static_dim=probe_dataset.metadata.trajectory_static_dim,
        behavior_feature_dim=probe_dataset.metadata.behavior_feature_dim,
        context_feature_dim=probe_dataset.metadata.context_feature_dim,
        vehicle_feature_dim=probe_dataset.metadata.vehicle_feature_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, data_cfg


def _predict_manifest(
    model: Stage2CrossingModel,
    manifest_path: Path,
    data_cfg: DataConfig,
    split: str,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    parts = ("train", "val", "test") if split == "all" else (split,)
    outputs: list[pd.DataFrame] = []
    for part in parts:
        dataset = Stage2Dataset(manifest_path, split=part, config=data_cfg)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        logits: list[float] = []
        with torch.no_grad():
            for batch in loader:
                labels = batch["label"]
                batch = {key: value.to(device) for key, value in batch.items()}
                batch["label"] = labels.to(device)
                logits.extend(float(value) for value in model(batch).cpu())
        meta = dataset.manifest[["sequence_id", "video_id", "pedestrian_id", "split", "y_intent"]].copy()
        meta["logit"] = logits
        outputs.append(meta)
    return pd.concat(outputs, axis=0, ignore_index=True)


def main() -> None:
    args = build_argparser().parse_args()
    total_weight = args.base_weight + args.stage1_weight
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Blend weights must sum to 1.0, got {total_weight:.6f}")

    device = torch.device(args.device)
    base_model, base_cfg = _load_model(args.base_checkpoint, args.base_manifest_path, device)
    stage1_model, stage1_cfg = _load_model(args.stage1_checkpoint, args.stage1_manifest_path, device)

    base = _predict_manifest(base_model, args.base_manifest_path, base_cfg, args.split, args.batch_size, device)
    stage1 = _predict_manifest(stage1_model, args.stage1_manifest_path, stage1_cfg, args.split, args.batch_size, device)

    merged = base.rename(columns={"logit": "base_logit"}).merge(
        stage1[["sequence_id", "logit"]].rename(columns={"logit": "stage1_logit"}),
        on="sequence_id",
        how="inner",
    )
    if len(merged) != len(base):
        raise RuntimeError("T42 manifests are not aligned on sequence_id")

    merged["base_prob_crossing"] = torch.sigmoid(torch.tensor(merged["base_logit"].to_numpy())).numpy()
    merged["stage1_prob_crossing"] = torch.sigmoid(torch.tensor(merged["stage1_logit"].to_numpy())).numpy()
    merged["prob_crossing"] = (
        args.base_weight * merged["base_prob_crossing"]
        + args.stage1_weight * merged["stage1_prob_crossing"]
    )
    merged["pred_crossing"] = (merged["prob_crossing"] >= 0.5).astype(int)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_id",
        "video_id",
        "pedestrian_id",
        "split",
        "y_intent",
        "base_logit",
        "stage1_logit",
        "base_prob_crossing",
        "stage1_prob_crossing",
        "prob_crossing",
        "pred_crossing",
    ]
    with args.output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged[fieldnames].to_dict(orient="records"))

    print(f"rows={len(merged)}")
    print(f"weights=base:{args.base_weight:.2f},stage1:{args.stage1_weight:.2f}")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main()
