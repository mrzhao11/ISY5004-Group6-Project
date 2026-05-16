from __future__ import annotations

import argparse
import json
import os
import pathlib
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import DataConfig, ModelConfig
from .infer_best import DEFAULT_BASE_CHECKPOINT, DEFAULT_STAGE1_CHECKPOINT
from .models import Stage2CrossingModel


TRAJECTORY_STATIC_DIM = 6
STAGE1_BEHAVIOR_DIM = 2


@dataclass(slots=True)
class WindowDetection:
    frame_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 2 crossing inference from a Stage 1 raw-video prediction CSV."
    )
    parser.add_argument(
        "--stage1-path",
        type=Path,
        default=Path("outputs/predictions/stage1_video_inference.csv"),
        help="CSV produced by stage1_inference.stage1.",
    )
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1_CHECKPOINT)
    parser.add_argument("--observation-length", type=int, default=8)
    parser.add_argument("--base-weight", type=float, default=0.96)
    parser.add_argument("--stage1-weight", type=float, default=0.04)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/predictions/stage2_video_inference.csv"),
    )
    return parser


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    # Team checkpoints may have been saved on Windows and can contain pathlib.WindowsPath.
    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_online_model(checkpoint_path: Path, device: torch.device) -> tuple[Stage2CrossingModel, DataConfig]:
    checkpoint = _load_checkpoint(checkpoint_path, device)
    data_cfg = DataConfig(**checkpoint["data_config"])
    allowed = {field.name for field in fields(ModelConfig)}
    model_cfg = ModelConfig(**{key: value for key, value in checkpoint["model_config"].items() if key in allowed})

    trajectory_static_dim = TRAJECTORY_STATIC_DIM if data_cfg.use_trajectory_static else 0
    behavior_feature_dim = STAGE1_BEHAVIOR_DIM if data_cfg.behavior_mode == "stage1_features" else 0
    model = Stage2CrossingModel(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        trajectory_static_dim=trajectory_static_dim,
        behavior_feature_dim=behavior_feature_dim,
        context_feature_dim=0,
        vehicle_feature_dim=0,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, data_cfg


def _load_stage1_predictions(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {
        "pedestrian_id",
        "window_index",
        "start_frame",
        "end_frame",
        "prob_action_walking",
        "prob_look_looking",
        "track_window_json",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"{path} is missing Stage 2 input columns: {sorted(missing)}. "
            "Regenerate the file with stage1_inference.stage1 so trajectory windows are included."
        )
    if table.empty:
        raise RuntimeError(f"No Stage 1 prediction rows found in {path}")
    return table


def _parse_track_window(value: str) -> list[WindowDetection]:
    records = json.loads(value)
    if not isinstance(records, list) or not records:
        raise ValueError("track_window_json must contain a non-empty list of detection records")
    detections = []
    for record in records:
        detections.append(
            WindowDetection(
                frame_id=int(record["frame_id"]),
                x1=float(record["x1"]),
                y1=float(record["y1"]),
                x2=float(record["x2"]),
                y2=float(record["y2"]),
                score=float(record.get("score", 1.0)),
            )
        )
    return detections


def _pad_detections(detections: list[WindowDetection], target_length: int) -> list[WindowDetection]:
    if len(detections) >= target_length:
        return detections[:target_length]
    return detections + [detections[-1]] * (target_length - len(detections))


def _trajectory_sequence(window_detections: list[WindowDetection], observation_length: int) -> torch.Tensor:
    dets = _pad_detections(window_detections, observation_length)
    width = np.array([det.x2 - det.x1 for det in dets], dtype=np.float32)
    height = np.array([det.y2 - det.y1 for det in dets], dtype=np.float32)
    center_x = np.array([(det.x1 + det.x2) / 2.0 for det in dets], dtype=np.float32)
    center_y = np.array([(det.y1 + det.y2) / 2.0 for det in dets], dtype=np.float32)
    dx = np.diff(center_x, prepend=center_x[0])
    dy = np.diff(center_y, prepend=center_y[0])
    dw = np.diff(width, prepend=width[0])
    dh = np.diff(height, prepend=height[0])
    scale = max(float(width[0]), float(height[0]), 1.0)
    features = np.stack(
        [
            (center_x - center_x[0]) / scale,
            (center_y - center_y[0]) / scale,
            width / scale,
            height / scale,
            dx / scale,
            dy / scale,
            dw / scale,
            dh / scale,
        ],
        axis=1,
    )
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


def _trajectory_static(window_detections: list[WindowDetection]) -> torch.Tensor:
    width = np.array([det.x2 - det.x1 for det in window_detections], dtype=np.float32)
    height = np.array([det.y2 - det.y1 for det in window_detections], dtype=np.float32)
    center_x = np.array([(det.x1 + det.x2) / 2.0 for det in window_detections], dtype=np.float32)
    center_y = np.array([(det.y1 + det.y2) / 2.0 for det in window_detections], dtype=np.float32)
    dx = np.diff(center_x)
    dy = np.diff(center_y)
    speed = np.sqrt(dx**2 + dy**2)
    displacement = float(np.sqrt((center_x[-1] - center_x[0]) ** 2 + (center_y[-1] - center_y[0]) ** 2))
    area = width * height
    bbox_scale_change = float((area[-1] - area[0]) / max(float(area[0]), 1.0))
    trajectory_length = float(speed.sum())
    centroid = np.array([center_x.mean(), center_y.mean()])
    centers = np.stack([center_x, center_y], axis=1)
    center_jitter = float(np.linalg.norm(centers - centroid, axis=1).mean())
    raw = np.array(
        [
            float(speed.mean()) if len(speed) else 0.0,
            float(speed.var()) if len(speed) else 0.0,
            displacement,
            bbox_scale_change,
            trajectory_length,
            center_jitter,
        ],
        dtype=np.float32,
    )
    compressed = np.sign(raw) * np.log1p(np.abs(raw))
    return torch.tensor(compressed, dtype=torch.float32).unsqueeze(0)


def _base_batch(
    window_detections: list[WindowDetection],
    observation_length: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "trajectory_seq": _trajectory_sequence(window_detections, observation_length).to(device),
        "trajectory_static": _trajectory_static(window_detections).to(device),
        "behavior_input": torch.empty((1, 0), dtype=torch.float32, device=device),
        "behavior_valid": torch.zeros((1, observation_length), dtype=torch.float32, device=device),
        "context_vector": torch.empty((1, 0), dtype=torch.float32, device=device),
        "vehicle_vector": torch.empty((1, 0), dtype=torch.float32, device=device),
        "label": torch.zeros((1,), dtype=torch.float32, device=device),
    }


def _stage1_batch(
    window_detections: list[WindowDetection],
    observation_length: int,
    p_walking: float,
    p_looking: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = _base_batch(window_detections, observation_length, device)
    batch["behavior_input"] = torch.tensor([[p_walking, p_looking]], dtype=torch.float32, device=device)
    return batch


def run_stage2_from_stage1_csv(
    *,
    stage1_path: Path,
    output_path: Path | None = None,
    base_checkpoint: Path = DEFAULT_BASE_CHECKPOINT,
    stage1_checkpoint: Path = DEFAULT_STAGE1_CHECKPOINT,
    observation_length: int = 8,
    base_weight: float = 0.96,
    stage1_weight: float = 0.04,
    device: str = "cuda",
) -> pd.DataFrame:
    total_weight = base_weight + stage1_weight
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Blend weights must sum to 1.0, got {total_weight:.6f}")
    if observation_length <= 0:
        raise ValueError("--observation-length must be positive")

    torch_device = torch.device(device)
    stage1_predictions = _load_stage1_predictions(stage1_path)
    base_model, _ = _load_online_model(base_checkpoint, torch_device)
    stage1_model, _ = _load_online_model(stage1_checkpoint, torch_device)

    rows = []
    with torch.no_grad():
        for row in stage1_predictions.itertuples(index=False):
            window_dets = _parse_track_window(str(row.track_window_json))
            p_walking = float(row.prob_action_walking)
            p_looking = float(row.prob_look_looking)

            base_logit = float(base_model(_base_batch(window_dets, observation_length, torch_device)).cpu()[0])
            stage1_logit = float(
                stage1_model(
                    _stage1_batch(
                        window_dets,
                        observation_length,
                        p_walking,
                        p_looking,
                        torch_device,
                    )
                ).cpu()[0]
            )
            base_prob = float(torch.sigmoid(torch.tensor(base_logit)))
            stage1_prob = float(torch.sigmoid(torch.tensor(stage1_logit)))
            fused_prob = base_weight * base_prob + stage1_weight * stage1_prob

            output_row = {
                "stage1_source_path": str(stage1_path),
                "pedestrian_id": str(row.pedestrian_id),
                "window_index": int(row.window_index),
                "start_frame": int(row.start_frame),
                "end_frame": int(row.end_frame),
                "prob_action_walking": p_walking,
                "prob_look_looking": p_looking,
                "base_logit": base_logit,
                "stage1_logit": stage1_logit,
                "base_prob_crossing": base_prob,
                "stage1_aux_prob_crossing": stage1_prob,
                "prob_crossing": fused_prob,
                "pred_crossing": int(fused_prob >= 0.5),
            }
            if hasattr(row, "video_path"):
                output_row["video_path"] = str(row.video_path)
            rows.append(output_row)

    output = pd.DataFrame(rows)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
        output.attrs["output_path"] = str(output_path)
    output.attrs["stage1_source_path"] = str(stage1_path)
    return output


def main() -> None:
    args = build_argparser().parse_args()
    output = run_stage2_from_stage1_csv(
        stage1_path=args.stage1_path,
        output_path=args.output_path,
        base_checkpoint=args.base_checkpoint,
        stage1_checkpoint=args.stage1_checkpoint,
        observation_length=args.observation_length,
        base_weight=args.base_weight,
        stage1_weight=args.stage1_weight,
        device=args.device,
    )
    print(f"stage1_source={args.stage1_path}")
    print(f"exported={args.output_path}")
    print(f"windows={len(output)}")
    for row in output.itertuples(index=False):
        print(
            "window={window_index} frames={start_frame}-{end_frame} "
            "p_crossing={prob_crossing:.4f} pred_crossing={pred_crossing}".format(**row._asdict())
        )


if __name__ == "__main__":
    main()
