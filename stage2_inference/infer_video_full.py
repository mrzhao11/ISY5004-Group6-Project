from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stage1_inference.infer_video import (
    _build_windows,
    _crop_track_frames,
    _detect_with_yolo,
    _load_checkpoint,
    _load_models,
    _predict_window,
    _read_detections_csv,
    _select_track,
)

from .infer_best import (
    DEFAULT_BASE_CHECKPOINT,
    DEFAULT_BASE_MANIFEST,
    DEFAULT_STAGE1_CHECKPOINT,
    DEFAULT_STAGE1_MANIFEST,
    _load_model,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run raw video -> Stage 1 -> Stage 2 T42 inference")
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--pedestrian-id", type=str, default=None)
    parser.add_argument("--detections-csv", type=Path, default=None)
    parser.add_argument("--yolo-model-path", type=Path, default=Path("models/stage1/person_detector.pt"))
    parser.add_argument("--action-checkpoint-path", type=Path, default=Path("models/stage1/action_sequence_swin3d_t.pt"))
    parser.add_argument("--look-checkpoint-path", type=Path, default=Path("models/stage1/look_frame_swin_t.pt"))
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--stage1-checkpoint", type=Path, default=DEFAULT_STAGE1_CHECKPOINT)
    parser.add_argument("--base-manifest-path", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--stage1-manifest-path", type=Path, default=DEFAULT_STAGE1_MANIFEST)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--observation-length", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--look-max-frames", type=int, default=15)
    parser.add_argument("--det-conf", type=float, default=0.35)
    parser.add_argument("--track-iou-threshold", type=float, default=0.30)
    parser.add_argument("--base-weight", type=float, default=0.96)
    parser.add_argument("--stage1-weight", type=float, default=0.04)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-path", type=Path, default=Path("outputs/predictions/full_video_inference.csv"))
    return parser


def _window_detections(detections, pedestrian_id: str, frame_ids: list[int]):
    by_frame = {det.frame_id: det for det in detections if det.pedestrian_id == pedestrian_id}
    return [by_frame[frame_id] for frame_id in frame_ids]


def _trajectory_sequence(window_detections, observation_length: int) -> torch.Tensor:
    dets = window_detections[:observation_length]
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


def _trajectory_static(window_detections) -> torch.Tensor:
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


def _base_batch(window_detections, observation_length: int, device: str) -> dict[str, torch.Tensor]:
    return {
        "trajectory_seq": _trajectory_sequence(window_detections, observation_length).to(device),
        "trajectory_static": _trajectory_static(window_detections).to(device),
        "behavior_input": torch.empty((1, 0), dtype=torch.float32, device=device),
        "behavior_valid": torch.zeros((1, observation_length), dtype=torch.float32, device=device),
        "context_vector": torch.empty((1, 0), dtype=torch.float32, device=device),
        "vehicle_vector": torch.empty((1, 0), dtype=torch.float32, device=device),
        "label": torch.zeros((1,), dtype=torch.float32, device=device),
    }


def _stage1_batch(window_detections, observation_length: int, p_walking: float, p_looking: float, device: str):
    batch = _base_batch(window_detections, observation_length, device)
    batch["behavior_input"] = torch.tensor([[p_walking, p_looking]], dtype=torch.float32, device=device)
    return batch


def main() -> None:
    args = build_argparser().parse_args()
    if args.detections_csv is None and args.yolo_model_path is None:
        raise ValueError("Provide --detections-csv or --yolo-model-path")
    if args.clip_length < args.observation_length:
        raise ValueError("--clip-length must be >= --observation-length")

    action_checkpoint = _load_checkpoint(args.action_checkpoint_path, args.device)
    look_checkpoint = _load_checkpoint(args.look_checkpoint_path, args.device)
    detections = (
        _read_detections_csv(args.detections_csv)
        if args.detections_csv is not None
        else _detect_with_yolo(args.video_path, args.yolo_model_path, args.det_conf, args.track_iou_threshold)
    )
    pedestrian_id = _select_track(detections, args.pedestrian_id)
    base_model, _ = _load_model(args.base_checkpoint, args.base_manifest_path, torch.device(args.device))
    stage1_model, _ = _load_model(args.stage1_checkpoint, args.stage1_manifest_path, torch.device(args.device))

    with tempfile.TemporaryDirectory(prefix="full_video_crops_") as tmp_dir:
        crops = _crop_track_frames(args.video_path, detections, pedestrian_id, Path(tmp_dir))
        windows = _build_windows(crops, args.clip_length, args.window_stride)
        action_model, look_model, action_data_config, look_data_config = _load_models(
            args, action_checkpoint, look_checkpoint
        )
        rows = []
        with torch.no_grad():
            for window_index, window in enumerate(windows):
                stage1_row = _predict_window(
                    [crop.path for crop in window],
                    args,
                    action_model,
                    look_model,
                    action_data_config,
                    look_data_config,
                )
                window_dets = _window_detections(detections, pedestrian_id, [crop.frame_id for crop in window])
                base_logit = float(base_model(_base_batch(window_dets, args.observation_length, args.device)).cpu()[0])
                aux_logit = float(
                    stage1_model(
                        _stage1_batch(
                            window_dets,
                            args.observation_length,
                            float(stage1_row["prob_action_walking"]),
                            float(stage1_row["prob_look_looking"]),
                            args.device,
                        )
                    ).cpu()[0]
                )
                base_prob = float(torch.sigmoid(torch.tensor(base_logit)))
                aux_prob = float(torch.sigmoid(torch.tensor(aux_logit)))
                fused_prob = args.base_weight * base_prob + args.stage1_weight * aux_prob
                rows.append(
                    {
                        "video_path": str(args.video_path),
                        "pedestrian_id": pedestrian_id,
                        "window_index": window_index,
                        "start_frame": window[0].frame_id,
                        "end_frame": window[-1].frame_id,
                        "prob_action_walking": stage1_row["prob_action_walking"],
                        "prob_look_looking": stage1_row["prob_look_looking"],
                        "base_prob_crossing": base_prob,
                        "stage1_aux_prob_crossing": aux_prob,
                        "prob_crossing": fused_prob,
                        "pred_crossing": int(fused_prob >= 0.5),
                    }
                )

    output = pd.DataFrame(rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)
    print(f"exported={args.output_path}")
    print(f"pedestrian_id={pedestrian_id}")
    print(f"windows={len(output)}")
    for row in output.itertuples(index=False):
        print(
            "window={window_index} frames={start_frame}-{end_frame} "
            "p_walking={prob_action_walking:.4f} p_looking={prob_look_looking:.4f} "
            "p_crossing={prob_crossing:.4f} pred_crossing={pred_crossing}".format(**row._asdict())
        )


if __name__ == "__main__":
    main()
