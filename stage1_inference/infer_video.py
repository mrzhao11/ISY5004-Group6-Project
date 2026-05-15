from __future__ import annotations

import argparse
import csv
import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import DataConfig, ModelConfig
from .dataset import _load_frame
from .infer_best import ACTION_NAMES, LOOK_NAMES, _load_model_config
from .models import build_model, build_single_frame_model


@dataclass(slots=True)
class Detection:
    frame_id: int
    pedestrian_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0


@dataclass(slots=True)
class TrackCrop:
    frame_id: int
    path: Path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 1 inference directly on a raw video.")
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--pedestrian-id", type=str, default=None)
    parser.add_argument("--detections-csv", type=Path, default=None)
    parser.add_argument("--yolo-model-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=Path("outputs/predictions/stage1_video_inference.csv"))
    parser.add_argument("--action-checkpoint-path", type=Path, default=Path("models/stage1/action_sequence_swin3d_t.pt"))
    parser.add_argument("--look-checkpoint-path", type=Path, default=Path("models/stage1/look_frame_swin_t.pt"))
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--look-max-frames", type=int, default=15)
    parser.add_argument("--det-conf", type=float, default=0.35)
    parser.add_argument("--track-iou-threshold", type=float, default=0.30)
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def _load_checkpoint(path: Path, device: str) -> dict:
    import torch

    try:
        return torch.load(path, map_location=device)
    except pickle.UnpicklingError:
        return torch.load(path, map_location=device, weights_only=False)


def _load_data_config(values: dict) -> DataConfig:
    config = DataConfig()
    for key, value in values.items():
        if hasattr(config, key):
            setattr(config, key, Path(value) if key in {"processed_root", "manifest_path"} else value)
    config.augment = False
    return config


def _read_detections_csv(path: Path) -> list[Detection]:
    detections: list[Detection] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_id", "pedestrian_id", "x1", "y1", "x2", "y2"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            detections.append(
                Detection(
                    frame_id=int(row["frame_id"]),
                    pedestrian_id=str(row["pedestrian_id"]),
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                    x2=float(row["x2"]),
                    y2=float(row["y2"]),
                    score=float(row.get("score") or 1.0),
                )
            )
    return detections


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def _detect_with_yolo(video_path: Path, yolo_model_path: Path, conf: float, iou_threshold: float) -> list[Detection]:
    # Keep Ultralytics settings inside the project instead of depending on a user-profile path.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("outputs") / "ultralytics"))
    try:
        import cv2
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Raw video detection requires opencv-python and ultralytics, or provide --detections-csv."
        ) from exc

    model = YOLO(str(yolo_model_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    tracks: dict[str, tuple[float, float, float, float]] = {}
    detections: list[Detection] = []
    next_track_id = 1
    frame_id = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        result = model.predict(frame_bgr, conf=conf, classes=[0], verbose=False)[0]
        frame_boxes: list[tuple[float, float, float, float, float]] = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].detach().cpu().tolist()]
                score = float(box.conf[0].detach().cpu())
                frame_boxes.append((x1, y1, x2, y2, score))

        used_tracks: set[str] = set()
        new_tracks: dict[str, tuple[float, float, float, float]] = {}
        for x1, y1, x2, y2, score in frame_boxes:
            bbox = (x1, y1, x2, y2)
            best_track_id = None
            best_iou = 0.0
            for track_id, previous_bbox in tracks.items():
                if track_id in used_tracks:
                    continue
                overlap = _iou(bbox, previous_bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_track_id = track_id
            if best_track_id is None or best_iou < iou_threshold:
                best_track_id = f"ped_{next_track_id:03d}"
                next_track_id += 1
            used_tracks.add(best_track_id)
            new_tracks[best_track_id] = bbox
            detections.append(Detection(frame_id, best_track_id, x1, y1, x2, y2, score))
        tracks = new_tracks
        frame_id += 1
    cap.release()
    return detections


def _select_track(detections: list[Detection], pedestrian_id: str | None) -> str:
    if pedestrian_id is not None:
        if any(det.pedestrian_id == pedestrian_id for det in detections):
            return pedestrian_id
        raise RuntimeError(f"Pedestrian id '{pedestrian_id}' was not found in detections")
    counts = pd.Series([det.pedestrian_id for det in detections]).value_counts()
    if counts.empty:
        raise RuntimeError("No pedestrian detections found")
    return str(counts.index[0])


def _crop_track_frames(video_path: Path, detections: list[Detection], pedestrian_id: str, output_dir: Path) -> list[TrackCrop]:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("opencv-python is required for raw video frame extraction") from exc

    by_frame = {det.frame_id: det for det in detections if det.pedestrian_id == pedestrian_id}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    crop_paths: list[TrackCrop] = []
    frame_id = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_id in by_frame:
            det = by_frame[frame_id]
            height, width = frame_bgr.shape[:2]
            x1 = max(0, min(width - 1, int(round(det.x1))))
            y1 = max(0, min(height - 1, int(round(det.y1))))
            x2 = max(x1 + 1, min(width, int(round(det.x2))))
            y2 = max(y1 + 1, min(height, int(round(det.y2))))
            crop_bgr = frame_bgr[y1:y2, x1:x2]
            crop_path = output_dir / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(crop_path), crop_bgr)
            crop_paths.append(TrackCrop(frame_id=frame_id, path=crop_path))
        frame_id += 1
    cap.release()
    if not crop_paths:
        raise RuntimeError(f"No crops were generated for pedestrian id {pedestrian_id}")
    return crop_paths


def _uniform_select(paths: list[Path], count: int) -> list[Path]:
    if count <= 0 or len(paths) <= count:
        return paths
    indices = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[int(index)] for index in indices]


def _load_clip_tensor(paths: list[Path], data_config: DataConfig, clip_length: int, device: str):
    import torch

    selected = _uniform_select(paths, clip_length)
    if len(selected) < clip_length:
        selected = selected + [selected[-1]] * (clip_length - len(selected))
    frames = [
        _load_frame(path, data_config.image_size, data_config.resize_mode, data_config.pad_mode, "full", False)
        for path in selected[:clip_length]
    ]
    return torch.tensor(np.stack(frames, axis=0), dtype=torch.float32).unsqueeze(0).to(device)


def _load_frame_batch(paths: list[Path], data_config: DataConfig, max_frames: int, device: str):
    import torch

    selected = _uniform_select(paths, max_frames)
    frames = [
        _load_frame(path, data_config.image_size, data_config.resize_mode, data_config.pad_mode, data_config.look_region, False)
        for path in selected
    ]
    return torch.tensor(np.stack(frames, axis=0), dtype=torch.float32).to(device)


def _build_windows(crops: list[TrackCrop], clip_length: int, stride: int) -> list[list[TrackCrop]]:
    crops = sorted(crops, key=lambda item: item.frame_id)
    if len(crops) <= clip_length:
        return [crops]
    stride = max(1, stride)
    windows = [crops[start : start + clip_length] for start in range(0, len(crops) - clip_length + 1, stride)]
    if windows[-1][-1].frame_id != crops[-1].frame_id:
        windows.append(crops[-clip_length:])
    return windows


def _predict_window(
    crop_paths: list[Path],
    args: argparse.Namespace,
    action_model,
    look_model,
    action_data_config: DataConfig,
    look_data_config: DataConfig,
) -> dict[str, object]:
    import torch

    with torch.no_grad():
        action_clip = _load_clip_tensor(crop_paths, action_data_config, args.clip_length, args.device)
        action_logits, _, action_embedding = action_model(action_clip)
        action_probs = torch.softmax(action_logits, dim=1)[0].detach().cpu().numpy()

        look_batch = _load_frame_batch(crop_paths, look_data_config, args.look_max_frames, args.device)
        look_logits, look_embeddings = look_model(look_batch)
        look_probs = torch.softmax(look_logits, dim=1).mean(dim=0).detach().cpu().numpy()
        look_embedding = look_embeddings.mean(dim=0).detach().cpu().numpy()
        embedding = np.concatenate([action_embedding[0].detach().cpu().numpy(), look_embedding], axis=0)

    row: dict[str, object] = {
        "action_pred": ACTION_NAMES[int(np.argmax(action_probs))],
        "look_pred": LOOK_NAMES[int(np.argmax(look_probs))],
        "prob_action_standing": float(action_probs[0]),
        "prob_action_walking": float(action_probs[1]),
        "prob_look_not_looking": float(look_probs[0]),
        "prob_look_looking": float(look_probs[1]),
    }
    for idx, value in enumerate(embedding):
        row[f"embedding_{idx}"] = float(value)
    return row


def _load_models(args: argparse.Namespace, action_checkpoint: dict, look_checkpoint: dict):
    action_data_config = _load_data_config(action_checkpoint["data_config"])
    look_data_config = _load_data_config(look_checkpoint["data_config"])
    action_model_config = _load_model_config(action_checkpoint["model_config"])
    look_model_config = _load_model_config(look_checkpoint["model_config"])
    action_model_config.pretrained = False
    look_model_config.pretrained = False

    action_model = build_model(action_model_config).to(args.device)
    action_model.load_state_dict(action_checkpoint["model_state_dict"])
    action_model.eval()
    look_model = build_single_frame_model(look_model_config, "look").to(args.device)
    look_model.load_state_dict(look_checkpoint["model_state_dict"])
    look_model.eval()
    return action_model, look_model, action_data_config, look_data_config


def main() -> None:
    args = build_argparser().parse_args()
    if args.detections_csv is None and args.yolo_model_path is None:
        raise ValueError("Provide either --detections-csv or --yolo-model-path for raw video inference")

    action_checkpoint = _load_checkpoint(args.action_checkpoint_path, args.device)
    look_checkpoint = _load_checkpoint(args.look_checkpoint_path, args.device)
    detections = (
        _read_detections_csv(args.detections_csv)
        if args.detections_csv is not None
        else _detect_with_yolo(args.video_path, args.yolo_model_path, args.det_conf, args.track_iou_threshold)
    )
    pedestrian_id = _select_track(detections, args.pedestrian_id)

    with tempfile.TemporaryDirectory(prefix="stage1_video_crops_") as tmp_dir:
        crops = _crop_track_frames(args.video_path, detections, pedestrian_id, Path(tmp_dir))
        windows = _build_windows(crops, args.clip_length, args.window_stride)
        action_model, look_model, action_data_config, look_data_config = _load_models(
            args,
            action_checkpoint,
            look_checkpoint,
        )
        rows = []
        for window_index, window in enumerate(windows):
            row = _predict_window(
                [crop.path for crop in window],
                args,
                action_model,
                look_model,
                action_data_config,
                look_data_config,
            )
            row.update(
                {
                    "video_path": str(args.video_path),
                    "pedestrian_id": pedestrian_id,
                    "window_index": window_index,
                    "start_frame": window[0].frame_id,
                    "end_frame": window[-1].frame_id,
                    "num_window_frames": len(window),
                    "num_track_frames": len(crops),
                }
            )
            rows.append(row)

    output = pd.DataFrame(rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)
    print(f"exported={args.output_path}")
    print(f"pedestrian_id={pedestrian_id}")
    print(f"windows={len(output)}")
    for row in output.itertuples(index=False):
        print(
            "window={window_index} frames={start_frame}-{end_frame} "
            "action={action_pred} p_walking={prob_action_walking:.4f} "
            "look={look_pred} p_looking={prob_look_looking:.4f}".format(
                window_index=row.window_index,
                start_frame=row.start_frame,
                end_frame=row.end_frame,
                action_pred=row.action_pred,
                prob_action_walking=row.prob_action_walking,
                look_pred=row.look_pred,
                prob_look_looking=row.prob_look_looking,
            )
        )


if __name__ == "__main__":
    main()
