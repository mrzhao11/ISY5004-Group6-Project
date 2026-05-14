from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig, ModelConfig

ACTION_NAMES = ("standing", "walking")
LOOK_NAMES = ("not_looking", "looking")


def _track_sequence_id(sequence_id: str) -> str:
    parts = str(sequence_id).split("__")
    if len(parts) >= 3 and "_" in parts[-1]:
        return "__".join(parts[:-1])
    return str(sequence_id)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 1 inference with a sequence action model and a frame look model."
    )
    parser.add_argument(
        "--action-checkpoint-path",
        type=Path,
        default=Path("models/stage1/action_sequence_swin3d_t.pt"),
        help="Sequence model checkpoint for action standing/walking.",
    )
    parser.add_argument(
        "--look-checkpoint-path",
        type=Path,
        default=Path("models/stage1/look_frame_swin_t.pt"),
        help="Single-frame model checkpoint for look not-looking/looking.",
    )
    parser.add_argument("--action-manifest-path", type=Path, default=Path("stage1/artifacts/stage1_manifest.csv"))
    parser.add_argument(
        "--frame-manifest-path",
        type=Path,
        default=Path("dataset/processed/jaad_frame_stage1/frame_manifest.csv"),
    )
    parser.add_argument("--sequence-source-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=Path("stage1/artifacts/stage1_best_inference.csv"))
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--action-sequence-region", choices=["full", "action", "look"], default="full")
    parser.add_argument("--look-max-frames-per-sequence", type=int, default=0)
    parser.add_argument("--look-frame-sampling", choices=["first", "uniform", "random"], default="uniform")
    parser.add_argument("--action-batch-size", type=int, default=8)
    parser.add_argument("--look-batch-size", type=int, default=128)
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
    path_fields = {"processed_root", "manifest_path"}
    for key, value in values.items():
        if hasattr(config, key):
            setattr(config, key, Path(value) if key in path_fields else value)
    config.augment = False
    return config


def _load_model_config(values: dict) -> ModelConfig:
    config = ModelConfig()
    for key, value in values.items():
        if hasattr(config, key):
            setattr(config, key, value)
    # Checkpoints already contain trained backbone weights. Avoid network downloads
    # from torchvision during inference in demo/deployment environments.
    config.pretrained = False
    return config


def _filter_manifest(source_path: Path, sequence_ids: set[str], output_path: Path) -> Path:
    manifest = pd.read_csv(source_path)
    filtered = manifest[manifest["sequence_id"].astype(str).isin(sequence_ids)]
    if filtered.empty:
        raise RuntimeError(f"No rows in {source_path} match the requested sequence ids")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_path, index=False)
    return output_path


def _select_action_clip(batch: dict[str, object], region: str):
    if region == "full":
        return batch["clip"]
    if region == "action":
        return batch["action_clip"]
    if region == "look":
        return batch["look_clip"]
    raise ValueError(f"Unsupported action sequence region: {region}")


def _forward_action_model(model, batch: dict[str, object], region: str):
    if getattr(model, "expects_regions", False):
        return model(batch["action_clip"], batch["look_clip"])
    return model(_select_action_clip(batch, region))


def _predict_action_sequences(
    checkpoint: dict,
    manifest_path: Path,
    split: str,
    batch_size: int,
    device: str,
    sequence_region: str,
) -> dict[str, dict[str, np.ndarray]]:
    import torch
    from torch.utils.data import DataLoader

    from .dataset import Stage1Dataset
    from .models import build_model

    data_config = _load_data_config(checkpoint["data_config"])
    data_config.manifest_path = manifest_path
    model_config = _load_model_config(checkpoint["model_config"])
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    splits = ["train", "val", "test"] if split == "all" else [split]
    outputs: dict[str, dict[str, np.ndarray | str]] = {}
    with torch.no_grad():
        for current_split in splits:
            dataset = Stage1Dataset(manifest_path, current_split, data_config)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for batch in loader:
                moved = {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
                action_logits, _, embeddings = _forward_action_model(model, moved, sequence_region)
                probs = torch.softmax(action_logits, dim=1).detach().cpu().numpy()
                embeddings_np = embeddings.detach().cpu().numpy()
                for idx, sequence_id in enumerate(batch["sequence_id"]):
                    sequence_id = str(sequence_id)
                    outputs[str(sequence_id)] = {
                        "sequence_id": sequence_id,
                        "track_sequence_id": _track_sequence_id(sequence_id),
                        "probs": probs[idx],
                        "embedding": embeddings_np[idx],
                    }
    return outputs


def _predict_look_frames(
    checkpoint: dict,
    manifest_path: Path,
    split: str,
    batch_size: int,
    device: str,
    max_frames_per_sequence: int,
    frame_sampling: str,
) -> dict[str, dict[str, np.ndarray]]:
    import torch
    from torch.utils.data import DataLoader

    from .frame_dataset import FrameTaskDataset
    from .models import build_single_frame_model

    data_config = _load_data_config(checkpoint["data_config"])
    data_config.manifest_path = manifest_path
    data_config.max_frames_per_track = max_frames_per_sequence
    data_config.frame_sampling = frame_sampling
    model_config = _load_model_config(checkpoint["model_config"])
    model = build_single_frame_model(model_config, "look").to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    raw_outputs: dict[str, dict[str, list[np.ndarray]]] = {}
    splits = ["train", "val", "test"] if split == "all" else [split]
    with torch.no_grad():
        for current_split in splits:
            dataset = FrameTaskDataset(manifest_path, current_split, data_config, "look")
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for batch in loader:
                images = batch["image"].to(device)
                image_aux = batch.get("image_aux")
                if image_aux is None:
                    logits, embeddings = model(images)
                else:
                    logits, embeddings = model(images, image_aux.to(device))
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                embeddings_np = embeddings.detach().cpu().numpy()
                for idx, sequence_id in enumerate(batch["sequence_id"]):
                    record = raw_outputs.setdefault(_track_sequence_id(str(sequence_id)), {"probs": [], "embeddings": []})
                    record["probs"].append(probs[idx])
                    record["embeddings"].append(embeddings_np[idx])

    return {
        sequence_id: {
            "probs": np.mean(record["probs"], axis=0),
            "embedding": np.mean(record["embeddings"], axis=0),
        }
        for sequence_id, record in raw_outputs.items()
    }


def main() -> None:
    args = build_argparser().parse_args()

    action_checkpoint = _load_checkpoint(args.action_checkpoint_path, args.device)
    look_checkpoint = _load_checkpoint(args.look_checkpoint_path, args.device)

    action_manifest_path = args.action_manifest_path
    frame_manifest_path = args.frame_manifest_path
    if args.sequence_source_path is not None:
        sequence_ids = set(pd.read_csv(args.sequence_source_path)["sequence_id"].astype(str))
        action_manifest_path = _filter_manifest(
            args.action_manifest_path,
            sequence_ids,
            args.output_path.parent / "_infer_action_manifest_filtered.csv",
        )
        frame_manifest_path = _filter_manifest(
            args.frame_manifest_path,
            sequence_ids,
            args.output_path.parent / "_infer_frame_manifest_filtered.csv",
        )

    action_outputs = _predict_action_sequences(
        action_checkpoint,
        action_manifest_path,
        args.split,
        args.action_batch_size,
        args.device,
        args.action_sequence_region,
    )
    look_outputs = _predict_look_frames(
        look_checkpoint,
        frame_manifest_path,
        args.split,
        args.look_batch_size,
        args.device,
        args.look_max_frames_per_sequence,
        args.look_frame_sampling,
    )

    action_by_track: dict[str, dict[str, np.ndarray | str]] = {}
    for action_sequence_id, record in action_outputs.items():
        action_by_track[f"{record['track_sequence_id']}::{action_sequence_id}"] = record

    rows: list[dict[str, object]] = []
    missing_look = 0
    for _, action_record in sorted(action_by_track.items()):
        action_sequence_id = str(action_record["sequence_id"])
        track_sequence_id = str(action_record["track_sequence_id"])
        if track_sequence_id not in look_outputs:
            missing_look += 1
            continue
        look_record = look_outputs[track_sequence_id]
        action_probs = action_record["probs"]
        look_probs = look_record["probs"]
        action_embedding = action_record["embedding"]
        look_embedding = look_record["embedding"]
        behavior_embedding = np.concatenate([action_embedding, look_embedding], axis=0)

        row: dict[str, object] = {
            "sequence_id": action_sequence_id,
            "track_sequence_id": track_sequence_id,
            "action_pred": ACTION_NAMES[int(np.argmax(action_probs))],
            "look_pred": LOOK_NAMES[int(np.argmax(look_probs))],
        }
        for idx, name in enumerate(ACTION_NAMES):
            row[f"prob_action_{name}"] = float(action_probs[idx])
        for idx, name in enumerate(LOOK_NAMES):
            row[f"prob_look_{name}"] = float(look_probs[idx])
        for idx, value in enumerate(behavior_embedding):
            row[f"embedding_{idx}"] = float(value)
        rows.append(row)

    output = pd.DataFrame(rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_path, index=False)
    print(f"exported={args.output_path}")
    print(f"rows={len(output)}")
    print(f"missing_look_for_action_sequences={missing_look}")
    print(f"action_sequences={len(action_outputs)}")
    print(f"look_tracks={len(look_outputs)}")


if __name__ == "__main__":
    main()
