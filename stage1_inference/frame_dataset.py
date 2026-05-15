from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("PyTorch is required for stage1.frame_dataset") from exc

from .config import DataConfig
from .dataset import ACTION_TO_INDEX, LOOK_TO_INDEX, _load_frame


@dataclass(slots=True)
class FrameSample:
    sequence_id: str
    frame_path: Path
    label: int


class FrameTaskDataset(Dataset):
    def __init__(self, manifest_path: Path | str, split: str, config: DataConfig, task: str):
        self.config = config
        self.task = task
        manifest = pd.read_csv(manifest_path)
        self.manifest = manifest[manifest["split"] == split].reset_index(drop=True)
        if self.manifest.empty:
            raise RuntimeError(f"No samples found for split '{split}'")
        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(f"No frame samples found for split '{split}' and task '{task}'")

    def _label_index(self, label: str) -> int | None:
        if self.task == "action":
            return ACTION_TO_INDEX.get(label)
        if self.task == "look":
            return LOOK_TO_INDEX.get(label)
        raise ValueError(f"Unsupported task: {self.task}")

    def _region(self) -> str:
        return self.config.action_region if self.task == "action" else self.config.look_region

    def _select_rows(self, rows: list[dict[str, str]], sequence_id: str) -> list[dict[str, str]]:
        max_rows = self.config.max_frames_per_track or self.config.clip_length
        if len(rows) <= max_rows:
            return rows
        if self.config.frame_sampling == "first":
            return rows[:max_rows]
        if self.config.frame_sampling == "uniform":
            indices = np.linspace(0, len(rows) - 1, max_rows).round().astype(int)
            return [rows[int(index)] for index in indices]
        if self.config.frame_sampling == "random":
            seed = int(hashlib.md5(sequence_id.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            indices = sorted(rng.choice(len(rows), size=max_rows, replace=False).tolist())
            return [rows[index] for index in indices]
        raise ValueError(f"Unsupported frame_sampling: {self.config.frame_sampling}")

    def _build_samples(self) -> list[FrameSample]:
        if "crop_path" in self.manifest.columns:
            return self._build_direct_frame_samples()
        return self._build_sequence_frame_samples()

    def _build_direct_frame_samples(self) -> list[FrameSample]:
        samples: list[FrameSample] = []
        label_column = "action_label" if self.task == "action" else "look_label"
        rows = [row._asdict() for row in self.manifest.itertuples(index=False)]
        if self.config.max_frames_per_track > 0:
            grouped_rows: dict[str, list[dict[str, object]]] = {}
            for row_data in rows:
                label = str(row_data.get(label_column, "unknown")).strip().lower()
                sequence_id = str(row_data.get("sequence_id") or row_data.get("frame_sample_id") or "")
                grouped_rows.setdefault(f"{sequence_id}__{label}", []).append(row_data)
            rows = []
            for group_key, group in grouped_rows.items():
                group = sorted(group, key=lambda item: int(item.get("frame_id", 0)))
                rows.extend(self._select_rows(group, group_key))

        for row_data in rows:
            label = str(row_data.get(label_column, "unknown")).strip().lower()
            label_index = self._label_index(label)
            if label_index is None:
                continue
            frame_path = Path(str(row_data["crop_path"]))
            if not frame_path.exists():
                continue
            sequence_id = str(row_data.get("sequence_id") or row_data.get("frame_sample_id") or frame_path.stem)
            samples.append(FrameSample(sequence_id, frame_path, label_index))
        return samples

    def _build_sequence_frame_samples(self) -> list[FrameSample]:
        samples: list[FrameSample] = []
        label_column = "action_label" if self.task == "action" else "look_label"
        for row in self.manifest.itertuples(index=False):
            crop_dir = Path(str(row.crop_dir))
            with Path(str(row.track_path)).open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                track_rows = list(reader)
                for track_row in self._select_rows(track_rows, str(row.sequence_id)):
                    label = (track_row.get(label_column) or "unknown").strip().lower()
                    label_index = self._label_index(label)
                    if label_index is None:
                        continue
                    frame_id = int(track_row["frame_id"])
                    frame_path = crop_dir / f"frame_{frame_id:06d}.jpg"
                    if not frame_path.exists():
                        continue
                    samples.append(FrameSample(str(row.sequence_id), frame_path, label_index))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        frame = _load_frame(
            sample.frame_path,
            self.config.image_size,
            self.config.resize_mode,
            self.config.pad_mode,
            self._region(),
            self.config.augment,
        )
        item = {
            "sequence_id": sample.sequence_id,
            "image": torch.tensor(frame, dtype=torch.float32),
            "label": torch.tensor(sample.label, dtype=torch.long),
        }
        if self.config.aux_region != "none":
            aux_frame = _load_frame(
                sample.frame_path,
                self.config.image_size,
                self.config.resize_mode,
                self.config.pad_mode,
                self.config.aux_region,
                self.config.augment,
            )
            item["image_aux"] = torch.tensor(aux_frame, dtype=torch.float32)
        return item
