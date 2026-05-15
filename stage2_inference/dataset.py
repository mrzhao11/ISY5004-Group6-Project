from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("PyTorch is required for stage2.dataset") from exc

from .config import DataConfig

BEHAVIOR_TO_INDEX = {
    "walking": 0,
    "standing": 1,
    "looking": 2,
    "stopped": 3,
}

MOTION_TO_INDEX = {
    "left": 0,
    "right": 1,
    "up": 2,
    "down": 3,
}


def _feature_columns_by_prefix(table: pd.DataFrame, prefix: str) -> list[str]:
    columns = []
    for column in table.columns:
        if not column.startswith(prefix):
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        if values.notna().any():
            columns.append(column)
    return columns


@dataclass(slots=True)
class DatasetMetadata:
    trajectory_static_dim: int
    behavior_feature_dim: int
    context_feature_dim: int
    vehicle_feature_dim: int


class Stage2Dataset(Dataset):
    def __init__(self, manifest_path: Path | str, split: str, config: DataConfig):
        self.manifest_path = Path(manifest_path)
        self.config = config
        self.split = split
        self.manifest = pd.read_csv(self.manifest_path)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        if self.manifest.empty:
            raise RuntimeError(f"No samples found for split '{split}' in {self.manifest_path}")

        self.traj_static_columns = _feature_columns_by_prefix(self.manifest, "traj_") if config.use_trajectory_static else []
        self.behavior_feature_columns = _feature_columns_by_prefix(self.manifest, "behavior_")
        self.context_feature_columns = _feature_columns_by_prefix(self.manifest, "context_")
        self.vehicle_feature_columns = _feature_columns_by_prefix(self.manifest, "vehicle_")

        self.metadata = DatasetMetadata(
            trajectory_static_dim=len(self.traj_static_columns),
            behavior_feature_dim=len(self.behavior_feature_columns),
            context_feature_dim=len(self.context_feature_columns),
            vehicle_feature_dim=len(self.vehicle_feature_columns),
        )

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_track(self, track_path: str) -> pd.DataFrame:
        return pd.read_csv(
            track_path,
            usecols=[
                "sequence_id",
                "video_id",
                "pedestrian_id",
                "frame_id",
                "width",
                "height",
                "center_x",
                "center_y",
                "behavior_label",
                "crossing_label",
            ],
        )

    def _build_trajectory_sequence(self, track: pd.DataFrame) -> torch.Tensor:
        observed = track.iloc[: self.config.observation_length].copy()
        observed["dx"] = observed["center_x"].diff().fillna(0.0)
        observed["dy"] = observed["center_y"].diff().fillna(0.0)
        observed["dw"] = observed["width"].diff().fillna(0.0)
        observed["dh"] = observed["height"].diff().fillna(0.0)
        scale = float(max(observed["width"].iloc[0], observed["height"].iloc[0], 1.0))
        observed["rel_center_x"] = (observed["center_x"] - observed["center_x"].iloc[0]) / scale
        observed["rel_center_y"] = (observed["center_y"] - observed["center_y"].iloc[0]) / scale
        observed["norm_width"] = observed["width"] / scale
        observed["norm_height"] = observed["height"] / scale
        observed["norm_dx"] = observed["dx"] / scale
        observed["norm_dy"] = observed["dy"] / scale
        observed["norm_dw"] = observed["dw"] / scale
        observed["norm_dh"] = observed["dh"] / scale
        features = observed[
            ["rel_center_x", "rel_center_y", "norm_width", "norm_height", "norm_dx", "norm_dy", "norm_dw", "norm_dh"]
        ].to_numpy(dtype=np.float32)
        return torch.tensor(features, dtype=torch.float32)

    def _build_behavior_inputs(self, row: pd.Series, track: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        observed = track.iloc[: self.config.observation_length]
        indices = []
        valid_mask = []
        for label in observed["behavior_label"].fillna("unknown").str.lower():
            if label in BEHAVIOR_TO_INDEX:
                indices.append(BEHAVIOR_TO_INDEX[label])
                valid_mask.append(1.0)
            else:
                indices.append(0)
                valid_mask.append(0.0)

        behavior_indices = torch.tensor(indices, dtype=torch.long)
        behavior_valid = torch.tensor(valid_mask, dtype=torch.float32)

        if self.config.behavior_mode == "gt_behavior":
            return behavior_indices, behavior_valid

        feature_values = row[self.behavior_feature_columns].to_numpy(dtype=np.float32)
        feature_values = np.nan_to_num(feature_values, nan=0.0)
        stage1_features = torch.tensor(feature_values, dtype=torch.float32)
        return stage1_features, behavior_valid

    def _build_optional_vector(self, row: pd.Series, columns: list[str], compress: bool = False) -> torch.Tensor:
        values = row[columns].to_numpy(dtype=np.float32) if columns else np.empty((0,), dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0)
        if compress and values.size > 0:
            values = np.sign(values) * np.log1p(np.abs(values))
        return torch.tensor(values, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.manifest.iloc[index]
        track = self._load_track(str(row["track_path"]))

        trajectory_seq = self._build_trajectory_sequence(track) if self.config.use_trajectory_sequence else torch.empty((0, 8), dtype=torch.float32)
        trajectory_static = self._build_optional_vector(row, self.traj_static_columns, compress=True)
        context_vector = self._build_optional_vector(row, self.context_feature_columns)
        vehicle_vector = self._build_optional_vector(row, self.vehicle_feature_columns, compress=True)

        if self.config.behavior_mode == "disabled":
            behavior_input = torch.empty((0,), dtype=torch.float32)
            behavior_valid = torch.zeros((self.config.observation_length,), dtype=torch.float32)
        else:
            behavior_input, behavior_valid = self._build_behavior_inputs(row, track)

        return {
            "trajectory_seq": trajectory_seq,
            "trajectory_static": trajectory_static,
            "behavior_input": behavior_input,
            "behavior_valid": behavior_valid,
            "context_vector": context_vector,
            "vehicle_vector": vehicle_vector,
            "label": torch.tensor(float(row["y_intent"]), dtype=torch.float32),
        }
