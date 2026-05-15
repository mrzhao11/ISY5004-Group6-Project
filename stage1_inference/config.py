from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class DataConfig:
    processed_root: Path = Path("dataset/processed/jaad")
    tracks_dirname: str = "tracks"
    crops_dirname: str = "crops"
    manifest_path: Path = Path("stage1/artifacts/stage1_manifest.csv")
    max_tracks: Optional[int] = None
    clip_length: int = 16
    max_frames_per_track: int = 0
    image_size: int = 112
    resize_mode: str = "letterbox"
    pad_mode: str = "mean"
    action_region: str = "bottom_half"
    look_region: str = "top_third"
    aux_region: str = "none"
    augment: bool = False
    frame_sampling: str = "first"
    label_mode: str = "majority"
    require_behavior_annotation: bool = True
    require_look_annotation: bool = False
    train_ratio: float = 0.7
    val_ratio: float = 0.15


@dataclass(slots=True)
class ModelConfig:
    architecture: str = "video_swin"
    embedding_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.1
    num_heads: int = 4
    num_layers: int = 2
    num_action_classes: int = 2
    num_look_classes: int = 2
    pretrained: bool = False
    freeze_backbone: bool = False
    freeze_backbone_until: str = "none"


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    output_dir: Path = Path("stage1/artifacts/checkpoints")


@dataclass(slots=True)
class Stage1Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
