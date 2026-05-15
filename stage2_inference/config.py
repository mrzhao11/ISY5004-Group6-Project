from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class DataConfig:
    processed_root: Path = Path("dataset/processed/jaad")
    tracks_dirname: str = "tracks"
    manifest_path: Path = Path("stage2/artifacts/stage2_manifest.csv")
    max_tracks: Optional[int] = None
    behavior_mode: str = "disabled"
    behavior_source_path: Optional[Path] = None
    context_source_path: Optional[Path] = None
    vehicle_source_path: Optional[Path] = None
    observation_length: int = 8
    prediction_horizon: int = 8
    min_observation_non_crossing: bool = True
    require_behavior_annotation: bool = False
    allow_unknown_behavior_in_future: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    random_seed: int = 42
    use_trajectory_sequence: bool = True
    use_trajectory_static: bool = True


@dataclass(slots=True)
class ModelConfig:
    architecture: str = "tcl"
    trajectory_encoder: str = "gru"
    behavior_encoder: str = "gru"
    behavior_fusion: str = "token"
    hidden_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.1
    behavior_num_classes: int = 4
    stage1_behavior_dim: int = 256
    context_dim: int = 0
    vehicle_dim: int = 0


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    log_every: int = 20


@dataclass(slots=True)
class Stage2Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
