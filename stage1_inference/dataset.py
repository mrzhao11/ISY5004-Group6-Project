from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("PyTorch is required for stage1.dataset") from exc

from .config import DataConfig

ACTION_TO_INDEX = {
    "standing": 0,
    "walking": 1,
}
LOOK_TO_INDEX = {
    "not-looking": 0,
    "looking": 1,
}
INDEX_TO_ACTION = {value: key for key, value in ACTION_TO_INDEX.items()}
INDEX_TO_LOOK = {value: key for key, value in LOOK_TO_INDEX.items()}


@dataclass(slots=True)
class DatasetMetadata:
    num_action_classes: int
    num_look_classes: int


def _region_crop(image: Image.Image, region: str) -> Image.Image:
    width, height = image.size
    if region == "full":
        return image
    if region == "top_third":
        return image.crop((0, 0, width, max(1, height // 3)))
    if region == "bottom_half":
        return image.crop((0, height // 2, width, height))
    if region == "top_half":
        return image.crop((0, 0, width, max(1, height // 2)))
    raise ValueError(f"Unsupported region: {region}")


def _letterbox_resize(image: Image.Image, image_size: int, pad_mode: str) -> Image.Image:
    width, height = image.size
    scale = min(image_size / width, image_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = image.resize((new_width, new_height))
    if pad_mode == "black":
        fill = (0, 0, 0)
    elif pad_mode == "mean":
        mean = np.asarray(image, dtype=np.float32).reshape(-1, 3).mean(axis=0)
        fill = tuple(int(x) for x in mean)
    else:
        raise ValueError(f"Unsupported pad_mode: {pad_mode}")
    canvas = Image.new("RGB", (image_size, image_size), fill)
    left = (image_size - new_width) // 2
    top = (image_size - new_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _load_frame(
    path: Path,
    image_size: int,
    resize_mode: str,
    pad_mode: str,
    region: str,
    augment: bool = False,
) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    image = _region_crop(image, region)
    if augment and min(image.size) > 8:
        width, height = image.size
        scale = float(np.random.uniform(0.9, 1.0))
        crop_width = max(1, int(round(width * scale)))
        crop_height = max(1, int(round(height * scale)))
        left = int(np.random.randint(0, max(1, width - crop_width + 1)))
        top = int(np.random.randint(0, max(1, height - crop_height + 1)))
        image = image.crop((left, top, left + crop_width, top + crop_height))
    if resize_mode == "letterbox":
        image = _letterbox_resize(image, image_size, pad_mode)
    elif resize_mode == "stretch":
        image = image.resize((image_size, image_size))
    else:
        raise ValueError(f"Unsupported resize_mode: {resize_mode}")
    if augment and np.random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if augment:
        image = ImageEnhance.Contrast(image).enhance(float(np.random.uniform(0.9, 1.1)))
    array = np.asarray(image, dtype=np.float32) / 255.0
    if augment:
        factor = float(np.random.uniform(0.85, 1.15))
        array = np.clip(array * factor, 0.0, 1.0)
    array = np.transpose(array, (2, 0, 1))
    return array


class Stage1Dataset(Dataset):
    def __init__(self, manifest_path: Path | str, split: str, config: DataConfig):
        self.manifest = pd.read_csv(manifest_path)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        if self.manifest.empty:
            raise RuntimeError(f"No Stage 1 samples found for split '{split}'")
        self.config = config
        self.metadata = DatasetMetadata(num_action_classes=len(ACTION_TO_INDEX), num_look_classes=len(LOOK_TO_INDEX))

    def __len__(self) -> int:
        return len(self.manifest)

    def _load_clip(self, crop_dir: Path, region: str) -> torch.Tensor:
        frame_paths = sorted(crop_dir.glob("*.jpg"))[: self.config.clip_length]
        if len(frame_paths) < self.config.clip_length:
            raise RuntimeError(f"{crop_dir} has fewer than {self.config.clip_length} frames")
        clip = np.stack(
            [
                _load_frame(
                    path,
                    self.config.image_size,
                    self.config.resize_mode,
                    self.config.pad_mode,
                    region,
                    self.config.augment,
                )
                for path in frame_paths
            ],
            axis=0,
        )
        return torch.tensor(clip, dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.manifest.iloc[index]
        crop_dir = Path(str(row["crop_dir"]))
        clip = self._load_clip(crop_dir, "full")
        action_clip = self._load_clip(crop_dir, self.config.action_region)
        look_clip = self._load_clip(crop_dir, self.config.look_region)
        action_label = ACTION_TO_INDEX.get(str(row["action_label"]), -1)
        look_label = LOOK_TO_INDEX.get(str(row["look_label"]), -1)
        return {
            "sequence_id": str(row["sequence_id"]),
            "clip": clip,
            "action_clip": action_clip,
            "look_clip": look_clip,
            "action_label": torch.tensor(action_label, dtype=torch.long),
            "look_label": torch.tensor(look_label, dtype=torch.long),
        }
