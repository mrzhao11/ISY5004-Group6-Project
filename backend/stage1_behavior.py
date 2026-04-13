from __future__ import annotations

import hashlib

from .schemas import BehaviorPrediction


class Stage1BehaviorRecognizer:
    """Current behavior-recognition module (ready for YOLO + CNN/LSTM upgrade)."""

    _BEHAVIOR_LABELS = ["walking", "standing", "looking", "waiting"]

    def predict(self, video_path: str, pedestrian_id: str, window: int = 16) -> BehaviorPrediction:
        seed_text = f"{video_path}:{pedestrian_id}:{window}"
        digest = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
        bucket = int(digest[:2], 16)

        label = self._BEHAVIOR_LABELS[bucket % len(self._BEHAVIOR_LABELS)]
        confidence = round(0.6 + ((bucket % 35) / 100), 3)

        return BehaviorPrediction(label=label, confidence=confidence, temporal_window=window)
