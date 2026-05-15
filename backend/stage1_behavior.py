from __future__ import annotations

from pathlib import Path

from stage1_inference.infer_video import run_video_inference

from .schemas import BehaviorPrediction, BehaviorWindowPrediction, CropPreview


class Stage1BehaviorRecognizer:
    """Stage 1 raw-video behavior inference backed by stage1_inference.infer_video."""

    def predict(
        self,
        *,
        video_path: str,
        pedestrian_id: str | None = None,
        detections_csv: str | None = None,
        yolo_model_path: str | None = None,
        clip_length: int = 16,
        window_stride: int = 8,
        look_max_frames: int = 15,
        device: str = "cpu",
        output_path: str | None = None,
    ) -> BehaviorPrediction:
        output = run_video_inference(
            video_path=Path(video_path),
            pedestrian_id=pedestrian_id,
            detections_csv=Path(detections_csv) if detections_csv else None,
            yolo_model_path=Path(yolo_model_path) if yolo_model_path else None,
            output_path=Path(output_path) if output_path else None,
            clip_length=clip_length,
            window_stride=window_stride,
            look_max_frames=look_max_frames,
            device=device,
        )
        crop_previews = self._load_crop_previews(output.attrs.get("crop_previews", []))
        if output.empty:
            raise RuntimeError("Stage 1 inference produced no pedestrian windows")

        action_prob_cols = ["prob_action_standing", "prob_action_walking"]
        look_prob_cols = ["prob_look_not_looking", "prob_look_looking"]
        action_mean = output[action_prob_cols].mean()
        look_mean = output[look_prob_cols].mean()
        action_label = "walking" if action_mean["prob_action_walking"] >= action_mean["prob_action_standing"] else "standing"
        look_label = "looking" if look_mean["prob_look_looking"] >= look_mean["prob_look_not_looking"] else "not-looking"
        action_confidence = float(max(action_mean["prob_action_standing"], action_mean["prob_action_walking"]))
        look_confidence = float(max(look_mean["prob_look_not_looking"], look_mean["prob_look_looking"]))

        windows = [
            BehaviorWindowPrediction(
                window_index=int(row.window_index),
                start_frame=int(row.start_frame),
                end_frame=int(row.end_frame),
                action_label=str(row.action_pred),
                action_confidence=float(max(row.prob_action_standing, row.prob_action_walking)),
                look_label=str(row.look_pred).replace("_", "-"),
                look_confidence=float(max(row.prob_look_not_looking, row.prob_look_looking)),
            )
            for row in output.itertuples(index=False)
        ]
        primary_window = max(
            windows,
            key=lambda item: (item.action_confidence + item.look_confidence) / 2,
        )
        return BehaviorPrediction(
            pedestrian_id=str(output["pedestrian_id"].iloc[0]),
            action_label=action_label,
            action_confidence=round(action_confidence, 4),
            look_label=look_label,
            look_confidence=round(look_confidence, 4),
            temporal_window=clip_length,
            windows_analyzed=len(windows),
            track_frame_count=int(output["num_track_frames"].iloc[0]),
            stage1_output_path=str(output_path) if output_path else None,
            primary_window=primary_window,
            windows=windows,
            crop_previews=crop_previews,
        )

    @staticmethod
    def _load_crop_previews(previews: list[dict[str, object]]) -> list[CropPreview]:
        crop_previews = []
        for preview in previews:
            crop_previews.append(
                CropPreview(
                    frame_id=int(preview["frame_id"]),
                    image_data=str(preview["image_data"]),
                )
            )
        return crop_previews
