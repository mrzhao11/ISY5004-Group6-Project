from __future__ import annotations

import uuid
from pathlib import Path

from .schemas import AnalyzeRequest, AnalyzeResponse
from .stage1_behavior import Stage1BehaviorRecognizer
from .stage2_risk import Stage2RiskPredictor


class TwoStagePedestrianPipeline:
    def __init__(self) -> None:
        self.stage1 = Stage1BehaviorRecognizer()
        self.stage2 = Stage2RiskPredictor()

    def run(self, request: AnalyzeRequest) -> AnalyzeResponse:
        request_id = uuid.uuid4().hex
        stage1_output_path = Path("outputs/predictions") / f"stage1_video_inference_{request_id}.csv"
        stage2_output_path = Path("outputs/predictions") / f"stage2_video_inference_{request_id}.csv"
        stage1_result = self.stage1.predict(
            video_path=request.video_path,
            pedestrian_id=request.pedestrian_id,
            detections_csv=request.detections_csv,
            yolo_model_path=request.yolo_model_path,
            clip_length=request.clip_length,
            window_stride=request.window_stride,
            look_max_frames=request.look_max_frames,
            device=request.device,
            output_path=str(stage1_output_path),
        )
        stage2_result = self.stage2.predict(
            behavior=stage1_result,
            stage1_output_path=str(stage1_output_path),
            stage2_output_path=str(stage2_output_path),
            device=request.device,
            include_context=request.include_context,
        )

        return AnalyzeResponse(
            request_id=request_id,
            stage1_behavior=stage1_result,
            stage2_risk=stage2_result,
            notes=[
                "The system runs pedestrian behavior inference from mp4 input.",
                "Action labels are inferred from sliding crop windows; look labels are inferred from sampled pedestrian frames.",
                "Crossing risk is inferred from Stage 1 behavior probabilities and tracked trajectory features.",
            ],
        )
