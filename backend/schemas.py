from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    video_path: str = Field(..., description="Path to a driving video clip")
    pedestrian_id: Optional[str] = Field(default=None, description="Optional tracked pedestrian id")
    detections_csv: Optional[str] = Field(
        default=None,
        description="Optional CSV with frame_id,pedestrian_id,x1,y1,x2,y2,score columns",
    )
    yolo_model_path: Optional[str] = Field(
        default="models/stage1/person_detector.pt",
        description="Path to a YOLO person detector weight file used when detections_csv is not provided",
    )
    clip_length: int = Field(default=16, description="Number of frames per Stage 1 action window")
    window_stride: int = Field(default=8, description="Sliding-window stride for Stage 1 video inference")
    look_max_frames: int = Field(default=15, description="Maximum number of frames sampled for look inference")
    device: str = Field(default="cpu", description="Inference device, for example cpu or cuda")
    include_context: bool = Field(
        default=True, description="Whether to include scene context in Stage 2"
    )


class DemoVideo(BaseModel):
    filename: str
    label: str


class DemoVideoRequest(BaseModel):
    filename: str


class BehaviorWindowPrediction(BaseModel):
    window_index: int
    start_frame: int
    end_frame: int
    action_label: str
    action_confidence: float
    look_label: str
    look_confidence: float


class CropPreview(BaseModel):
    frame_id: int
    image_data: str


class BehaviorPrediction(BaseModel):
    pedestrian_id: str
    action_label: str
    action_confidence: float
    look_label: str
    look_confidence: float
    temporal_window: int
    windows_analyzed: int
    track_frame_count: int
    stage1_output_path: Optional[str] = None
    primary_window: Optional[BehaviorWindowPrediction] = None
    windows: List[BehaviorWindowPrediction] = Field(default_factory=list)
    crop_previews: List[CropPreview] = Field(default_factory=list)


class RiskPrediction(BaseModel):
    status: str
    message: str
    crossing_probability: Optional[float] = None
    risk_level: Optional[str] = None
    feature_summary: Dict[str, float]


class AnalyzeResponse(BaseModel):
    request_id: str
    stage1_behavior: BehaviorPrediction
    stage2_risk: RiskPrediction
    notes: List[str]
