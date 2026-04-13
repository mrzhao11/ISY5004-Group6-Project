from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    video_path: str = Field(..., description="Path to a driving video clip")
    pedestrian_id: Optional[str] = Field(
        default="ped_001", description="Optional tracked pedestrian id"
    )
    include_context: bool = Field(
        default=True, description="Whether to include scene context in Stage 2"
    )


class BehaviorPrediction(BaseModel):
    label: str
    confidence: float
    temporal_window: int


class RiskPrediction(BaseModel):
    crossing_probability: float
    risk_level: str
    feature_summary: Dict[str, float]


class AnalyzeResponse(BaseModel):
    request_id: str
    stage1_behavior: BehaviorPrediction
    stage2_risk: RiskPrediction
    notes: List[str]
