from __future__ import annotations

from .schemas import BehaviorPrediction, RiskPrediction


class Stage2RiskPredictor:
    """Stage 2 integration point. Returns pending until a trained model is provided."""

    def predict(self, behavior: BehaviorPrediction, include_context: bool = True) -> RiskPrediction:
        feature_summary = {
            "action_confidence": behavior.action_confidence,
            "look_confidence": behavior.look_confidence,
            "windows_analyzed": float(behavior.windows_analyzed),
            "scene_context_flag": 1.0 if include_context else 0.0,
        }

        return RiskPrediction(
            status="pending",
            message="Stage 2 crossing-risk model is not connected yet.",
            crossing_probability=None,
            risk_level=None,
            feature_summary=feature_summary,
        )
