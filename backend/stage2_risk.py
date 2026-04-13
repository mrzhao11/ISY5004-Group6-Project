from __future__ import annotations

from .schemas import BehaviorPrediction, RiskPrediction


class Stage2RiskPredictor:
    """Current risk-prediction module (ready for trained XGBoost integration)."""

    _BEHAVIOR_TO_RISK = {
        "walking": 0.68,
        "standing": 0.24,
        "looking": 0.54,
        "waiting": 0.38,
    }

    def predict(self, behavior: BehaviorPrediction, include_context: bool = True) -> RiskPrediction:
        base_prob = self._BEHAVIOR_TO_RISK.get(behavior.label, 0.45)
        confidence_gain = (behavior.confidence - 0.6) * 0.4
        context_gain = 0.08 if include_context else -0.04

        crossing_probability = max(0.01, min(0.99, round(base_prob + confidence_gain + context_gain, 3)))

        if crossing_probability >= 0.7:
            risk_level = "High"
        elif crossing_probability >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        feature_summary = {
            "trajectory_speed_norm": 0.52,
            "behavior_confidence": behavior.confidence,
            "scene_context_flag": 1.0 if include_context else 0.0,
        }

        return RiskPrediction(
            crossing_probability=crossing_probability,
            risk_level=risk_level,
            feature_summary=feature_summary,
        )
