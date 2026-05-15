from __future__ import annotations

from pathlib import Path

from stage2_inference.stage2 import run_stage2_from_stage1_csv

from .schemas import BehaviorPrediction, RiskPrediction


class Stage2RiskPredictor:
    """Stage 2 crossing-risk inference backed by stage2_inference.stage2."""

    def predict(
        self,
        *,
        behavior: BehaviorPrediction,
        stage1_output_path: str,
        stage2_output_path: str,
        device: str = "cpu",
        include_context: bool = True,
    ) -> RiskPrediction:
        output = run_stage2_from_stage1_csv(
            stage1_path=Path(stage1_output_path),
            output_path=Path(stage2_output_path),
            device=device,
        )
        if output.empty:
            raise RuntimeError("Stage 2 inference produced no crossing-risk windows")

        probability = float(output["prob_crossing"].mean())
        risk_level = "High" if probability >= 0.5 else "Low"
        feature_summary = {
            "action_confidence": behavior.action_confidence,
            "look_confidence": behavior.look_confidence,
            "windows_analyzed": float(behavior.windows_analyzed),
            "stage2_windows_analyzed": float(len(output)),
            "mean_base_prob_crossing": float(output["base_prob_crossing"].mean()),
            "mean_stage1_aux_prob_crossing": float(output["stage1_aux_prob_crossing"].mean()),
            "scene_context_flag": 1.0 if include_context else 0.0,
        }

        primary = output.iloc[int((output["prob_crossing"] - probability).abs().argmin())]
        return RiskPrediction(
            status="completed",
            message="Crossing-risk prediction completed from Stage 1 behavior and tracked trajectory features.",
            crossing_probability=round(probability, 4),
            risk_level=risk_level,
            feature_summary=feature_summary,
            stage2_output_path=stage2_output_path,
            primary_window_index=int(primary["window_index"]),
        )
