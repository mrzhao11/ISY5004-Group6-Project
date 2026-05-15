"""
Stage 2: Pedestrian Crossing Intention Prediction

Auto-detects available model file in the same directory and loads accordingly:
  model.onnx  → onnxruntime (fastest, recommended for benchmarking)
  model.ubj   → XGBoost     (native XGBoost binary format)
  model.pt    → PyTorch     (TCL or other neural model)
  (none)      → numpy simulation (smoke-test / CI mode)

Inputs:
  traj_features     [B, 16, 6]   float32  — [cx, cy, w, h, Δcx, Δcy] per frame
  behavior_probs    [B, 4]       float32  — from Stage 1
  behavior_embedding[B, 256]     float32  — from Stage 1
  context_features  [B, 6]       float32  — [crosswalk, traffic_light, stop_sign, weather, time_of_day, crossing_loc]
  vehicle_features  [B, 4]       float32  — [moving_slow, moving_fast, slowing_down, speeding_up]

Outputs:
  crossing_prob [B, 1]  float32
  risk_level    [B, 1]  int32     (0=Low, 1=Medium, 2=High)
"""

import os
import numpy as np
import triton_python_backend_utils as pb_utils

_MODE_SIM   = "simulation"
_MODE_PT    = "pytorch"
_MODE_XGB   = "xgboost"
_MODE_ONNX  = "onnxruntime"

_BEHAVIOR_WEIGHTS = np.array([0.70, 0.20, 0.55, 0.30], dtype=np.float32)


class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.join(args["model_repository"], args["model_version"])
        onnx_path = os.path.join(model_dir, "model.onnx")
        xgb_path  = os.path.join(model_dir, "model.ubj")
        pt_path   = os.path.join(model_dir, "model.pt")

        if os.path.exists(onnx_path):
            import onnxruntime as ort
            self.session = ort.InferenceSession(onnx_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.mode = _MODE_ONNX
            pb_utils.Logger.log_info("stage2_crossing: loaded ONNX model")

        elif os.path.exists(xgb_path):
            import xgboost as xgb
            self.model = xgb.Booster()
            self.model.load_model(xgb_path)
            self.mode = _MODE_XGB
            pb_utils.Logger.log_info("stage2_crossing: loaded XGBoost model")

        elif os.path.exists(pt_path):
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = torch.load(pt_path, map_location=self.device)
            self.model.eval()
            self.mode = _MODE_PT
            pb_utils.Logger.log_info("stage2_crossing: loaded PyTorch model")

        else:
            self.mode = _MODE_SIM
            pb_utils.Logger.log_info(
                "stage2_crossing: no model file found — running in simulation mode")

    def execute(self, requests):
        responses = []
        for request in requests:
            traj   = pb_utils.get_input_tensor_by_name(request, "traj_features").as_numpy()
            b_prob = pb_utils.get_input_tensor_by_name(request, "behavior_probs").as_numpy()
            b_emb  = pb_utils.get_input_tensor_by_name(request, "behavior_embedding").as_numpy()
            ctx    = pb_utils.get_input_tensor_by_name(request, "context_features").as_numpy()
            veh    = pb_utils.get_input_tensor_by_name(request, "vehicle_features").as_numpy()
            B = traj.shape[0]

            if self.mode == _MODE_ONNX:
                crossing_prob = self._infer_onnx(traj, b_prob, b_emb, ctx, veh, B)
            elif self.mode == _MODE_XGB:
                crossing_prob = self._infer_xgb(traj, b_prob, b_emb, ctx, veh, B)
            elif self.mode == _MODE_PT:
                crossing_prob = self._infer_pt(traj, b_prob, b_emb, ctx, veh, B)
            else:
                crossing_prob = self._infer_sim(traj, b_prob, b_emb, ctx, veh, B)

            risk_level = np.where(
                crossing_prob >= 0.70, 2,
                np.where(crossing_prob >= 0.40, 1, 0)
            ).astype(np.int32)

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("crossing_prob", crossing_prob),
                pb_utils.Tensor("risk_level",    risk_level),
            ]))
        return responses

    # ------------------------------------------------------------------

    def _build_feature_vector(self, traj, b_prob, b_emb, ctx, veh, B):
        # Flatten all inputs into a single feature vector: 96+4+256+6+4 = 366 dims
        return np.concatenate([
            traj.reshape(B, -1),  # 16*6 = 96
            b_prob,               # 4
            b_emb,                # 256
            ctx,                  # 6
            veh,                  # 4
        ], axis=1).astype(np.float32)

    def _infer_onnx(self, traj, b_prob, b_emb, ctx, veh, B):
        features = self._build_feature_vector(traj, b_prob, b_emb, ctx, veh, B)
        out = self.session.run(None, {"features": features})
        return out[0].reshape(B, 1).astype(np.float32)

    def _infer_xgb(self, traj, b_prob, b_emb, ctx, veh, B):
        import xgboost as xgb
        features = self._build_feature_vector(traj, b_prob, b_emb, ctx, veh, B)
        prob = self.model.predict(xgb.DMatrix(features))
        return prob.reshape(B, 1).astype(np.float32)

    def _infer_pt(self, traj, b_prob, b_emb, ctx, veh, B):
        import torch
        features = self._build_feature_vector(traj, b_prob, b_emb, ctx, veh, B)
        x = torch.from_numpy(features).to(self.device)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).cpu().numpy()
        return prob.reshape(B, 1).astype(np.float32)

    def _infer_sim(self, traj, b_prob, b_emb, ctx, veh, B):
        beh_score    = (b_prob * _BEHAVIOR_WEIGHTS).sum(axis=1)
        delta        = traj[:, :, 4:6]
        motion_score = np.clip(np.linalg.norm(delta, axis=-1).mean(axis=1) / 10.0, 0.0, 1.0)
        ctx_score    = np.clip(ctx[:, 0] * 0.4 + ctx[:, 1] * 0.3, 0.0, 1.0)
        veh_score    = np.clip(veh[:, 2] * 0.3, 0.0, 1.0)
        prob = np.clip(
            0.35 * beh_score + 0.30 * motion_score + 0.20 * ctx_score + 0.15 * veh_score,
            0.01, 0.99,
        ).astype(np.float32).reshape(B, 1)
        return prob

    # ------------------------------------------------------------------

    def finalize(self):
        pass
