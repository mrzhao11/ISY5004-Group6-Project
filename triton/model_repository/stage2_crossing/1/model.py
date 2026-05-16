"""
Stage 2: Pedestrian Crossing Intention Prediction

Loads Stage2CrossingModel from a checkpoint dict (model_state_dict + model_config).
Falls back to XGBoost (.ubj), flat-feature PyTorch (.pt nn.Module), ONNX, or
numpy simulation if no compatible file is found.

Inputs (from Triton):
  traj_features     [B, 16, 6]   float32  — [cx, cy, w, h, Δcx, Δcy] per frame
  behavior_probs    [B, 4]       float32  — from Stage 1
  behavior_embedding[B, 256]     float32  — from Stage 1
  context_features  [B, 6]       float32  — scene context
  vehicle_features  [B, 4]       float32  — vehicle cues

Outputs:
  crossing_prob [B, 1]  float32
  risk_level    [B, 1]  int32     (0=Low, 1=Medium, 2=High)
"""

import os
import sys
import numpy as np
import triton_python_backend_utils as pb_utils

# Make stage2_inference importable inside the Triton container.
# docker-compose mounts ./stage2_inference → /stage2_inference
_STAGE2_MODULE_PATH = "/stage2_inference"
if _STAGE2_MODULE_PATH not in sys.path:
    sys.path.insert(0, os.path.dirname(_STAGE2_MODULE_PATH))

_MODE_SIM   = "simulation"
_MODE_YZY   = "yzy_stage2"
_MODE_PT    = "pytorch_flat"
_MODE_XGB   = "xgboost"
_MODE_ONNX  = "onnxruntime"

# How many trajectory frames the model was trained on (observation_length in data_config)
_OBS_LEN = 8
# Dimensions inferred from state dict (trajectory_static_branch.net.0 input size)
_TRAJ_STATIC_DIM = 6

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
            self._load_pt(pt_path)

        else:
            self.mode = _MODE_SIM
            pb_utils.Logger.log_info("stage2_crossing: no model file — simulation mode")

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_pt(self, pt_path: str):
        import torch, torch.nn as nn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded = torch.load(pt_path, map_location=self.device, weights_only=False)

        if isinstance(loaded, nn.Module):
            self.model = loaded.eval()
            self.mode = _MODE_PT
            pb_utils.Logger.log_info("stage2_crossing: loaded flat PyTorch model")
            return

        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            try:
                self._load_yzy_checkpoint(loaded)
                return
            except Exception as exc:
                pb_utils.Logger.log_info(
                    f"stage2_crossing: checkpoint load failed ({exc}) — simulation mode")
                self.mode = _MODE_SIM
                return

        self.mode = _MODE_SIM
        pb_utils.Logger.log_info(
            "stage2_crossing: model.pt unrecognised format — simulation mode")

    def _load_yzy_checkpoint(self, ckpt: dict):
        import torch
        import dataclasses

        from stage2_inference.models import Stage2CrossingModel
        from stage2_inference.config import DataConfig, ModelConfig

        dc_fields = {f.name for f in dataclasses.fields(DataConfig)}
        mc_fields = {f.name for f in dataclasses.fields(ModelConfig)}

        raw_dc = {k: v for k, v in ckpt["data_config"].items() if k in dc_fields}
        raw_mc = {k: v for k, v in ckpt["model_config"].items() if k in mc_fields}

        data_cfg = DataConfig(**raw_dc)
        model_cfg = ModelConfig(**raw_mc)

        sd = ckpt["model_state_dict"]
        traj_static_dim = _TRAJ_STATIC_DIM
        if "trajectory_static_branch.net.0.weight" in sd:
            traj_static_dim = sd["trajectory_static_branch.net.0.weight"].shape[1]

        behavior_feature_dim = 0
        if "behavior_branch.embedding.weight" in sd:
            behavior_feature_dim = sd["behavior_branch.embedding.weight"].shape[0]
        elif "behavior_branch.proj.0.weight" in sd:
            behavior_feature_dim = sd["behavior_branch.proj.0.weight"].shape[1]

        model = Stage2CrossingModel(
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            trajectory_static_dim=traj_static_dim,
            behavior_feature_dim=behavior_feature_dim,
            context_feature_dim=0,
            vehicle_feature_dim=0,
        )
        model.load_state_dict(sd)
        model.eval()

        self.model    = model.to(self.device)
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.traj_static_dim = traj_static_dim
        self.mode = _MODE_YZY
        pb_utils.Logger.log_info(
            f"stage2_crossing: Stage2CrossingModel loaded "
            f"(arch={model_cfg.architecture}, traj_static={traj_static_dim}, "
            f"behavior_mode={data_cfg.behavior_mode})")

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

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
            elif self.mode == _MODE_YZY:
                crossing_prob = self._infer_yzy(traj, B)
            elif self.mode == _MODE_PT:
                crossing_prob = self._infer_pt_flat(traj, b_prob, b_emb, ctx, veh, B)
            else:
                crossing_prob = self._infer_sim(traj, b_prob, B)

            risk_level = np.where(
                crossing_prob >= 0.70, 2,
                np.where(crossing_prob >= 0.40, 1, 0)
            ).astype(np.int32)

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("crossing_prob", np.ascontiguousarray(crossing_prob)),
                pb_utils.Tensor("risk_level",    np.ascontiguousarray(risk_level)),
            ]))
        return responses

    # ------------------------------------------------------------------
    # Inference backends
    # ------------------------------------------------------------------

    def _traj_to_model_inputs(self, traj: np.ndarray, B: int):
        """
        Convert Triton traj_features [B, 16, 6] → trajectory_seq [B, 8, 8] + trajectory_static [B, 6].
        """
        obs = traj[:, :_OBS_LEN, :]               # [B, 8, 6]
        cx = obs[:, :, 0];  cy = obs[:, :, 1]
        w  = obs[:, :, 2];  h  = obs[:, :, 3]
        dx = obs[:, :, 4];  dy = obs[:, :, 5]

        scale = np.maximum(w[:, 0], h[:, 0])[:, np.newaxis].clip(min=1e-6)  # [B, 1]

        rel_cx   = (cx - cx[:, :1]) / scale
        rel_cy   = (cy - cy[:, :1]) / scale
        norm_w   = w / scale
        norm_h   = h / scale
        norm_dx  = dx / scale
        norm_dy  = dy / scale
        norm_dw  = np.zeros_like(norm_dx)
        norm_dh  = np.zeros_like(norm_dy)

        traj_seq = np.ascontiguousarray(np.stack(
            [rel_cx, rel_cy, norm_w, norm_h, norm_dx, norm_dy, norm_dw, norm_dh],
            axis=-1,
        ), dtype=np.float32)  # [B, 8, 8]

        speed    = np.sqrt(traj[:, :, 4]**2 + traj[:, :, 5]**2)  # [B, 16]
        speed_mean = speed.mean(axis=1)
        speed_var  = speed.var(axis=1)
        disp       = np.sqrt((traj[:, -1, 0] - traj[:, 0, 0])**2 +
                             (traj[:, -1, 1] - traj[:, 0, 1])**2)
        bbox_scale_change = ((traj[:, -1, 2] * traj[:, -1, 3]) /
                             np.maximum(traj[:, 0, 2] * traj[:, 0, 3], 1e-6)) - 1.0
        traj_length  = speed.sum(axis=1)
        center_jitter = speed.std(axis=1)

        static_raw = np.stack(
            [speed_mean, speed_var, disp, bbox_scale_change, traj_length, center_jitter],
            axis=-1,
        ).astype(np.float32)  # [B, 6]
        traj_static = np.ascontiguousarray(
            np.sign(static_raw) * np.log1p(np.abs(static_raw)))

        return traj_seq, traj_static

    def _infer_yzy(self, traj: np.ndarray, B: int) -> np.ndarray:
        import torch
        traj_seq, traj_static = self._traj_to_model_inputs(traj, B)
        batch = {
            "trajectory_seq":    torch.from_numpy(traj_seq).to(self.device),
            "trajectory_static": torch.from_numpy(traj_static).to(self.device),
        }
        with torch.no_grad():
            logit = self.model(batch)
            prob  = torch.sigmoid(logit).cpu().numpy()
        return np.ascontiguousarray(prob.reshape(B, 1), dtype=np.float32)

    def _build_flat_features(self, traj, b_prob, b_emb, ctx, veh, B):
        return np.ascontiguousarray(np.concatenate([
            traj.reshape(B, -1),  # 16*6 = 96
            b_prob,               # 4
            b_emb,                # 256
            ctx,                  # 6
            veh,                  # 4
        ], axis=1), dtype=np.float32)  # total 366

    def _infer_onnx(self, traj, b_prob, b_emb, ctx, veh, B):
        features = self._build_flat_features(traj, b_prob, b_emb, ctx, veh, B)
        out = self.session.run(None, {"features": features})
        return np.ascontiguousarray(out[0].reshape(B, 1), dtype=np.float32)

    def _infer_xgb(self, traj, b_prob, b_emb, ctx, veh, B):
        import xgboost as xgb
        features = self._build_flat_features(traj, b_prob, b_emb, ctx, veh, B)
        prob = self.model.predict(xgb.DMatrix(features))
        return np.ascontiguousarray(prob.reshape(B, 1), dtype=np.float32)

    def _infer_pt_flat(self, traj, b_prob, b_emb, ctx, veh, B):
        import torch
        features = self._build_flat_features(traj, b_prob, b_emb, ctx, veh, B)
        x = torch.from_numpy(features).to(self.device)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(x)).cpu().numpy()
        return np.ascontiguousarray(prob.reshape(B, 1), dtype=np.float32)

    def _infer_sim(self, traj, b_prob, B):
        beh_score    = (b_prob * _BEHAVIOR_WEIGHTS).sum(axis=1)
        delta        = traj[:, :, 4:6]
        motion_score = np.clip(np.linalg.norm(delta, axis=-1).mean(axis=1) / 10.0, 0.0, 1.0)
        prob = np.clip(
            0.50 * beh_score + 0.50 * motion_score,
            0.01, 0.99,
        ).astype(np.float32).reshape(B, 1)
        return np.ascontiguousarray(prob)

    # ------------------------------------------------------------------

    def finalize(self):
        pass
