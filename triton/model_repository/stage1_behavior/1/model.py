"""
Stage 1: Pedestrian Behavior Understanding

Loads two models from the checkpoint files in the same directory:
  action_sequence_swin3d_t.pt  — TorchvisionVideoSwinBehaviorModel (video sequence → action + embedding)
  look_frame_swin_t.pt         — SingleFrameClassifier (single frame → look probability)
  (neither found)              — numpy simulation (smoke-test / CI mode)

Behavior_probs[B,4] is a joint distribution (sums to 1) over the two independent predictions:
  [0] walking  = P(walking) * P(not_looking)
  [1] standing = P(standing) * P(not_looking)
  [2] looking  = P(walking)  * P(looking)
  [3] waiting  = P(standing) * P(looking)

Behavior_embedding[B,256] comes from the action model's embedding head.

Input : ped_clip [B, 16, 224, 224, 3]  float32  (BTHWC)
Output: behavior_probs [B, 4]           float32
        behavior_embedding [B, 256]     float32
"""

import os
import sys
import pathlib
import numpy as np
import triton_python_backend_utils as pb_utils

# Make stage1_inference importable — docker-compose mounts ./stage1_inference → /stage1_inference
_STAGE1_MODULE_PATH = "/stage1_inference"
if _STAGE1_MODULE_PATH not in sys.path:
    sys.path.insert(0, os.path.dirname(_STAGE1_MODULE_PATH))

_MODE_SIM  = "simulation"
_MODE_REAL = "real"

# Clip spatial size the action model was trained on
_ACTION_CLIP_SIZE = 112


class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.join(args["model_repository"], args["model_version"])
        action_path = os.path.join(model_dir, "action_sequence_swin3d_t.pt")
        look_path   = os.path.join(model_dir, "look_frame_swin_t.pt")

        if os.path.exists(action_path) and os.path.exists(look_path):
            try:
                self._load_models(action_path, look_path)
            except Exception as exc:
                self.mode = _MODE_SIM
                pb_utils.Logger.log_info(
                    f"stage1_behavior: model load failed ({exc}) — simulation mode")
        else:
            self.mode = _MODE_SIM
            pb_utils.Logger.log_info(
                "stage1_behavior: checkpoint files not found — simulation mode")

    # ------------------------------------------------------------------

    def _load_checkpoint(self, path: str):
        import torch
        # Checkpoints were saved on Windows; map WindowsPath → PosixPath on Linux.
        if os.name != "nt":
            pathlib.WindowsPath = pathlib.PosixPath
        return torch.load(path, map_location=self.device, weights_only=False)

    def _make_model_config(self, raw: dict):
        from stage1_inference.config import ModelConfig
        cfg = ModelConfig()
        for k, v in raw.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.pretrained = False  # weights come from checkpoint; skip torchvision download
        return cfg

    def _load_models(self, action_path: str, look_path: str):
        import torch
        from stage1_inference.models import build_model, build_single_frame_model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Action model: TorchvisionVideoSwinBehaviorModel (clip → action_logits, embedding)
        action_ckpt = self._load_checkpoint(action_path)
        action_cfg  = self._make_model_config(action_ckpt["model_config"])
        action_model = build_model(action_cfg)
        action_model.load_state_dict(action_ckpt["model_state_dict"])
        action_model.eval()
        self.action_model = action_model.to(self.device)

        # Look model: SingleFrameClassifier with swin_t backbone (frame → look_logits)
        look_ckpt = self._load_checkpoint(look_path)
        look_cfg  = self._make_model_config(look_ckpt["model_config"])
        look_model = build_single_frame_model(look_cfg, "look")
        look_model.load_state_dict(look_ckpt["model_state_dict"])
        look_model.eval()
        self.look_model = look_model.to(self.device)

        self.mode = _MODE_REAL
        pb_utils.Logger.log_info(
            f"stage1_behavior: loaded action={action_cfg.architecture} "
            f"look={look_cfg.architecture} on device={self.device}")

    # ------------------------------------------------------------------

    def execute(self, requests):
        responses = []
        for request in requests:
            clip = pb_utils.get_input_tensor_by_name(request, "ped_clip").as_numpy()
            # clip: [B, 16, 224, 224, 3]  BTHWC float32
            B = clip.shape[0]

            if self.mode == _MODE_REAL:
                behavior_probs, behavior_embedding = self._infer_real(clip, B)
            else:
                behavior_probs, behavior_embedding = self._infer_sim(clip, B)

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("behavior_probs",     behavior_probs),
                pb_utils.Tensor("behavior_embedding", behavior_embedding),
            ]))
        return responses

    # ------------------------------------------------------------------

    def _infer_real(self, clip: np.ndarray, B: int):
        import torch
        import torch.nn.functional as F

        # BTHWC → BTCHW, then resize 224→112 for action model
        clip_t = torch.from_numpy(clip.transpose(0, 1, 4, 2, 3)).float()  # [B,16,3,224,224]
        BT = B * 16
        clip_112 = F.interpolate(
            clip_t.reshape(BT, 3, 224, 224),
            size=(_ACTION_CLIP_SIZE, _ACTION_CLIP_SIZE),
            mode="bilinear", align_corners=False,
        ).reshape(B, 16, 3, _ACTION_CLIP_SIZE, _ACTION_CLIP_SIZE).to(self.device)

        # Middle frame for look model: [B,3,224,224]
        mid_frame = torch.from_numpy(
            clip[:, 8, :, :, :].transpose(0, 3, 1, 2)
        ).float().to(self.device)

        with torch.no_grad():
            # ACTION_NAMES = ("standing", "walking") → index 0=standing, 1=walking
            action_logits, _, embedding = self.action_model(clip_112)
            action_probs = torch.softmax(action_logits, dim=1).cpu().numpy()  # [B,2]

            # LOOK_NAMES = ("not_looking", "looking") → index 0=not_looking, 1=looking
            look_logits, _ = self.look_model(mid_frame)
            look_probs = torch.softmax(look_logits, dim=1).cpu().numpy()  # [B,2]

            embedding_np = embedding.cpu().numpy().astype(np.float32)  # [B,256]

        p_standing    = action_probs[:, 0]
        p_walking     = action_probs[:, 1]
        p_not_looking = look_probs[:, 0]
        p_looking     = look_probs[:, 1]

        # Joint distribution summing to 1
        behavior_probs = np.ascontiguousarray(np.stack([
            p_walking  * p_not_looking,   # walking
            p_standing * p_not_looking,   # standing
            p_walking  * p_looking,        # looking
            p_standing * p_looking,        # waiting
        ], axis=1), dtype=np.float32)     # [B, 4]

        return behavior_probs, np.ascontiguousarray(embedding_np)

    def _infer_sim(self, clip: np.ndarray, B: int):
        seed = int(abs(float(clip.mean())) * 1e6) % (2 ** 31)
        rng = np.random.default_rng(seed)
        logits = rng.random((B, 4)).astype(np.float32)
        exp_l  = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs  = np.ascontiguousarray(exp_l / exp_l.sum(axis=1, keepdims=True), dtype=np.float32)
        emb    = np.ascontiguousarray(
            rng.standard_normal((B, 256)).astype(np.float32) * 0.1)
        return probs, emb

    # ------------------------------------------------------------------

    def finalize(self):
        pass
