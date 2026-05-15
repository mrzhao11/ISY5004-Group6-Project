"""
Stage 1: Pedestrian Behavior Understanding

Auto-detects available model file in the same directory and loads accordingly:
  model.onnx  → onnxruntime (fastest, recommended for benchmarking)
  model.pt    → PyTorch     (easiest to produce from training)
  (neither)   → numpy simulation (smoke-test / CI mode)

Input : ped_clip [B, 16, 224, 224, 3]  float32
Output: behavior_probs [B, 4]           float32  (walking, standing, looking, waiting)
        behavior_embedding [B, 256]     float32
"""

import os
import numpy as np
import triton_python_backend_utils as pb_utils

_MODE_SIM   = "simulation"
_MODE_PT    = "pytorch"
_MODE_ONNX  = "onnxruntime"


class TritonPythonModel:
    def initialize(self, args):
        model_dir = os.path.join(args["model_repository"], args["model_version"])
        onnx_path = os.path.join(model_dir, "model.onnx")
        pt_path   = os.path.join(model_dir, "model.pt")

        if os.path.exists(onnx_path):
            import onnxruntime as ort
            self.session = ort.InferenceSession(onnx_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.mode = _MODE_ONNX
            pb_utils.Logger.log_info("stage1_behavior: loaded ONNX model")

        elif os.path.exists(pt_path):
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # B should assign self.model to their actual model class before this line
            # e.g. self.model = VideoSwinTransformer(); self.model.load_state_dict(...)
            self.model = torch.load(pt_path, map_location=self.device)
            self.model.eval()
            self.mode = _MODE_PT
            pb_utils.Logger.log_info("stage1_behavior: loaded PyTorch model")

        else:
            self.mode = _MODE_SIM
            pb_utils.Logger.log_info(
                "stage1_behavior: no model file found — running in simulation mode")

    def execute(self, requests):
        responses = []
        for request in requests:
            clip = pb_utils.get_input_tensor_by_name(request, "ped_clip").as_numpy()
            # clip: [B, 16, 224, 224, 3]
            B = clip.shape[0]

            if self.mode == _MODE_ONNX:
                behavior_probs, behavior_embedding = self._infer_onnx(clip, B)
            elif self.mode == _MODE_PT:
                behavior_probs, behavior_embedding = self._infer_pt(clip, B)
            else:
                behavior_probs, behavior_embedding = self._infer_sim(clip, B)

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("behavior_probs",     behavior_probs),
                pb_utils.Tensor("behavior_embedding", behavior_embedding),
            ]))
        return responses

    # ------------------------------------------------------------------

    def _infer_onnx(self, clip, B):
        # ONNX model expects [B, 16, 3, 224, 224] (NCTHW)
        x = clip.transpose(0, 1, 4, 2, 3).astype(np.float32)
        out = self.session.run(None, {"ped_clip": x})
        # out[0]: behavior_probs [B,4], out[1]: behavior_embedding [B,256]
        return out[0].astype(np.float32), out[1].astype(np.float32)

    def _infer_pt(self, clip, B):
        import torch
        # Convert [B, 16, 224, 224, 3] → [B, 16, 3, 224, 224]
        x = torch.from_numpy(clip.transpose(0, 1, 4, 2, 3)).to(self.device)
        with torch.no_grad():
            logits, embedding = self.model(x)
            exp_l = torch.exp(logits - logits.max(dim=1, keepdim=True).values)
            probs = (exp_l / exp_l.sum(dim=1, keepdim=True)).cpu().numpy().astype(np.float32)
        return probs, embedding.cpu().numpy().astype(np.float32)

    def _infer_sim(self, clip, B):
        seed = int(abs(float(clip.mean())) * 1e6) % (2 ** 31)
        rng = np.random.default_rng(seed)
        logits = rng.random((B, 4)).astype(np.float32)
        exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = (exp_l / exp_l.sum(axis=1, keepdims=True)).astype(np.float32)
        emb   = rng.standard_normal((B, 256)).astype(np.float32) * 0.1
        return probs, emb

    # ------------------------------------------------------------------

    def finalize(self):
        pass
