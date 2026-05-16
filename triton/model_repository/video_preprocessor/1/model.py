"""
video_preprocessor — Triton Python backend

Reads an MP4 from a host-mounted path, uniformly samples 32 frames at original
resolution. No resizing — YOLO handles its own input scaling in detect_track.

The client must pass a path that exists inside the Triton container. The
docker-compose mount is `./data:/data:ro`, so client paths under `./data`
resolve to `/data/...` here. The model also accepts absolute paths to handle
files outside that mount during tests.

Input : video_path  [1]              STRING  (path inside the container)
Output: frames      [32, H, W, 3]    FP32    (normalised to [0, 1])
"""

import os

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils

_NUM_FRAMES = 32
_CONTAINER_DATA_ROOT = "/data"


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "video_path")
            arr = raw.as_numpy()                       # shape [1], dtype object
            path = arr.flatten()[0]
            if isinstance(path, bytes):
                path = path.decode("utf-8")
            path = _resolve(path)

            try:
                frames = _decode(path)
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("frames", frames)]
                ))
            except Exception as e:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], error=pb_utils.TritonError(str(e))
                ))
        return responses

    def finalize(self):
        pass


def _resolve(path: str) -> str:
    """Map a client-side relative path to the container-mounted /data path.

    docker-compose mounts ./data → /data:ro inside the container.
    Clients may pass either a relative path ("data/foo.mp4") or an absolute
    container path ("/data/foo.mp4"); both are accepted unchanged or normalised.
    """
    if path.startswith("data/"):
        return _CONTAINER_DATA_ROOT + "/" + path[len("data/"):]
    return path


def _decode(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found in container: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cv2 cannot open: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        raise ValueError(f"Empty or unreadable video: {path}")

    indices = np.linspace(0, total - 1, _NUM_FRAMES, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            h, w = frames[0].shape[:2] if frames else (1080, 1920)
            frames.append(np.zeros((h, w, 3), dtype=np.float32))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype(np.float32) / 255.0)
    cap.release()

    return np.stack(frames, axis=0)   # [32, H, W, 3]
