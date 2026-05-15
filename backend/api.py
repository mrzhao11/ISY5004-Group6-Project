from __future__ import annotations

import shutil
import tempfile
import re
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .pipeline import TwoStagePedestrianPipeline
from .schemas import AnalyzeRequest, AnalyzeResponse, DemoVideo, DemoVideoRequest

app = FastAPI(
    title="Pedestrian Behavior and Crossing Analysis API",
    description="API for raw-video pedestrian behavior inference and crossing-risk analysis",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = TwoStagePedestrianPipeline()
DEMO_VIDEO_DIR = Path("data/demo")
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}


def _build_default_request(video_path: Path) -> AnalyzeRequest:
    return AnalyzeRequest(
        video_path=str(video_path),
        pedestrian_id=None,
        detections_csv=None,
        yolo_model_path="models/stage1/person_detector.pt",
        clip_length=16,
        window_stride=8,
        look_max_frames=15,
        device="cpu",
        include_context=True,
    )


def _demo_video_label(path: Path) -> str:
    label = path.stem.replace("_", " ")
    label = re.sub(r"([A-Za-z])(\d)", r"\1 \2", label)
    return label.title()


def _resolve_demo_video(filename: str) -> Path:
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid demo video name.")
    video_path = DEMO_VIDEO_DIR / safe_name
    if video_path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES or not video_path.is_file():
        raise HTTPException(status_code=404, detail="Demo video was not found.")
    return video_path


@app.get("/api/v1/health")
def health() -> dict:
    return {"status": "ok", "service": "pedestrian-behavior-risk-demo"}


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    return pipeline.run(request)


@app.get("/api/v1/demo-videos", response_model=list[DemoVideo])
def list_demo_videos() -> list[DemoVideo]:
    if not DEMO_VIDEO_DIR.exists():
        return []
    videos = []
    for path in sorted(DEMO_VIDEO_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_SUFFIXES:
            videos.append(DemoVideo(filename=path.name, label=_demo_video_label(path)))
    return videos


@app.get("/api/v1/demo-videos/{filename}")
def get_demo_video(filename: str) -> FileResponse:
    return FileResponse(_resolve_demo_video(filename), media_type="video/mp4")


@app.post("/api/v1/analyze-demo", response_model=AnalyzeResponse)
def analyze_demo(request: DemoVideoRequest) -> AnalyzeResponse:
    return pipeline.run(_build_default_request(_resolve_demo_video(request.filename)))


@app.post("/api/v1/analyze-video", response_model=AnalyzeResponse)
def analyze_video(video: UploadFile = File(...)) -> AnalyzeResponse:
    suffix = Path(video.filename or "uploaded_video.mp4").suffix.lower()
    if suffix not in SUPPORTED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail="Please upload a supported video file.")

    with tempfile.TemporaryDirectory(prefix="uploaded_video_") as temp_dir:
        video_path = Path(temp_dir) / f"input{suffix}"
        with video_path.open("wb") as handle:
            shutil.copyfileobj(video.file, handle)

        return pipeline.run(_build_default_request(video_path))
