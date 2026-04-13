from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import TwoStagePedestrianPipeline
from .schemas import AnalyzeRequest, AnalyzeResponse

app = FastAPI(
    title="Pedestrian Behavior and Risk API",
    description="API for a two-stage pedestrian behavior and crossing-risk system",
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


@app.get("/api/v1/health")
def health() -> dict:
    return {"status": "ok", "service": "pedestrian-risk-demo"}


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    return pipeline.run(request)
