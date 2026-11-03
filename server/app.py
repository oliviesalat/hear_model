"""HTTP API for HeAR embeddings on PyTorch.

Mirrors the request/response contract of hear/python/serving (Vertex AI style):
POST /predict {"instances": [...]} -> {"predictions": [...]}, GET /health -> "ok".

Env: HEAR_MODEL (HF id or local dir, default google/hear-pytorch),
HEAR_DEVICE (auto|cpu|mps|cuda, default auto).
"""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from server.audio import InstanceError, instance_to_clip, wav_to_notebook_clip
from server.embedder import Embedder, pick_device
from server.schemas import (
    DemoResult,
    EmbeddingResult,
    ErrorDetail,
    ErrorResult,
    PredictRequest,
    PredictResponse,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Tests inject a fake embedder before startup.
    if not hasattr(app.state, "embedder"):
        device = pick_device(os.environ.get("HEAR_DEVICE", "auto"))
        model_path = os.environ.get("HEAR_MODEL", "google/hear-pytorch")
        logger.info("Loading %s on %s", model_path, device)
        app.state.embedder = Embedder(model_path, device)
    yield


app = FastAPI(title="HeAR embeddings", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        {"error": "Body must be JSON with an `instances` list."}, status_code=400
    )


@app.exception_handler(Exception)
async def internal_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse({"error": "Internal server error."}, status_code=500)


@app.get("/", include_in_schema=False)
def demo() -> FileResponse:
    """Serves the drag-and-drop demo page; same origin as /predict, so no CORS."""
    return FileResponse(Path(__file__).with_name("demo.html"))


@app.post("/demo/predict", response_model=DemoResult, include_in_schema=False)
def demo_predict(
    request: Request, wav: bytes = Body(media_type="audio/wav")
) -> DemoResult:
    """Embeds a raw WAV of any length the way the PyTorch quick start notebook does.

    Kept apart from /predict, whose contract (exactly 2 s per clip) mirrors the upstream server.
    """
    try:
        notebook_clip = wav_to_notebook_clip(wav)
    except InstanceError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    embedding = request.app.state.embedder.embed(notebook_clip.clip[np.newaxis])[0]
    return DemoResult(
        embedding=embedding.tolist(),
        sample_rate=notebook_clip.sample_rate,
        channels=notebook_clip.channels,
        seconds=notebook_clip.seconds,
    )


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    # The model loads during startup, so a listening server is a ready one.
    return "ok"


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Returns one HeAR embedding per instance."""
    predictions: list[EmbeddingResult | ErrorResult | None] = [None] * len(
        body.instances
    )
    clips: list[np.ndarray] = []
    clip_positions: list[int] = []
    for position, instance in enumerate(body.instances):
        try:
            clips.append(instance_to_clip(instance))
            clip_positions.append(position)
        except InstanceError as error:
            predictions[position] = ErrorResult(
                error=ErrorDetail(
                    description=f"Failed to get prediction for instance. Reason: {error}"
                )
            )

    if clips:
        embeddings = request.app.state.embedder.embed(np.stack(clips))
        for position, embedding in zip(clip_positions, embeddings, strict=True):
            predictions[position] = EmbeddingResult(embedding=embedding.tolist())
    return PredictResponse(predictions=predictions)
