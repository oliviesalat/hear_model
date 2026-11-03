"""Request/response models of the HeAR embeddings API."""

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    instances: list[dict[str, Any]] = Field(
        description=(
            "Audio clips. Each has exactly one of `input_array` (32000 floats: 2 s mono "
            "16 kHz in [-1, 1]) or `input_bytes` (base64 WAV, resampled to 16 kHz, mixed to mono)."
        )
    )


class EmbeddingResult(BaseModel):
    embedding: list[float] = Field(description="HeAR embedding, 512 floats.")


class DemoResult(BaseModel):
    embedding: list[float] = Field(description="HeAR embedding, 512 floats.")
    sample_rate: int = Field(description="Sample rate of the uploaded WAV.")
    channels: int = Field(description="Channel count of the uploaded WAV.")
    seconds: float = Field(
        description="Duration of the uploaded WAV; only the first 2 s are embedded."
    )


class ErrorDetail(BaseModel):
    description: str = Field(description="Why this instance failed.")


class ErrorResult(BaseModel):
    error: ErrorDetail


class PredictResponse(BaseModel):
    predictions: list[EmbeddingResult | ErrorResult] = Field(
        description="One item per instance, in request order. A bad instance does not fail the others."
    )
