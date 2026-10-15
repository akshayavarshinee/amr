"""Request/response schemas for FastAPI prediction service."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    fasta_content: str | None = Field(None, description="Raw FASTA format text string.")
    sequence: str | None = Field(None, description="Raw DNA nucleotide sequence string.")
    features: dict[str, Any] | None = Field(None, description="Pre-computed k-mer frequency feature dictionary.")


class PredictResponse(BaseModel):
    prediction: str
    probability: float
    confidence: str
    threshold: float = 0.5
    model_version: str = "ecoli-cipro-v1"
    feature_version: str = "kmer-6-v1"
    warnings: list[str] = []


class BatchPredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., description="List of genomic samples (FASTA content, sequence, or k-mer feature dicts).")


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    feature_version: str = "kmer-6-v1"
    model_path: str
    decision_threshold: float
    registered_model_name: str | None = None


class DriftFeatureResult(BaseModel):
    feature: str
    psi: float
    drifted: bool


class DriftResponse(BaseModel):
    status: str
    detail: str | None = None
    n_samples: int | None = None
    psi_threshold: float | None = None
    n_drifted_features: int | None = None
    features: list[DriftFeatureResult] = []


class RetrainResponse(BaseModel):
    promoted: bool
    metric: str
    incumbent_score: float
    candidate_score: float
    min_improvement: float
