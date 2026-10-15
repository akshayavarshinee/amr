"""Phase 4: FastAPI prediction service.

Endpoints: /health, /predict, /batch-predict, /models/current
(monitoring/drift and retrain endpoints are Phase 6/7 scope).

All service metadata (title, version, batch size limit) comes from
params.yaml. Model loading is lazy + cached so a missing artifact
doesn't crash the app at import time.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException

from amr_guard.models.predict import decision_threshold, default_model_path, load_model, predict_batch, predict_single
from amr_guard.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    DriftResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    RetrainResponse,
)
from amr_guard.models.retrain import run_retrain
from amr_guard.monitoring.drift import compute_drift_report
from amr_guard.monitoring.logger import log_prediction, log_predictions
from amr_guard.utils.config import load_params

config = load_params()
api_config = config["api"]

app = FastAPI(title=api_config["title"], version=api_config["version"])


@lru_cache(maxsize=1)
def get_model():
    return load_model()


def _model_or_503():
    try:
        return get_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model artifact not found: {exc}") from exc


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model_loaded = default_model_path().exists()
    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.get("/models/current", response_model=ModelInfoResponse)
def models_current() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_version=api_config["model_version"],
        model_path=str(default_model_path()),
        decision_threshold=decision_threshold(),
        registered_model_name=config["mlflow"].get("registered_model_name"),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model = _model_or_503()
    input_payload = request.model_dump(exclude_none=True)
    if not input_payload:
        raise HTTPException(status_code=400, detail="Must provide fasta_content, sequence, or features dictionary.")
    result = predict_single(model, input_payload)
    log_prediction(input_payload)
    result["model_version"] = api_config["model_version"]
    return PredictResponse(**result)


@app.post("/batch-predict", response_model=BatchPredictResponse)
def batch_predict(request: BatchPredictRequest) -> BatchPredictResponse:
    max_batch_size = api_config["max_batch_size"]
    if len(request.rows) > max_batch_size:
        raise HTTPException(status_code=400, detail=f"Batch size {len(request.rows)} exceeds max_batch_size={max_batch_size}")

    model = _model_or_503()
    df = pd.DataFrame(request.rows)
    results_df = predict_batch(model, df)
    log_predictions(request.rows)
    results = [
        PredictResponse(**row, model_version=api_config["model_version"], warnings=[])
        for row in results_df.to_dict(orient="records")
    ]
    return BatchPredictResponse(results=results)


@app.get("/monitoring/drift", response_model=DriftResponse)
def monitoring_drift() -> DriftResponse:
    return DriftResponse(**compute_drift_report())


@app.post("/retrain", response_model=RetrainResponse)
def retrain() -> RetrainResponse:
    result = run_retrain()
    if result["promoted"]:
        get_model.cache_clear()
    return RetrainResponse(**result)
