"""Prediction interface for genomic sequence AMR models.

Loads saved artifact bundle containing model, SVD transformer, k-mer vocabulary,
and decision threshold. Uses canonical prepare_features to handle input FASTA,
sequences, or k-mer frequency dictionaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from amr_guard.data.preprocess import prepare_features
from amr_guard.utils.config import load_params


def default_model_path() -> Path:
    config = load_params()
    return Path(config["paths"]["models_dir"]) / config["prediction"]["default_model_file"]


def decision_threshold() -> float:
    report_path = Path(load_params()["paths"]["reports_dir"]) / "decision_threshold.json"
    if not report_path.exists():
        return 0.5
    try:
        with open(report_path) as f:
            return json.load(f)["threshold"]
    except Exception:
        return 0.5


def load_model(model_path: str | Path | None = None):
    path = model_path if model_path is not None else default_model_path()
    return joblib.load(path)


def _unpack_artifact(artifact_or_model) -> tuple[object, object | None, list[str], float, dict]:
    if isinstance(artifact_or_model, dict) and "model" in artifact_or_model:
        raw_model = artifact_or_model["model"]
        svd = artifact_or_model.get("svd", None)
        vocab = artifact_or_model.get("vocabulary", artifact_or_model.get("feature_names", []))
        thresh = float(artifact_or_model.get("decision_threshold", decision_threshold()))
        metadata = artifact_or_model.get("metadata", {})
        return raw_model, svd, vocab, thresh, metadata
    else:
        raw_model = artifact_or_model
        thresh = decision_threshold()
        return raw_model, None, [], thresh, {}


def predict_single(artifact_or_model, inputs: Any, threshold: float | None = None) -> dict:
    raw_model, svd, vocab, artifact_thresh, metadata = _unpack_artifact(artifact_or_model)
    thresh = threshold if threshold is not None else artifact_thresh
    thresholds_config = load_params()["prediction"]["confidence_thresholds"]

    # Process features using canonical preparation path
    X = prepare_features(inputs, feature_names=vocab)

    # Apply SVD transformation if model was trained on SVD features
    if svd is not None:
        X_trans = pd.DataFrame(svd.transform(X), columns=[f"svd_{i}" for i in range(svd.n_components)])
    else:
        X_trans = X

    prob = float(raw_model.predict_proba(X_trans)[0][1])
    pred = int(prob >= thresh)

    confidence = "Low"
    for level, level_threshold in thresholds_config.items():
        if prob >= level_threshold:
            confidence = level
            break

    return {
        "prediction": "Resistant" if pred == 1 else "Susceptible",
        "probability": round(prob, 4),
        "confidence": confidence,
        "threshold": thresh,
        "model_version": metadata.get("model_version", "ecoli-cipro-v1"),
        "feature_version": metadata.get("feature_version", "kmer-6-v1"),
    }


def predict_batch(artifact_or_model, inputs: Any, threshold: float | None = None) -> pd.DataFrame:
    raw_model, svd, vocab, artifact_thresh, metadata = _unpack_artifact(artifact_or_model)
    thresh = threshold if threshold is not None else artifact_thresh
    thresholds_config = load_params()["prediction"]["confidence_thresholds"]

    X = prepare_features(inputs, feature_names=vocab)

    if svd is not None:
        X_trans = pd.DataFrame(svd.transform(X), columns=[f"svd_{i}" for i in range(svd.n_components)])
    else:
        X_trans = X

    probs = raw_model.predict_proba(X_trans)[:, 1]
    preds = (probs >= thresh).astype(int)

    def conf(p: float) -> str:
        for level, level_threshold in thresholds_config.items():
            if p >= level_threshold:
                return level
        return "Low"

    return pd.DataFrame({
        "prediction": ["Resistant" if p == 1 else "Susceptible" for p in preds],
        "probability": probs.round(4),
        "confidence": [conf(p) for p in probs],
        "threshold": thresh,
        "model_version": metadata.get("model_version", "ecoli-cipro-v1"),
        "feature_version": metadata.get("feature_version", "kmer-6-v1"),
    })
