"""Phase 6: PSI-based drift detection between training data and served predictions.

Reference distribution = data/processed/train.csv (the encoded feature matrix
the model was trained on). Current distribution = reports/prediction_log.csv,
appended to by amr_guard.monitoring.logger on every served /predict and
/batch-predict request. Feature values in both are in the same representation
(the client is expected to send already-encoded values matching training).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from amr_guard.utils.config import load_params


def _psi(reference: pd.Series, current: pd.Series, bins: int) -> float:
    reference = reference.dropna().astype(float)
    current = current.dropna().astype(float)
    if reference.empty or current.empty:
        return 0.0

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.linspace(reference.min(), reference.max() + 1e-9, bins + 1)

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-4, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-4, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_drift_report() -> dict:
    config = load_params()
    monitoring_config = config["monitoring"]

    reference_path = Path(config["paths"]["processed_dir"]) / "train.csv"
    log_path = Path(config["paths"]["reports_dir"]) / monitoring_config["log_file"]

    min_samples = monitoring_config["min_samples"]
    threshold = monitoring_config["psi_threshold"]
    bins = monitoring_config["psi_bins"]

    if not reference_path.exists():
        return {"status": "no_reference", "detail": f"{reference_path} not found. Run preprocessing first."}
    if not log_path.exists():
        return {"status": "no_data", "detail": f"{log_path} not found. No predictions served yet."}

    reference_df = pd.read_csv(reference_path).drop(columns=["label"], errors="ignore")
    current_df = pd.read_csv(log_path)

    n_samples = len(current_df)
    if n_samples < min_samples:
        return {
            "status": "insufficient_data",
            "detail": f"Only {n_samples} logged predictions, need at least {min_samples}.",
            "n_samples": n_samples,
        }

    shared_features = [c for c in reference_df.columns if c in current_df.columns]
    features = []
    for col in shared_features:
        score = _psi(reference_df[col], current_df[col], bins=bins)
        features.append({"feature": col, "psi": round(score, 4), "drifted": score > threshold})

    features.sort(key=lambda f: f["psi"], reverse=True)

    return {
        "status": "ok",
        "n_samples": n_samples,
        "psi_threshold": threshold,
        "n_drifted_features": sum(1 for f in features if f["drifted"]),
        "features": features,
    }
