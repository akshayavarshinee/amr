"""
Prediction interface: loads a saved model artifact and returns predictions
for a feature row or a DataFrame of rows.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


DEFAULT_MODEL_PATH = Path("models/artifacts/xgboost_phase1.pkl")

CONFIDENCE_THRESHOLDS = {
    "High": 0.75,
    "Medium": 0.55,
    "Low": 0.0,
}


def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    return joblib.load(model_path)


def predict_single(model, features: dict) -> dict:
    X = pd.DataFrame([features])
    prob = model.predict_proba(X)[0][1]
    pred = int(prob >= 0.5)

    confidence = "Low"
    for level, threshold in CONFIDENCE_THRESHOLDS.items():
        if prob >= threshold:
            confidence = level
            break

    return {
        "prediction": "Resistant" if pred == 1 else "Susceptible",
        "probability": round(float(prob), 4),
        "confidence": confidence,
    }


def predict_batch(model, df: pd.DataFrame) -> pd.DataFrame:
    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= 0.5).astype(int)

    def conf(p: float) -> str:
        for level, threshold in CONFIDENCE_THRESHOLDS.items():
            if p >= threshold:
                return level
        return "Low"

    return pd.DataFrame({
        "prediction": ["Resistant" if p == 1 else "Susceptible" for p in preds],
        "probability": probs.round(4),
        "confidence": [conf(p) for p in probs],
    })
