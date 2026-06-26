"""
Model training for Phase 1.

Trains three model tracks:
  1. Rule baseline  – predicts majority class (simple frequency-based rule)
  2. Logistic Regression  – interpretable ML baseline
  3. XGBoost  – primary Phase 1 model saved as artifact

All models are evaluated via amr_guard.models.evaluate.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from amr_guard.models.evaluate import evaluate_model


def train_rule_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> DummyClassifier:
    clf = DummyClassifier(strategy="most_frequent", random_state=42)
    clf.fit(X_train, y_train)
    return clf


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

    # Calibrate probabilities using cross-validation on training set
    calibrated = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


def run(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str | Path = "models/artifacts",
    reports_dir: str | Path = "reports",
) -> dict:
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # --- Rule baseline ---
    print("Training rule baseline...")
    rule_model = train_rule_baseline(X_train, y_train)
    results["rule_baseline"] = evaluate_model(rule_model, X_test, y_test, model_name="Rule Baseline")

    # --- Logistic Regression ---
    print("Training logistic regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    results["logistic_regression"] = evaluate_model(lr_model, X_test, y_test, model_name="Logistic Regression")
    joblib.dump(lr_model, output_dir / "logistic_regression.pkl")

    # --- XGBoost ---
    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    results["xgboost"] = evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")
    joblib.dump(xgb_model, output_dir / "xgboost_phase1.pkl")
    print(f"XGBoost model saved to {output_dir / 'xgboost_phase1.pkl'}")

    # --- Comparison table ---
    comparison = []
    for name, metrics in results.items():
        row = {"model": name}
        row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        comparison.append(row)
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(reports_dir / "model_comparison.csv", index=False)

    full_report = {k: {m: v for m, v in metrics.items() if not isinstance(v, np.ndarray)} for k, metrics in results.items()}
    with open(reports_dir / "evaluation_report.json", "w") as f:
        json.dump(full_report, f, indent=2)

    print("\n=== Model Comparison ===")
    print(comparison_df.to_string(index=False))

    return {"models": {"rule_baseline": rule_model, "logistic_regression": lr_model, "xgboost": xgb_model}, "results": results}


if __name__ == "__main__":
    from amr_guard.data.preprocess import run as preprocess

    data = preprocess(r"C:\Users\2504787\Downloads\BVBRC_genome_amr.csv")
    run(data["X_train"], data["X_test"], data["y_train"], data["y_test"])
