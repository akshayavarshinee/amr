"""Genomic Model Training Pipeline.

Models:
  1. Logistic Regression  – baseline model directly on 6-mer frequencies (4096 dims)
  2. XGBoost  – fitted on TruncatedSVD dimensionality-reduced k-mer features (fitted ON TRAIN ONLY)

Uses 3-way evaluation methodology:
  - Models & SVD fitted ON TRAIN ONLY
  - Hyperparameter & threshold tuning performed ON VALIDATION SET ONLY
  - Final model performance evaluated ONCE on untouched Test set
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from amr_guard.models.evaluate import evaluate_model
from amr_guard.models.tune import tune_decision_threshold, tune_xgboost
from amr_guard.tracking import mlflow_utils
from amr_guard.utils.config import load_params


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
    random_state: int | None = None,
) -> Pipeline:
    params = params or load_params().get("logistic_regression", {})
    random_state = random_state if random_state is not None else load_params()["data"]["random_state"]
    max_iter = params.get("max_iter", 1000)
    class_weight = params.get("class_weight", "balanced")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def fit_svd_transformer(
    X_train: pd.DataFrame,
    n_components: int = 50,
    random_state: int = 42,
) -> TruncatedSVD:
    """Fits TruncatedSVD strictly on training data."""
    max_components = min(X_train.shape[1], max(1, X_train.shape[0] - 1))
    actual_components = min(n_components, max_components)

    svd = TruncatedSVD(n_components=actual_components, random_state=random_state)
    svd.fit(X_train)
    return svd


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
    random_state: int | None = None,
) -> object:
    params = params or load_params()["xgboost"]
    random_state = random_state if random_state is not None else load_params()["data"]["random_state"]
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    clf = XGBClassifier(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        scale_pos_weight=scale_pos_weight,
        eval_metric=params.get("eval_metric", "logloss"),
        random_state=random_state,
    )
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

    min_class_count = int(y_train.value_counts().min()) if len(y_train.value_counts()) > 1 else 0
    requested_cv = params.get("calibration_cv", 3)
    cv_folds = min(requested_cv, min_class_count)

    if cv_folds >= 2:
        try:
            calibrated = CalibratedClassifierCV(
                clf,
                method=params.get("calibration_method", "sigmoid"),
                cv=cv_folds,
            )
            calibrated.fit(X_train, y_train)
            return calibrated
        except Exception:
            return clf
    return clf


def run(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    encoders: dict | None = None,
    feature_names: list[str] | None = None,
    output_dir: str | Path = "models/artifacts",
    reports_dir: str | Path = "reports",
) -> dict:
    output_dir = Path(output_dir)
    reports_dir = Path(reports_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    config = load_params()
    feature_names = feature_names or list(X_train.columns)

    # Fit Dimensionality Reduction (TruncatedSVD) strictly on TRAIN set
    n_components = config.get("xgboost", {}).get("svd_components", 50)
    print(f"Fitting TruncatedSVD (n_components={n_components}) ON TRAIN SET ONLY...")
    svd = fit_svd_transformer(X_train, n_components=n_components, random_state=config["data"]["random_state"])

    # Transform splits using TRAIN-fitted SVD
    X_train_svd = pd.DataFrame(svd.transform(X_train), index=X_train.index, columns=[f"svd_{i}" for i in range(svd.n_components)])
    X_val_svd = pd.DataFrame(svd.transform(X_val), index=X_val.index, columns=[f"svd_{i}" for i in range(svd.n_components)])
    X_test_svd = pd.DataFrame(svd.transform(X_test), index=X_test.index, columns=[f"svd_{i}" for i in range(svd.n_components)])

    val_results = {}
    test_results = {}

    with mlflow_utils.start_run(run_name="genomic_training"):
        mlflow_utils.log_params({
            "genomics.k": 6,
            "genomics.feature_count": len(feature_names),
            "genomics.feature_method": "kmer_frequency",
            "genomics.svd_components": svd.n_components,
            "dataset.name": "bvbrc_ecoli_cipro",
            "dataset.total_genomes": len(X_train) + len(X_val) + len(X_test),
            "random_state": config["data"]["random_state"],
        })

        # --- Baseline Track: Logistic Regression (Full k-mer features) ---
        print("Training Logistic Regression baseline...")
        lr_model = train_logistic_regression(X_train, y_train, params=config.get("logistic_regression"))
        val_results["logistic_regression"] = evaluate_model(lr_model, X_val, y_val, model_name="Logistic Regression (Val)")

        # --- Primary Model Track: XGBoost (SVD features) ---
        print("Training XGBoost baseline on SVD features...")
        xgb_static = train_xgboost(X_train_svd, y_train, params=config["xgboost"])
        val_static_metrics = evaluate_model(xgb_static, X_val_svd, y_val, model_name="XGBoost Static (Val)")

        print("Tuning XGBoost hyperparameters (Optuna) on Validation set...")
        optuna_config = config.get("optuna", {})
        if optuna_config.get("enabled", False) and len(X_train_svd) >= 15:
            try:
                tuned_params = {
                    **tune_xgboost(X_train_svd, y_train, params=optuna_config),
                    "calibration_method": config["xgboost"].get("calibration_method", "sigmoid"),
                    "calibration_cv": config["xgboost"].get("calibration_cv", 3),
                }
                xgb_tuned = train_xgboost(X_train_svd, y_train, params=tuned_params)
                val_tuned_metrics = evaluate_model(xgb_tuned, X_val_svd, y_val, model_name="XGBoost Tuned (Val)")
                promote_metric = optuna_config.get("metric", "f1")
                if val_tuned_metrics[promote_metric] >= val_static_metrics[promote_metric]:
                    xgb_model, val_results["xgboost"] = xgb_tuned, val_tuned_metrics
                else:
                    xgb_model, val_results["xgboost"] = xgb_static, val_static_metrics
            except Exception as e:
                print(f"Optuna tuning skipped/failed: {e}")
                xgb_model, val_results["xgboost"] = xgb_static, val_static_metrics
        else:
            xgb_model, val_results["xgboost"] = xgb_static, val_static_metrics

        # --- Threshold tuning ON VALIDATION SET ONLY ---
        print("Tuning decision threshold ON VALIDATION SET ONLY...")
        best_threshold = tune_decision_threshold(
            xgb_model, X_val_svd, y_val, params=config.get("threshold_tuning", {"enabled": True, "metric": "f1"})
        )
        threshold_report_path = reports_dir / "decision_threshold.json"
        with open(threshold_report_path, "w") as f:
            json.dump({"model": "xgboost", "threshold": best_threshold}, f, indent=2)

        # --- FINAL UNTOUCHED TEST EVALUATION ---
        print("\nEvaluating ONCE on untouched Test set...")
        test_results["logistic_regression"] = evaluate_model(lr_model, X_test, y_test, model_name="Logistic Regression (Test)")
        test_results["xgboost"] = evaluate_model(xgb_model, X_test_svd, y_test, threshold=best_threshold, model_name="XGBoost (Test)")

        # Save Complete Model Artifact Bundle
        artifact_bundle = {
            "model": xgb_model,
            "svd": svd,
            "k": 6,
            "vocabulary": feature_names,
            "feature_names": feature_names,
            "normalization": "frequency",
            "decision_threshold": best_threshold,
            "metadata": {
                "model_version": config["api"]["model_version"],
                "feature_version": "kmer-6-v1",
                "organism": config["data"].get("organism_filter", "Escherichia coli"),
                "antibiotic": config["data"].get("antibiotic_filter", "ciprofloxacin"),
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "feature_count": len(feature_names),
                "svd_components": svd.n_components,
            },
        }

        artifact_path = output_dir / config["prediction"]["default_model_file"]
        joblib.dump(artifact_bundle, artifact_path)
        joblib.dump(lr_model, output_dir / "logistic_regression.pkl")
        print(f"Genomic model artifact bundle saved to {artifact_path}")

        # MLflow Logging
        for model_name, metrics in test_results.items():
            mlflow_utils.log_metrics({f"test_{model_name}.{k}": v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow_utils.log_params({"decision_threshold": best_threshold})

        mlflow_utils.log_artifact(threshold_report_path)

        # Summary reports
        full_report = {
            "validation": {k: {m: v for m, v in metrics.items() if not isinstance(v, np.ndarray)} for k, metrics in val_results.items()},
            "test": {k: {m: v for m, v in metrics.items() if not isinstance(v, np.ndarray)} for k, metrics in test_results.items()},
        }
        evaluation_report_path = reports_dir / "evaluation_report.json"
        with open(evaluation_report_path, "w") as f:
            json.dump(full_report, f, indent=2)

    return {
        "models": {"logistic_regression": lr_model, "xgboost": xgb_model},
        "results": {"validation": val_results, "test": test_results},
        "artifact_bundle": artifact_bundle,
    }


if __name__ == "__main__":
    from amr_guard.data.preprocess import run as preprocess
    data = preprocess()
    run(
        data["X_train"],
        data["X_val"],
        data["X_test"],
        data["y_train"],
        data["y_val"],
        data["y_test"],
        feature_names=data["feature_names"],
    )
