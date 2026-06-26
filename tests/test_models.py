"""Unit tests for model training and evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from amr_guard.models.train import train_rule_baseline, train_logistic_regression, train_xgboost
from amr_guard.models.evaluate import evaluate_model
from amr_guard.models.predict import predict_single, predict_batch


def _make_data(n: int = 100, n_features: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.integers(0, 5, size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(rng.integers(0, 2, size=n), name="label")
    return X, y


def test_rule_baseline_trains():
    X, y = _make_data()
    model = train_rule_baseline(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_logistic_regression_trains():
    X, y = _make_data()
    model = train_logistic_regression(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_xgboost_trains():
    X, y = _make_data()
    model = train_xgboost(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_evaluate_returns_required_metrics():
    X, y = _make_data()
    model = train_rule_baseline(X, y)
    metrics = evaluate_model(model, X, y, model_name="Test", plots_dir="/tmp")
    for key in ["accuracy", "precision", "recall", "f1", "mcc", "confusion_matrix"]:
        assert key in metrics


def test_evaluate_metrics_in_range():
    X, y = _make_data()
    model = train_logistic_regression(X, y)
    metrics = evaluate_model(model, X, y, model_name="LR Test", plots_dir="/tmp")
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert -1.0 <= metrics["mcc"] <= 1.0


def test_predict_single_output_shape():
    X, y = _make_data()
    model = train_xgboost(X, y)
    features = X.iloc[0].to_dict()
    result = predict_single(model, features)
    assert "prediction" in result
    assert "probability" in result
    assert "confidence" in result
    assert result["prediction"] in {"Resistant", "Susceptible"}
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_batch_output_shape():
    X, y = _make_data()
    model = train_xgboost(X, y)
    result = predict_batch(model, X)
    assert len(result) == len(X)
    assert "prediction" in result.columns
    assert "probability" in result.columns
    assert "confidence" in result.columns


def test_model_metric_above_random():
    X, y = _make_data(n=200)
    model = train_xgboost(X, y)
    metrics = evaluate_model(model, X, y, model_name="XGB", plots_dir="/tmp")
    # XGBoost on training data should beat random (0.5 accuracy)
    assert metrics["accuracy"] > 0.5
