"""Unit tests for genomic model training, evaluation, artifact bundling, and prediction."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from amr_guard.data.kmer import get_kmer_vocabulary
from amr_guard.models.train import train_logistic_regression, train_xgboost, fit_svd_transformer
from amr_guard.models.evaluate import evaluate_model
from amr_guard.models.predict import predict_single, predict_batch


def _make_genomic_data(n: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    vocab = get_kmer_vocabulary(k=6)
    X = pd.DataFrame(rng.uniform(0, 0.01, size=(n, len(vocab))), columns=vocab)
    y = pd.Series(rng.integers(0, 2, size=n), name="label")
    return X, y, vocab


class TestGenomicModels(unittest.TestCase):

    def test_logistic_regression_trains(self):
        X, y, _ = _make_genomic_data()
        model = train_logistic_regression(X, y)
        preds = model.predict(X)
        self.assertEqual(len(preds), len(y))

    def test_svd_fitting_on_train_only(self):
        X, _, _ = _make_genomic_data(n=30)
        svd = fit_svd_transformer(X, n_components=10)
        X_trans = svd.transform(X)
        self.assertEqual(X_trans.shape, (30, 10))

    def test_xgboost_trains(self):
        X, y, _ = _make_genomic_data()
        svd = fit_svd_transformer(X, n_components=5)
        X_svd = pd.DataFrame(svd.transform(X), columns=[f"svd_{i}" for i in range(5)])
        model = train_xgboost(X_svd, y)
        preds = model.predict(X_svd)
        self.assertEqual(len(preds), len(y))

    def test_evaluate_returns_required_metrics(self):
        X, y, _ = _make_genomic_data()
        model = train_logistic_regression(X, y)
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = evaluate_model(model, X, y, model_name="TestLR", plots_dir=tmpdir)
            for key in ["accuracy", "precision", "recall", "f1", "mcc", "confusion_matrix"]:
                self.assertIn(key, metrics)
            self.assertTrue(0.0 <= metrics["accuracy"] <= 1.0)
            self.assertTrue(0.0 <= metrics["f1"] <= 1.0)

    def test_predict_single_output_shape(self):
        X, y, vocab = _make_genomic_data()
        svd = fit_svd_transformer(X, n_components=5)
        X_svd = pd.DataFrame(svd.transform(X), columns=[f"svd_{i}" for i in range(5)])
        model = train_xgboost(X_svd, y)

        artifact = {
            "model": model,
            "svd": svd,
            "vocabulary": vocab,
            "feature_names": vocab,
            "decision_threshold": 0.5,
            "metadata": {"model_version": "v1", "feature_version": "kmer-6-v1"},
        }
        features = X.iloc[0].to_dict()
        result = predict_single(artifact, features)
        self.assertIn("prediction", result)
        self.assertIn("probability", result)
        self.assertIn("confidence", result)
        self.assertIn("threshold", result)
        self.assertIn(result["prediction"], {"Resistant", "Susceptible"})
        self.assertTrue(0.0 <= result["probability"] <= 1.0)

    def test_predict_batch_output_shape(self):
        X, y, vocab = _make_genomic_data()
        svd = fit_svd_transformer(X, n_components=5)
        X_svd = pd.DataFrame(svd.transform(X), columns=[f"svd_{i}" for i in range(5)])
        model = train_xgboost(X_svd, y)

        artifact = {
            "model": model,
            "svd": svd,
            "vocabulary": vocab,
            "feature_names": vocab,
            "decision_threshold": 0.5,
            "metadata": {"model_version": "v1", "feature_version": "kmer-6-v1"},
        }
        result = predict_batch(artifact, X)
        self.assertEqual(len(result), len(X))
        self.assertIn("prediction", result.columns)
        self.assertIn("probability", result.columns)


if __name__ == "__main__":
    unittest.main()
