"""Phase 7: automated retraining pipeline.

Retrains on the latest raw data into a scratch location, compares the new
XGBoost model's metric against the currently-deployed model's recorded
metric on the test set, and only promotes (overwrites the live artifact/report)
if it beats the incumbent by at least `retrain.min_improvement`. The incumbent
artifact is backed up before being overwritten so a bad promotion can be rolled back.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from amr_guard.data.preprocess import run as preprocess
from amr_guard.models.train import run as train
from amr_guard.utils.config import load_params


def _incumbent_metric(reports_dir: Path, metric: str) -> float:
    report_path = reports_dir / "evaluation_report.json"
    if not report_path.exists():
        return float("-inf")
    try:
        with open(report_path) as f:
            report = json.load(f)
        if "test" in report:
            return float(report["test"].get("xgboost", {}).get(metric, float("-inf")))
        return float(report.get("xgboost", {}).get(metric, float("-inf")))
    except Exception:
        return float("-inf")


def _backup_incumbent(output_dir: Path, model_file: str) -> None:
    incumbent_path = output_dir / model_file
    if incumbent_path.exists():
        backup_path = output_dir / model_file.replace(".pkl", "_prev.pkl")
        shutil.copy2(incumbent_path, backup_path)


def _promote(scratch_dir: Path, output_dir: Path, reports_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    for item in (scratch_dir / "models").glob("*"):
        shutil.copy2(item, output_dir / item.name)
    for item in (scratch_dir / "reports").glob("*"):
        shutil.copy2(item, reports_dir / item.name)


def run_retrain(csv_path: str | Path | None = None) -> dict:
    config = load_params()
    retrain_config = config["retrain"]
    metric = retrain_config["metric"]
    min_improvement = retrain_config["min_improvement"]

    csv_path = csv_path if csv_path is not None else config["data"]["raw_csv"]
    output_dir = Path(config["paths"]["models_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    model_file = config["prediction"]["default_model_file"]

    incumbent_score = _incumbent_metric(reports_dir, metric)

    with tempfile.TemporaryDirectory(prefix="amr_retrain_") as scratch:
        scratch_dir = Path(scratch)
        scratch_models = scratch_dir / "models"
        scratch_reports = scratch_dir / "reports"

        data = preprocess(csv_path, output_dir=scratch_dir / "processed")
        result = train(
            data["X_train"],
            data["X_val"],
            data["X_test"],
            data["y_train"],
            data["y_val"],
            data["y_test"],
            encoders=data["encoders"],
            feature_names=data["feature_names"],
            output_dir=scratch_models,
            reports_dir=scratch_reports,
        )

        test_results = result["results"]["test"]
        candidate_score = float(test_results["xgboost"][metric])

        promoted = candidate_score >= incumbent_score + min_improvement
        if promoted or incumbent_score == float("-inf"):
            promoted = True
            if retrain_config["keep_previous"]:
                _backup_incumbent(output_dir, model_file)
            _promote(scratch_dir, output_dir, reports_dir)

    return {
        "promoted": promoted,
        "metric": metric,
        "incumbent_score": incumbent_score if incumbent_score != float("-inf") else 0.0,
        "candidate_score": candidate_score,
        "min_improvement": min_improvement,
    }


if __name__ == "__main__":
    print(run_retrain())
