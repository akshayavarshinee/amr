"""MLflow tracking wrapper.

Reads tracking_uri / experiment_name from params.yaml (see
amr_guard.utils.config.load_params) so nothing is hardcoded here.

mlflow is an optional dependency (declared in pyproject.toml but not
required to be installed). If it isn't installed, every function
degrades to a no-op with a printed warning so the training pipeline
still runs end-to-end without it.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from amr_guard.utils.config import load_params

import os

try:
    import mlflow

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False


def _mlflow_config() -> dict:
    return load_params().get("mlflow", {})


@contextmanager
def start_run(run_name: str | None = None):
    """Context manager: starts an MLflow run against the configured
    tracking URI / experiment. Yields None and skips tracking if mlflow
    is not installed.
    """
    if not MLFLOW_AVAILABLE:
        print("[mlflow] not installed, skipping experiment tracking for this run.")
        yield None
        return

    config = _mlflow_config()
    mlflow.set_tracking_uri(config.get("tracking_uri"))
    mlflow.set_experiment(config.get("experiment_name"))
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_params(params: dict) -> None:
    if not MLFLOW_AVAILABLE:
        return
    mlflow.log_params(params)


def log_metrics(metrics: dict) -> None:
    if not MLFLOW_AVAILABLE:
        return
    numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    mlflow.log_metrics(numeric_metrics)


def log_artifact(path: str | Path) -> None:
    if not MLFLOW_AVAILABLE:
        return
    mlflow.log_artifact(str(path))


def log_model(model, artifact_path: str, registered_model_name: str | None = None) -> None:
    if not MLFLOW_AVAILABLE:
        return
    mlflow.sklearn.log_model(
        model,
        artifact_path=artifact_path,
        registered_model_name=registered_model_name,
    )
