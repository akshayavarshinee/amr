"""Phase 6: appends served prediction requests to a log used for drift monitoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from amr_guard.utils.config import load_params


def log_path() -> Path:
    config = load_params()
    return Path(config["paths"]["reports_dir"]) / config["monitoring"]["log_file"]


def log_prediction(features: dict) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_dict = {k: v for k, v in features.items() if isinstance(v, (int, float, str, bool)) and k not in ("fasta_content", "sequence")}
    if not clean_dict:
        clean_dict = {"timestamp": pd.Timestamp.now().isoformat()}
    row = pd.DataFrame([clean_dict])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def log_predictions(rows: list[dict]) -> None:
    if not rows:
        return
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
