"""Loads params.yaml so pipeline stages share one source of config
instead of hardcoding values inline.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[3] / "params.yaml"


@lru_cache(maxsize=None)
def load_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)
