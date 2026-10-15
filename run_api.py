"""Phase 4 API entry point.

Usage:
    python run_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import uvicorn

from amr_guard.utils.config import load_params

if __name__ == "__main__":
    api_config = load_params()["api"]
    uvicorn.run("amr_guard.api.main:app", host=api_config["host"], port=api_config["port"])
