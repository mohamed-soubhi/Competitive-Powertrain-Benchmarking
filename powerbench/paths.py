"""Canonical filesystem locations, resolved from the repo root.

Every script imports paths from here instead of re-deriving
``Path(__file__).parent.parent`` chains.
"""

from __future__ import annotations

from pathlib import Path

# repo root = parent of the ``powerbench`` package directory
ROOT = Path(__file__).resolve().parent.parent

MINING_DIR = ROOT / "1-mining"
CONFIG_DIR = MINING_DIR / "config"
RAW_DIR = MINING_DIR / "data" / "raw"
CLEAN_DIR = MINING_DIR / "data" / "cleaned"

ANALYSIS_DIR = ROOT / "2-analysis-dashboards"
ML_DIR = ROOT / "3-ml-prediction"
APP_DIR = ROOT / "app"

DUCKDB_PATH = CLEAN_DIR / "powerbench.duckdb"
MANIFEST_PATH = CLEAN_DIR / "manifest.json"


def ensure_dirs() -> None:
    """Create the data directories scripts write into."""
    for d in (RAW_DIR, CLEAN_DIR):
        d.mkdir(parents=True, exist_ok=True)
