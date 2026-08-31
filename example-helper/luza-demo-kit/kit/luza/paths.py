"""Canonical filesystem locations for the project.

Every script derives its paths from here instead of re-deriving
``Path(__file__).parent.parent...`` chains (which broke when scripts moved).
"""

from __future__ import annotations

from pathlib import Path

# repo root = parent of the ``luza`` package directory
ROOT = Path(__file__).resolve().parent.parent

MINING_DIR = ROOT / "1-mining"
CONFIG_DIR = MINING_DIR / "config"
RAW_DIR = MINING_DIR / "data" / "raw"
CLEAN_DIR = MINING_DIR / "data" / "cleaned"
EXPORT_DIR = MINING_DIR / "data" / "exports"

ANALYSIS_DIR = ROOT / "2-analysis-dashboards"
ANALYSIS_OUTPUT_DIR = ANALYSIS_DIR / "output"

ML_DIR = ROOT / "3-ml-prediction"
ML_OUTPUT_DIR = ML_DIR / "output"

DOCS_DIR = ROOT / "4-docs"
TECH_DIR = ROOT / "5-behind-tech"

MANIFEST_PATH = CLEAN_DIR / "manifest.json"


def ensure_dirs() -> None:
    """Create the output/data directories that scripts write into."""
    for d in (RAW_DIR, CLEAN_DIR, EXPORT_DIR, ANALYSIS_OUTPUT_DIR, ML_OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
