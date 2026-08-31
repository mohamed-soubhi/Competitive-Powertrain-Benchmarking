"""Centralised data loading + a dataset manifest.

Replaces the three near-identical ``load_data`` / ``load_specs`` implementations
in the analysis and ML scripts, and adds a manifest so every run records which
input file (and content hash) it consumed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from luza.paths import CLEAN_DIR, MANIFEST_PATH, RAW_DIR

# Columns coerced to numeric on load (superset used across all scripts).
_NUMERIC_COLS = [
    "model_year", "battery_useable_kwh", "battery_nominal_kwh", "architecture_v",
    "range_real_km", "range_rated_km", "range_wltp_km", "efficiency_wh_km",
    "power_kw", "torque_nm", "accel_0_100_s", "top_speed_kmh",
    "charging_dc_kw", "charging_ac_kw", "charge_time_10_80_min",
    "cargo_volume_l", "weight_kg", "cell_count",
]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def latest(pattern: str, directory: Path = CLEAN_DIR) -> Path | None:
    """Newest file matching ``pattern`` in ``directory`` (lexicographic on name)."""
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def read_raw_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in _NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_manifest() -> dict[str, Any] | None:
    if not MANIFEST_PATH.exists():
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _manifest_file(dataset: str) -> Path | None:
    """The ``CLEAN_DIR`` file the manifest pins for ``dataset``, if it still exists.

    ``reclean.py`` writes the manifest and it is the authority on which dataset is
    active. Without this, ``latest()`` would silently switch to any newer-*named*
    ``ev_database_*.csv`` a re-scrape drops in.
    """
    manifest = read_manifest()
    if not manifest:
        return None
    entry = manifest.get("datasets", {}).get(dataset)
    if not entry or "file" not in entry:
        return None
    cand = CLEAN_DIR / entry["file"]
    return cand if cand.exists() else None


def load_ev_specs(path: str | Path | None = None) -> pd.DataFrame:
    """Load the cleaned EV specs CSV (latest if ``path`` omitted).

    Adds a ``brand`` column and coerces numeric columns — the shared version of
    logic that was copy-pasted into three scripts.
    """
    if path is None:
        path = _manifest_file("ev_specs") or latest("ev_database_*.csv")
        if path is None:
            raise FileNotFoundError(f"no ev_database_*.csv in {CLEAN_DIR}")
    df = pd.read_csv(path)
    if "name" in df.columns:
        df["brand"] = df["name"].astype(str).str.split().str[0]
    return _coerce_numeric(df)


def load_patents(paths: list[str | Path] | None = None) -> pd.DataFrame:
    """Load + concat + de-duplicate patent CSVs (latest only if ``paths`` omitted)."""
    if paths is None:
        one = _manifest_file("patents") or latest("patents_greyb_*.csv")
        paths = [one] if one else []
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "patent_id" in df.columns:
        df = df.drop_duplicates(subset=["patent_id"]).reset_index(drop=True)
    return df


def write_manifest(datasets: dict[str, Path]) -> dict[str, Any]:
    """Record the active input files + their hashes to ``manifest.json``.

    ``datasets`` maps a logical name ('ev_specs', 'patents') to a file path.
    """
    entries = {}
    for name, path in datasets.items():
        path = Path(path)
        try:
            rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
        except OSError:
            rows = None
        entries[name] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "rows": rows,
        }
    manifest = {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": entries,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


_MANIFEST_PATTERNS = {
    "ev_specs": "ev_database_*.csv",
    "patents": "patents_greyb_*.csv",
}


def provenance_line(dataset: str = "ev_specs") -> str:
    """One-line "which file / what hash / how many rows" stamp for a dashboard.

    Prefers ``manifest.json`` (written by ``reclean.py`` / ``write_manifest``);
    falls back to hashing the latest matching CSV directly.
    """
    manifest = read_manifest()
    if manifest and dataset in manifest.get("datasets", {}):
        d = manifest["datasets"][dataset]
        return (
            f'{d["file"]} | sha256:{d["sha256"][:12]} | {d["rows"]} rows '
            f'| manifest {manifest.get("written_at", "?")}'
        )
    path = latest(_MANIFEST_PATTERNS.get(dataset, f"{dataset}_*.csv"))
    if path is None:
        return f"{dataset}: no dataset file found"
    rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
    return f"{path.name} | sha256:{sha256_file(path)[:12]} | {rows} rows | (no manifest)"


__all__ = [
    "RAW_DIR", "CLEAN_DIR",
    "sha256_file", "latest", "read_raw_json",
    "load_ev_specs", "load_patents",
    "write_manifest", "read_manifest", "provenance_line",
]
