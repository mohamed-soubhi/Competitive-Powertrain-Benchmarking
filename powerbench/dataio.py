"""DuckDB-backed store for the cleaned HDV dataset + a run manifest.

``reclean.py`` writes; the Streamlit app and analysis scripts read. The
manifest (``1-mining/data/cleaned/manifest.json``) is the authority on which
snapshot is live — loaders trust it before falling back to "newest file".
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from powerbench.paths import CLEAN_DIR, DUCKDB_PATH, MANIFEST_PATH, RAW_DIR

HDV_TABLE = "hdv"
RAW_PREFIX = "hdv_co2"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def latest_raw_snapshot() -> Path | None:
    files = sorted(RAW_DIR.glob(f"{RAW_PREFIX}_*.json"))
    return files[-1] if files else None


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)


def write_hdv(df: pd.DataFrame) -> Path:
    """Replace the ``hdv`` table in DuckDB and mirror it to Parquet."""
    con = connect()
    try:
        con.register("df_in", df)
        con.execute(f"CREATE OR REPLACE TABLE {HDV_TABLE} AS SELECT * FROM df_in")
        con.unregister("df_in")
    finally:
        con.close()
    parquet = CLEAN_DIR / f"{HDV_TABLE}.parquet"
    df.to_parquet(parquet, index=False)
    return parquet


def load_hdv() -> pd.DataFrame:
    """Read the cleaned HDV table (DuckDB first, then Parquet fallback)."""
    if DUCKDB_PATH.exists():
        con = connect(read_only=True)
        try:
            return con.execute(f"SELECT * FROM {HDV_TABLE}").fetch_df()
        except duckdb.CatalogException:
            pass
        finally:
            con.close()
    parquet = CLEAN_DIR / f"{HDV_TABLE}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    raise FileNotFoundError("no cleaned HDV dataset — run 2-pipeline/reclean.py")


def read_manifest() -> dict[str, Any] | None:
    if not MANIFEST_PATH.exists():
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def write_manifest(
    *, parquet_path: Path, rows: int, source_snapshot: Path, rejects: int
) -> dict[str, Any]:
    manifest = {
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": {
            HDV_TABLE: {
                "file": parquet_path.name,
                "duckdb": DUCKDB_PATH.name,
                "sha256": sha256_file(parquet_path),
                "rows": rows,
                "rejected_rows": rejects,
                "source_snapshot": source_snapshot.name,
                "source_snapshot_sha256": sha256_file(source_snapshot)[:16],
            }
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def provenance_line(dataset: str = HDV_TABLE) -> str:
    m = read_manifest()
    if m and dataset in m.get("datasets", {}):
        d = m["datasets"][dataset]
        return (
            f'{d["file"]} | sha256:{d["sha256"][:12]} | {d["rows"]} rows '
            f'({d["rejected_rows"]} rejected) | from {d["source_snapshot"]} '
            f'| manifest {m["written_at"]}'
        )
    return f"{dataset}: no manifest — run 2-pipeline/reclean.py"
