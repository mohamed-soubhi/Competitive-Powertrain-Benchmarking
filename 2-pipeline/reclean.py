#!/usr/bin/env python3
"""Validate a raw HDV snapshot and load it into the DuckDB store.

    raw JSON snapshot  ->  pydantic gate  ->  + derived columns  ->  DuckDB + Parquet + manifest

Offline: no network. Reads the newest ``1-mining/data/raw/hdv_co2_*.json``
(or ``--input``), rejects rows that fail plausibility bounds *loudly* (grouped
reason report), adds ``brand`` / ``oem_group`` / ``powertrain_class``, prints a
per-field coverage table, and writes ``manifest.json``.

Usage::

    uv run python 2-pipeline/reclean.py
    uv run python 2-pipeline/reclean.py --input 1-mining/data/raw/hdv_co2_2026-08-31.json -v
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from powerbench.dataio import (  # noqa: E402
    latest_raw_snapshot,
    provenance_line,
    write_hdv,
    write_manifest,
)
from powerbench.oem import canonical_oem, powertrain_class  # noqa: E402
from powerbench.paths import ensure_dirs  # noqa: E402
from powerbench.schema import clean_hdv_record  # noqa: E402

log = logging.getLogger("powerbench.pipeline.reclean")

_REASON_HEAD = 90  # chars of a rejection message used as its group key


def validate_rows(raw_rows: list[dict]) -> tuple[list[dict], collections.Counter]:
    """Return ``(clean_rows, reject_reason_counts)``."""
    clean: list[dict] = []
    reasons: collections.Counter = collections.Counter()
    for row in raw_rows:
        rec, err = clean_hdv_record(row)
        if rec is None:
            reasons[(err or "unknown")[:_REASON_HEAD]] += 1
            continue
        clean.append(rec)
    return clean, reasons


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``brand``, ``oem_group`` and ``powertrain_class``."""
    pairs = df["Manufacturer"].map(canonical_oem)
    df = df.assign(
        brand=pairs.map(lambda p: p[0]),
        oem_group=pairs.map(lambda p: p[1]),
        powertrain_class=df.apply(
            lambda r: powertrain_class(
                zero_emission=r.get("ZeroEmissionVehicle"),
                hybrid=r.get("HybridElectricHDV"),
                dual_fuel=r.get("DualFuelVehicle"),
                fuel_type=r.get("Engine_FuelType"),
            ),
            axis=1,
        ),
    )
    return df


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    cov = (
        pd.DataFrame(
            {
                "non_null": df.notna().sum(),
                "pct": (df.notna().sum() / n * 100).round(1),
                "n_unique": df.nunique(dropna=True),
            }
        )
        .sort_values("pct", ascending=False)
    )
    return cov


def domain_summary(df: pd.DataFrame) -> str:
    lines: list[str] = []
    if "MS_Year" in df:
        lines.append("rows/year:      " + df["MS_Year"].value_counts(dropna=False).sort_index().to_dict().__repr__())
    if "brand" in df:
        lines.append("rows/brand:     " + df["brand"].value_counts().to_dict().__repr__())
    if "powertrain_class" in df:
        lines.append("powertrain mix: " + df["powertrain_class"].value_counts().to_dict().__repr__())
    if "MS_SpecificCO2Emissions" in df:
        s = pd.to_numeric(df["MS_SpecificCO2Emissions"], errors="coerce")
        nonnull = s.notna().sum()
        zeros = (s == 0).sum()
        pos = (s > 0).sum()
        lines.append(
            f"MS_SpecificCO2Emissions: {nonnull} non-null, {zeros} == 0, {pos} > 0"
        )
    if "COL_CO2_gtkm" in df:
        s = pd.to_numeric(df["COL_CO2_gtkm"], errors="coerce")
        lines.append(f"COL_CO2_gtkm: {s.notna().sum()} non-null ({s.notna().mean()*100:.1f}%)")
    return "\n".join(lines)


def run(input_path: Path) -> int:
    ensure_dirs()
    envelope = json.loads(input_path.read_text(encoding="utf-8"))
    raw_rows = envelope.get("rows", [])
    log.info("snapshot %s: %d raw rows", input_path.name, len(raw_rows))

    clean, reasons = validate_rows(raw_rows)
    rejected = len(raw_rows) - len(clean)
    log.info("validated: %d clean, %d rejected", len(clean), rejected)
    if reasons:
        log.warning("rejection reasons (top 10):")
        for reason, count in reasons.most_common(10):
            log.warning("  %6d x  %s", count, reason)
    if not clean:
        log.error("no rows survived validation — check schema bounds vs snapshot")
        return 1

    df = add_derived(pd.DataFrame(clean))
    df = df.drop_duplicates(subset=["MS_PK_Vehicle"]).reset_index(drop=True)
    log.info("after dedup on MS_PK_Vehicle: %d rows, %d columns", len(df), df.shape[1])

    print("\n=== per-field coverage ===")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(coverage_report(df))
    print("\n=== domain summary ===")
    print(domain_summary(df))

    parquet = write_hdv(df)
    manifest = write_manifest(
        parquet_path=parquet,
        rows=len(df),
        source_snapshot=input_path,
        rejects=rejected,
    )
    log.info("wrote DuckDB table 'hdv' + %s", parquet.name)
    log.info("manifest: %s", json.dumps(manifest["datasets"]["hdv"], indent=2))
    print("\nprovenance:", provenance_line())
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", type=Path, help="raw snapshot JSON (default: newest in 1-mining/data/raw)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    path = args.input or latest_raw_snapshot()
    if not path or not path.exists():
        log.error("no raw snapshot found — run 1-mining/fetch_eea_hdv.py first")
        return 1
    return run(path)


if __name__ == "__main__":
    raise SystemExit(main())
