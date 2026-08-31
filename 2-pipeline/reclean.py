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
    all_raw_snapshots,
    provenance_line,
    write_hdv,
    write_manifest,
)
from powerbench.oem import canonical_oem, powertrain_class  # noqa: E402
from powerbench.paths import ensure_dirs  # noqa: E402
from powerbench.schema import clean_hdv_record, row_key  # noqa: E402
from powerbench.viewer_map import map_viewer_row  # noqa: E402

log = logging.getLogger("powerbench.pipeline.reclean")

_REASON_HEAD = 90  # chars of a rejection message used as its group key


def _prepare(row: dict, source: str) -> dict:
    """Shape a raw row for the schema gate according to its source snapshot."""
    if source == "eea_hdv_viewer":
        return map_viewer_row(row, int(row.get("_year")))
    row.setdefault("source_table", "vehicle")
    return row


def validate_rows(
    raw_rows: list[dict], source: str
) -> tuple[list[dict], collections.Counter]:
    """Return ``(clean_rows, reject_reason_counts)`` for one snapshot's rows."""
    clean: list[dict] = []
    reasons: collections.Counter = collections.Counter()
    for row in raw_rows:
        rec, err = clean_hdv_record(_prepare(row, source))
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


def run(input_paths: list[Path]) -> int:
    ensure_dirs()
    all_clean: list[dict] = []
    total_raw = 0
    for path in input_paths:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        raw_rows = envelope.get("rows", [])
        source = envelope.get("source", "eea_hdv_co2")
        total_raw += len(raw_rows)
        log.info("snapshot %s (%s): %d raw rows", path.name, source, len(raw_rows))

        clean, reasons = validate_rows(raw_rows, source)
        log.info("  -> %d clean, %d rejected", len(clean), len(raw_rows) - len(clean))
        for reason, count in reasons.most_common(6):
            log.warning("  %6d x  %s", count, reason)
        all_clean.extend(clean)

    rejected = total_raw - len(all_clean)
    if not all_clean:
        log.error("no rows survived validation across %d snapshot(s)", len(input_paths))
        return 1

    # de-duplicate on (year, MS_PK_Vehicle | vehicle_id) across all snapshots
    seen: set = set()
    deduped: list[dict] = []
    for rec in all_clean:
        k = row_key(rec)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(rec)
    log.info("combined: %d clean, %d after dedup, %d rejected", len(all_clean), len(deduped), rejected)

    df = add_derived(pd.DataFrame(deduped))
    log.info("final frame: %d rows, %d columns", len(df), df.shape[1])

    print("\n=== per-field coverage ===")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(coverage_report(df))
    print("\n=== domain summary ===")
    print(domain_summary(df))

    parquet = write_hdv(df)
    manifest = write_manifest(
        parquet_path=parquet,
        rows=len(df),
        source_snapshots=input_paths,
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
    p.add_argument(
        "--input", type=Path, nargs="+",
        help="raw snapshot JSON file(s) (default: newest hdv_co2_* and hdv_viewer_*)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    paths = list(args.input) if args.input else all_raw_snapshots()
    paths = [p for p in paths if p.exists()]
    if not paths:
        log.error("no raw snapshot found — run 1-mining/fetch_eea_hdv*.py first")
        return 1
    return run(paths)


if __name__ == "__main__":
    raise SystemExit(main())
