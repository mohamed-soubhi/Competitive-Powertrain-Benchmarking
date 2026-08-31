#!/usr/bin/env python3
"""Mine the EEA HDV *viewer* tables (one per reporting year) from Discodata.

Adds later reporting years than ``CO2_HeavyDutyVehicles`` (2019-2020). The
viewer is thinner (no engine ratings / WHTC-WHSC) but pre-joined and carries a
clean ``CO2v`` plus registration country. The year is the table name, not a
column, so one request set per year.

    uv run python 1-mining/fetch_eea_hdv_viewer.py
    uv run python 1-mining/fetch_eea_hdv_viewer.py --years 2023 --dry-run -v
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from powerbench.config import load_yaml  # noqa: E402
from powerbench.discodata import (  # noqa: E402
    DiscodataClient,
    NonRetryableDiscodataError,
    build_select,
    qualified_table,
)
from powerbench.paths import RAW_DIR, ensure_dirs  # noqa: E402

log = logging.getLogger("powerbench.mining.eea_hdv_viewer")
RAW_PREFIX = "hdv_viewer"


def _like(column: str, pattern: str) -> str:
    safe = pattern.replace("'", "''").lower()
    return f"LOWER([{column}]) LIKE N'%{safe}%'"


def build_chunk_queries(source: dict[str, Any]) -> list[dict[str, Any]]:
    db = source["access"]["database"]
    schema = source["access"]["schema"]
    pattern_tpl = source["access"]["table_pattern"]
    mfr_col = source["manufacturer_column"]
    cols = source["columns"]

    chunks: list[dict[str, Any]] = []
    for year in source["years"]:
        table = qualified_table(db, schema, pattern_tpl.format(year=year))
        for pat in source["manufacturer_patterns"]:
            chunks.append(
                {
                    "year": int(year),
                    "pattern": pat,
                    "sql": build_select(table, cols, where=[_like(mfr_col, pat)]),
                }
            )
    return chunks


def fetch_all(
    client: DiscodataClient, source: dict[str, Any], *, dry_run: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warn_at = int(source.get("max_rows_per_chunk_warn", 150_000))
    seen: set[Any] = set()
    rows: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []

    for chunk in build_chunk_queries(source):
        tag = f"{chunk['year']} / {chunk['pattern']}"
        if dry_run:
            log.info("[dry-run] %s\n    %s", tag, chunk["sql"])
            stats.append({"year": chunk["year"], "pattern": chunk["pattern"], "rows": None})
            continue
        try:
            got = client.run_sql(chunk["sql"])
        except NonRetryableDiscodataError as exc:
            log.error("chunk %s failed permanently: %s", tag, exc)
            stats.append({"year": chunk["year"], "pattern": chunk["pattern"], "rows": 0, "error": str(exc)})
            continue

        n_new = 0
        for r in got:
            key = (chunk["year"], r.get("vehicle_id") or id(r))
            if key in seen:
                continue
            seen.add(key)
            r["_year"] = chunk["year"]  # carry the table's year into the row
            rows.append(r)
            n_new += 1
        flag = "  <-- near cap" if len(got) >= warn_at else ""
        log.info("chunk %-26s %7d rows (%6d new)%s", tag, len(got), n_new, flag)
        stats.append({"year": chunk["year"], "pattern": chunk["pattern"], "rows": len(got), "new": n_new})

    return rows, stats


def write_snapshot(rows: list[dict[str, Any]], source: dict[str, Any], stats: list[dict[str, Any]]) -> Path:
    ensure_dirs()
    stamp = datetime.now(timezone.utc)
    out = RAW_DIR / f"{RAW_PREFIX}_{stamp.strftime('%Y-%m-%d')}.json"
    envelope = {
        "source": "eea_hdv_viewer",
        "access": source["access"],
        "fetched_at": stamp.isoformat(timespec="seconds"),
        "years": source["years"],
        "row_count": len(rows),
        "columns": source["columns"],
        "chunks": stats,
        "rows": rows,
    }
    payload = json.dumps(envelope, ensure_ascii=False, indent=2).encode("utf-8")
    out.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()[:16]
    prov = f"{out.name} | sha256:{sha} | {len(rows)} rows | years {source['years']} | fetched {envelope['fetched_at']}\n"
    (RAW_DIR / f"{RAW_PREFIX}_{stamp.strftime('%Y-%m-%d')}.prov.txt").write_text(prov, encoding="utf-8")
    log.info("wrote %s (%d rows)", out, len(rows))
    log.info("provenance: %s", prov.strip())
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", help="override sources.yaml years")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    source = dict(load_yaml("sources")["eea_hdv_viewer"])
    if args.years:
        source["years"] = args.years

    client = DiscodataClient(timeout=args.timeout)
    rows, stats = fetch_all(client, source, dry_run=args.dry_run)
    if args.dry_run:
        log.info("dry run: %d chunk queries built", len(stats))
        return 0
    if not rows:
        log.error("no rows fetched — check table pattern / year availability")
        return 1
    write_snapshot(rows, source, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
