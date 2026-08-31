#!/usr/bin/env python3
"""Mine EU heavy-duty-vehicle CO2 monitoring data from EEA Discodata.

Pulls the curated column set from ``[CO2Emission].[latest].[CO2_HeavyDutyVehicles]``
one ``(reporting year, manufacturer)`` chunk at a time (the endpoint's ``p``
pagination is unreliable, so we bound each request with a WHERE instead), then
writes a single raw JSON snapshot plus a provenance sidecar into
``1-mining/data/raw/``.

Downstream: ``reclean.py`` validates this snapshot with the pydantic gate and
loads it into the DuckDB store.

Usage::

    uv run python 1-mining/fetch_eea_hdv.py                 # all configured years
    uv run python 1-mining/fetch_eea_hdv.py --years 2019    # one year
    uv run python 1-mining/fetch_eea_hdv.py --dry-run -v    # print queries only
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

# allow "python 1-mining/fetch_eea_hdv.py" without an install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from powerbench.config import eea_hdv_source, hdv_columns  # noqa: E402
from powerbench.discodata import (  # noqa: E402
    DiscodataClient,
    NonRetryableDiscodataError,
    build_select,
    qualified_table,
)
from powerbench.paths import RAW_DIR, ensure_dirs  # noqa: E402

log = logging.getLogger("powerbench.mining.eea_hdv")

RAW_PREFIX = "hdv_co2"


def _like(column: str, pattern: str) -> str:
    """Case-insensitive LIKE fragment. ``pattern`` is trusted (from our YAML)."""
    safe = pattern.replace("'", "''")
    return f"LOWER([{column}]) LIKE N'%{safe.lower()}%'"


def build_chunk_queries(source: dict[str, Any]) -> list[dict[str, Any]]:
    """One query descriptor per (year, manufacturer pattern)."""
    table = qualified_table(
        source["access"]["database"],
        source["access"]["schema"],
        source["access"]["table"],
    )
    columns = hdv_columns(source)
    base_where: list[str] = []
    if not source.get("include_ms_reported", False):
        base_where.append("[Manufacturer] IS NOT NULL")

    chunks: list[dict[str, Any]] = []
    for year in source["years"]:
        for pattern in source["manufacturer_patterns"]:
            where = [*base_where, _like("Manufacturer", pattern), f"[MS_Year] = {int(year)}"]
            chunks.append(
                {
                    "year": int(year),
                    "pattern": pattern,
                    "sql": build_select(table, columns, where=where),
                }
            )
    return chunks


def fetch_all(
    client: DiscodataClient, source: dict[str, Any], *, dry_run: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return ``(rows, chunk_stats)``. Rows de-duplicated on ``MS_PK_Vehicle``."""
    warn_at = int(source.get("max_rows_per_chunk_warn", 150_000))
    seen: set[Any] = set()
    rows: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []

    for chunk in build_chunk_queries(source):
        tag = f"{chunk['year']} / {chunk['pattern']}"
        if dry_run:
            log.info("[dry-run] %s\n    %s", tag, chunk["sql"])
            stats.append({**_chunk_meta(chunk), "rows": None})
            continue
        try:
            got = client.run_sql(chunk["sql"])
        except NonRetryableDiscodataError as exc:
            log.error("chunk %s failed permanently: %s", tag, exc)
            stats.append({**_chunk_meta(chunk), "rows": 0, "error": str(exc)})
            continue

        n_new = 0
        for r in got:
            pk = r.get("MS_PK_Vehicle")
            key = pk if pk is not None else id(r)
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
            n_new += 1
        flag = "  <-- near cap, consider sub-splitting" if len(got) >= warn_at else ""
        log.info("chunk %-28s %7d rows (%6d new)%s", tag, len(got), n_new, flag)
        stats.append({**_chunk_meta(chunk), "rows": len(got), "new": n_new})

    return rows, stats


def _chunk_meta(chunk: dict[str, Any]) -> dict[str, Any]:
    return {"year": chunk["year"], "pattern": chunk["pattern"], "sql": chunk["sql"]}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_snapshot(
    rows: list[dict[str, Any]], source: dict[str, Any], stats: list[dict[str, Any]]
) -> Path:
    """Write the raw JSON envelope + a human-readable provenance sidecar."""
    ensure_dirs()
    stamp = datetime.now(timezone.utc)
    date_str = stamp.strftime("%Y-%m-%d")
    out = RAW_DIR / f"{RAW_PREFIX}_{date_str}.json"

    envelope = {
        "source": "eea_hdv_co2",
        "access": source["access"],
        "fetched_at": stamp.isoformat(timespec="seconds"),
        "years": source["years"],
        "row_count": len(rows),
        "columns": hdv_columns(source),
        "chunks": stats,
        "rows": rows,
    }
    payload = json.dumps(envelope, ensure_ascii=False, indent=2).encode("utf-8")
    out.write_bytes(payload)

    prov = (
        f"{out.name} | sha256:{_sha256_bytes(payload)[:16]} | {len(rows)} rows "
        f"| years {source['years']} | fetched {envelope['fetched_at']}\n"
    )
    (RAW_DIR / f"{RAW_PREFIX}_{date_str}.prov.txt").write_text(prov, encoding="utf-8")
    log.info("wrote %s (%d rows)", out, len(rows))
    log.info("provenance: %s", prov.strip())
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", type=int, nargs="+", help="override the years in sources.yaml")
    p.add_argument("--dry-run", action="store_true", help="print the SQL for each chunk, fetch nothing")
    p.add_argument("--timeout", type=int, default=120, help="per-request timeout seconds")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging (prints every query)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    source = dict(eea_hdv_source())
    if args.years:
        source["years"] = args.years

    client = DiscodataClient(timeout=args.timeout)
    rows, stats = fetch_all(client, source, dry_run=args.dry_run)
    if args.dry_run:
        log.info("dry run: %d chunk queries built, nothing fetched", len(stats))
        return 0
    if not rows:
        log.error("no rows fetched — check connectivity / table name / filters")
        return 1
    write_snapshot(rows, source, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
