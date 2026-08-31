"""Offline tests for the viewer miner's per-year availability preflight."""

import importlib.util
from pathlib import Path

from powerbench.discodata import NonRetryableDiscodataError

_SPEC = importlib.util.spec_from_file_location(
    "fev", Path(__file__).resolve().parent.parent / "1-mining" / "fetch_eea_hdv_viewer.py"
)
fev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fev)

_SOURCE = {
    "access": {"database": "CO2Emission", "schema": "latest", "table_pattern": "HDV_{year}_viewer"},
    "manufacturer_column": "OEM_ManufacturerName",
    "manufacturer_patterns": ["scania", "daf"],
    "columns": ["vehicle_id", "CO2v"],
    "years": [2023, 2026],
}


class _Client:
    """Stub: 2023 exists, everything else is an unknown object."""

    def __init__(self):
        self.calls = []

    def run_sql(self, sql, nr_of_hits=None):
        self.calls.append(sql)
        if "HDV_2023_viewer" in sql:
            if sql.startswith("SELECT TOP 1 1"):
                return [{"x": 1}]
            return [{"vehicle_id": "0xAB", "CO2v": 700.0}]
        raise NonRetryableDiscodataError(
            "Discodata error 10003: Invalid object name 'CO2Emission.latest.HDV_2026_viewer'."
        )


def test_year_available_true_false():
    c = _Client()
    assert fev.year_available(c, fev._table_for(_SOURCE, 2023)) is True
    assert fev.year_available(c, fev._table_for(_SOURCE, 2026)) is False


def test_fetch_all_skips_unpublished_year_keeps_published():
    c = _Client()
    rows, stats = fev.fetch_all(c, dict(_SOURCE))
    assert rows and all(r["_year"] == 2023 for r in rows)
    skipped = [s for s in stats if s.get("error") == "table not published"]
    assert [s["year"] for s in skipped] == [2026]
    # exactly one probe per year, not one error per manufacturer chunk
    probes = [s for s in c.calls if s.startswith("SELECT TOP 1 1")]
    assert len(probes) == 2


def test_fetch_all_all_unpublished_returns_empty():
    c = _Client()
    rows, stats = fev.fetch_all(c, {**_SOURCE, "years": [2026]})
    assert rows == []
    assert stats and stats[0]["error"] == "table not published"


def test_build_chunk_queries_uses_given_years():
    q = fev.build_chunk_queries(_SOURCE, [2023])
    assert {c["year"] for c in q} == {2023}
    assert len(q) == 2  # one per manufacturer pattern
