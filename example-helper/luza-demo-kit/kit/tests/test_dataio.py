"""Data-loading contracts (S2)."""

from luza import dataio
from luza.paths import CLEAN_DIR


def test_latest_picks_newest_by_name():
    got = dataio.latest("ev_database_*.csv")
    assert got is not None
    assert got.name.startswith("ev_database_")
    assert got.parent == CLEAN_DIR


def test_sha256_is_stable():
    path = dataio.latest("ev_database_*.csv")
    assert dataio.sha256_file(path) == dataio.sha256_file(path)
    assert len(dataio.sha256_file(path)) == 64


def test_load_ev_specs_shapes_the_frame():
    df = dataio.load_ev_specs()
    assert len(df) >= 30
    assert "brand" in df.columns
    assert df["brand"].iloc[0] == "Tesla"
    # numeric coercion applied
    assert str(df["power_kw"].dtype).startswith(("float", "int"))


def test_load_patents_dedupes_on_id():
    both = sorted(CLEAN_DIR.glob("patents_greyb_*.csv"))
    df = dataio.load_patents(list(both))
    assert not df.empty
    assert df["patent_id"].is_unique


def test_provenance_line_stamps_file_hash_and_rows():
    line = dataio.provenance_line("ev_specs")
    assert "ev_database_" in line
    assert "sha256:" in line
    assert "rows" in line
    # unknown dataset degrades gracefully, never raises
    assert "no dataset file found" in dataio.provenance_line("does_not_exist")
