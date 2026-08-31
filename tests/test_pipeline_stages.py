"""build_stages() from the Streamlit pipeline tab — pure, no Streamlit runtime."""

import importlib.util
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def _load_build_stages():
    # exec just the function body without importing streamlit widgets:
    src = _APP.read_text()
    start = src.index("def build_stages(")
    end = src.index("\ndef execute(")
    ns: dict = {"YEARS_VEHICLE": (2019, 2020), "YEARS_VIEWER": (2023,),
               "FETCH_VEHICLE": "F_V", "FETCH_VIEWER": "F_VW", "RECLEAN": "RC"}
    exec(compile(src[start:end], str(_APP), "exec"), ns)
    return ns["build_stages"]


build_stages = _load_build_stages()


def test_reclean_only_no_fetch():
    st = build_stages([2019, 2020, 2023], mine=False)
    assert len(st) == 1
    assert st[0][1] == ["RC"]


def test_mine_all_years():
    st = build_stages([2019, 2020, 2023], mine=True)
    labels = [s[0] for s in st]
    assert any("CO2_HeavyDutyVehicles" in x for x in labels)
    assert any("HDV_2023_viewer" in x for x in labels)
    assert st[-1][1] == ["RC"]
    assert st[0][1] == ["F_V", "--years", "2019", "2020"]


def test_mine_only_2023():
    st = build_stages([2023], mine=True)
    assert [s[1] for s in st] == [["F_VW", "--years", "2023"], ["RC"]]


def test_mine_only_2019():
    st = build_stages([2019], mine=True)
    assert st[0][1] == ["F_V", "--years", "2019"]
    assert len(st) == 2
