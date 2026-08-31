"""build_stages() from the Streamlit pipeline tab — pure, no Streamlit runtime."""

import importlib.util
from pathlib import Path

_APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"


def _load_build_stages():
    # exec just the function body without importing streamlit widgets:
    src = _APP.read_text(encoding="utf-8")
    start = src.index("def build_stages(")
    end = src.index("\ndef execute(")
    ns: dict = {"YEARS_VEHICLE": (2019, 2020), "YEARS_VIEWER": (2023,),
               "FETCH_VEHICLE": "F_V", "FETCH_VIEWER": "F_VW", "RECLEAN": "RC", "TRAIN_ML": "ML"}
    exec(compile(src[start:end], str(_APP), "exec"), ns)
    return ns["build_stages"]


build_stages = _load_build_stages()


def test_reload_only_no_fetch_with_train():
    st = build_stages([2019, 2020, 2023], mine=False, train=True)
    assert [s[1] for s in st] == [["RC"], ["ML"]]


def test_reload_only_no_train():
    st = build_stages([2019, 2020, 2023], mine=False, train=False)
    assert [s[1] for s in st] == [["RC"]]


def test_full_all_years():
    st = build_stages([2019, 2020, 2023], mine=True, train=True)
    labels = [s[0] for s in st]
    assert any("CO2_HeavyDutyVehicles" in x for x in labels)
    assert any("viewer" in x for x in labels)
    assert st[0][1] == ["F_V", "--years", "2019", "2020"]
    assert [s[1] for s in st][-2:] == [["RC"], ["ML"]]


def test_mine_only_2023():
    st = build_stages([2023], mine=True, train=False)
    assert [s[1] for s in st] == [["F_VW", "--years", "2023"], ["RC"]]


def test_mine_only_2019_with_train():
    st = build_stages([2019], mine=True, train=True)
    assert st[0][1] == ["F_V", "--years", "2019"]
    assert [s[1] for s in st] == [["F_V", "--years", "2019"], ["RC"], ["ML"]]


def test_extra_year_routes_through_viewer():
    st = build_stages([2023], mine=True, train=False, extra_years=[2024])
    vw = [x for x in st if "HDV_<year>_viewer" in x[0]]
    assert vw and vw[0][1] == ["F_VW", "--years", "2023", "2024"]


def test_extra_year_only_no_base():
    st = build_stages([], mine=True, train=False, extra_years=[2024])
    assert [x[1] for x in st] == [["F_VW", "--years", "2024"], ["RC"]]
