"""Offline tests for the pure SQL builder — no network."""

from powerbench.discodata import build_select, qualified_table


def test_qualified_table():
    assert qualified_table("CO2Emission", "latest", "CO2_HeavyDutyVehicles") == (
        "[CO2Emission].[latest].[CO2_HeavyDutyVehicles]"
    )


def test_build_select_minimal():
    sql = build_select("[T]", ["a", "b"])
    assert sql == "SELECT [a], [b] FROM [T]"


def test_build_select_predicates_and_quoting():
    sql = build_select(
        "[T]",
        ["a"],
        where=["[Manufacturer] IS NOT NULL"],
        equals={"MS_Year": 2019},
        in_lists={"VehicleGroup": [4, 5, 9]},
    )
    assert "WHERE [Manufacturer] IS NOT NULL" in sql
    assert "[MS_Year] = 2019" in sql
    assert "[VehicleGroup] IN (4, 5, 9)" in sql


def test_build_select_escapes_string_literal():
    sql = build_select("[T]", ["a"], equals={"name": "O'Brien"})
    assert "N'O''Brien'" in sql


def test_build_select_rejects_empty_columns():
    try:
        build_select("[T]", [])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_build_select_top_and_order():
    sql = build_select("[T]", ["a"], order_by=["a"], top=10)
    assert sql.startswith("SELECT TOP 10 [a] FROM [T] ORDER BY [a]")
