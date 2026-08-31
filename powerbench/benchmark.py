"""Pure aggregation helpers for the benchmarking dashboard.

DataFrame in, DataFrame out. No plotting, no Streamlit, no I/O — so the numbers
behind every chart are unit-testable offline.

Vocabulary
----------
metric      one CO2 / efficiency column, e.g. ``"CO2v"`` (VECTO declared g/km),
            ``"WHTC_CO2_gkwh"`` (engine-cycle g/kWh), ``"COL_CO2_gtkm"``.
dimension   one of ``brand`` / ``oem_group`` / ``powertrain_class`` /
            ``VehicleGroup`` / ``MS_Year``.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

# VehicleGroups that carry VECTO-simulated CO2 (long-haul / regional rigid +
# tractor). Buses / vocational / small groups report no CO2v and are excluded
# from CO2 benchmarking by default.
CO2_VEHICLE_GROUPS: tuple[int, ...] = (4, 5, 9, 10)

METRIC_LABELS: dict[str, str] = {
    "CO2v": "VECTO declared CO2 (g/km)",
    "WHTC_CO2_gkwh": "Engine WHTC CO2 (g/kWh)",
    "WHSC_CO2_gkwh": "Engine WHSC CO2 (g/kWh)",
    "COL_CO2_gtkm": "Long-haul CO2 (g/t-km)",
    "COL_FuelConsumption_l100km": "Long-haul fuel use (L/100km)",
    "MS_SpecificCO2Emissions": "MS-reported specific CO2",
}


def filter_frame(
    df: pd.DataFrame,
    *,
    years: Sequence[int] | None = None,
    brands: Sequence[str] | None = None,
    powertrains: Sequence[str] | None = None,
    vehicle_groups: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Row filter. ``None`` / empty for a dimension means 'all'."""
    mask = pd.Series(True, index=df.index)
    if years:
        mask &= df["MS_Year"].isin(list(years))
    if brands:
        mask &= df["brand"].isin(list(brands))
    if powertrains:
        mask &= df["powertrain_class"].isin(list(powertrains))
    if vehicle_groups:
        mask &= df["VehicleGroup"].isin(list(vehicle_groups))
    return df.loc[mask].copy()


def metric_by_dimension(
    df: pd.DataFrame,
    metric: str,
    dimension: str,
    *,
    min_count: int = 1,
) -> pd.DataFrame:
    """``mean`` / ``median`` / ``std`` / ``n`` of ``metric`` per ``dimension`` value.

    Rows with a null metric are ignored; groups with fewer than ``min_count``
    non-null values are dropped (thin OEM tails are noise).
    """
    sub = df[[dimension, metric]].dropna(subset=[metric])
    g = sub.groupby(dimension)[metric]
    out = g.agg(n="count", mean="mean", median="median", std="std").reset_index()
    out = out[out["n"] >= min_count].sort_values("mean").reset_index(drop=True)
    return out


def co2_trend(
    df: pd.DataFrame,
    metric: str = "CO2v",
    dimension: str = "brand",
    *,
    min_count: int = 30,
) -> pd.DataFrame:
    """Year x dimension mean of ``metric`` (long form: year, <dimension>, mean, n)."""
    sub = df[["MS_Year", dimension, metric]].dropna(subset=[metric])
    out = (
        sub.groupby(["MS_Year", dimension])[metric]
        .agg(mean="mean", n="count")
        .reset_index()
    )
    return out[out["n"] >= min_count].reset_index(drop=True)


def powertrain_mix(df: pd.DataFrame, dimension: str = "brand") -> pd.DataFrame:
    """Share of each powertrain_class within every ``dimension`` value (wide)."""
    ct = pd.crosstab(df[dimension], df["powertrain_class"], normalize="index")
    return ct.reset_index()


def numeric_correlations(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Pearson r matrix over ``columns`` that have >2 non-null values."""
    present = [c for c in columns if c in df.columns and df[c].notna().sum() > 2]
    return df[present].apply(pd.to_numeric, errors="coerce").corr()


def improvement_ranking(
    df: pd.DataFrame,
    metric: str = "CO2v",
    dimension: str = "brand",
    *,
    first_year: int,
    last_year: int,
    min_count: int = 30,
) -> pd.DataFrame:
    """Per-dimension change in mean ``metric`` from ``first_year`` to ``last_year``.

    Negative ``delta`` = CO2 went down = improvement.
    """
    tr = co2_trend(df, metric, dimension, min_count=min_count)
    wide = tr.pivot_table(index=dimension, columns="MS_Year", values="mean")
    if first_year not in wide or last_year not in wide:
        return pd.DataFrame(columns=[dimension, "first", "last", "delta", "pct"])
    out = pd.DataFrame(
        {
            dimension: wide.index,
            "first": wide[first_year].to_numpy(),
            "last": wide[last_year].to_numpy(),
        }
    )
    out["delta"] = out["last"] - out["first"]
    out["pct"] = out["delta"] / out["first"] * 100
    return out.dropna(subset=["delta"]).sort_values("delta").reset_index(drop=True)
