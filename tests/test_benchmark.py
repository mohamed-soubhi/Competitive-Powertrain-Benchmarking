"""Offline tests for the pure aggregation helpers."""

import numpy as np
import pandas as pd
import pytest

from powerbench.benchmark import (
    co2_trend,
    filter_frame,
    improvement_ranking,
    metric_by_dimension,
    numeric_correlations,
    powertrain_mix,
)


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 400
    brand = rng.choice(["Scania", "MAN", "DAF"], n)
    year = rng.choice([2019, 2020], n)
    base = {"Scania": 740.0, "MAN": 780.0, "DAF": 800.0}
    co2 = np.array([base[b] for b in brand]) + (year - 2019) * -10 + rng.normal(0, 5, n)
    return pd.DataFrame(
        {
            "MS_Year": year,
            "brand": brand,
            "powertrain_class": rng.choice(["Diesel ICE", "Gas (CNG/LNG)"], n, p=[0.9, 0.1]),
            "VehicleGroup": rng.choice([5, 9], n),
            "CO2v": co2,
            "Engine_RatedPower_kw": rng.normal(330, 40, n),
        }
    )


def test_filter_frame_none_is_all(df):
    assert len(filter_frame(df)) == len(df)


def test_filter_frame_combines_dimensions(df):
    out = filter_frame(df, years=[2019], brands=["Scania"])
    assert set(out["MS_Year"]) == {2019}
    assert set(out["brand"]) == {"Scania"}


def test_metric_by_dimension_sorted_ascending(df):
    out = metric_by_dimension(df, "CO2v", "brand")
    assert list(out["brand"]) == sorted(out["brand"], key=lambda b: out.set_index("brand").loc[b, "mean"])
    assert out.iloc[0]["brand"] == "Scania"  # lowest CO2 by construction


def test_metric_by_dimension_min_count_drops_thin_groups(df):
    out = metric_by_dimension(df, "CO2v", "brand", min_count=10_000)
    assert out.empty


def test_co2_trend_long_form(df):
    tr = co2_trend(df, "CO2v", "brand", min_count=1)
    assert set(tr.columns) == {"MS_Year", "brand", "mean", "n"}
    assert len(tr) == 6  # 3 brands x 2 years


def test_co2_trend_ignores_nulls(df):
    df.loc[df.index[:50], "CO2v"] = np.nan
    tr = co2_trend(df, "CO2v", "brand", min_count=1)
    assert tr["n"].sum() == df["CO2v"].notna().sum()


def test_powertrain_mix_rows_sum_to_one(df):
    mix = powertrain_mix(df, "brand")
    share_cols = [c for c in mix.columns if c != "brand"]
    assert np.allclose(mix[share_cols].sum(axis=1), 1.0)


def test_numeric_correlations_square_and_diag_one(df):
    corr = numeric_correlations(df, ["CO2v", "Engine_RatedPower_kw"])
    assert corr.shape == (2, 2)
    assert np.allclose(np.diag(corr), 1.0)


def test_improvement_ranking_detects_decline(df):
    rank = improvement_ranking(df, "CO2v", "brand", first_year=2019, last_year=2020, min_count=1)
    assert (rank["delta"] < 0).all()  # all brands improved by construction
    assert list(rank.columns) == ["brand", "first", "last", "delta", "pct"]


def test_improvement_ranking_missing_year_returns_empty(df):
    rank = improvement_ranking(df, "CO2v", "brand", first_year=2019, last_year=2023, min_count=1)
    assert rank.empty
