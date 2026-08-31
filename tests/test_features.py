"""Offline tests for the CO2v feature builder + leakage guard."""

import numpy as np
import pandas as pd
import pytest

from powerbench.features import (
    CO2V_FEATURES_BASE,
    CO2V_TARGET,
    LEAKY,
    assert_no_leakage,
    build_xy,
)


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 800
    return pd.DataFrame({
        "MS_Year": rng.choice([2019, 2020, 2023], n),
        "VehicleGroup": rng.choice([4, 5, 9, 10, 2], n),          # 2 must be filtered out
        "GrossVehicleMass_t": rng.uniform(10, 40, n),
        "CurbMassChassis_kg": rng.uniform(5000, 12000, n),
        "powertrain_class": rng.choice(["Diesel ICE", "Gas (CNG/LNG)"], n),
        "Engine_FuelType": rng.choice(["Diesel CI", None], n),
        "ZeroEmissionVehicle": rng.choice([0.0, 1.0], n),
        "HybridElectricHDV": 0.0,
        "DualFuelVehicle": 0.0,
        "VocationalVehicle": rng.choice([0.0, 1.0], n),
        "CO2v": rng.uniform(600, 900, n),
        "WHTC_CO2_gkwh": rng.uniform(600, 660, n),
    })


def test_leakage_raises_on_whtc():
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage([*CO2V_FEATURES_BASE, "WHTC_CO2_gkwh"], CO2V_TARGET)


def test_leaky_map_covers_engine_cycle_co2():
    assert {"WHTC_CO2_gkwh", "WHSC_CO2_gkwh"} <= LEAKY["CO2v"]


def test_build_xy_filters_groups_and_encodes(df):
    X, y = build_xy(df, CO2V_FEATURES_BASE, CO2V_TARGET, min_rows=50)
    assert len(X) == len(y)
    assert len(X) < len(df)                                   # group 2 rows dropped
    assert "WHTC_CO2_gkwh" not in X.columns                   # not a feature -> not leaked in
    assert any(c.startswith("powertrain_class_") for c in X.columns)  # one-hot happened
    assert X.notna().all().all()                              # median-filled
    assert np.isfinite(X.to_numpy(float)).all()


def test_build_xy_drops_all_empty_feature(df, capsys):
    df["Engine_FuelType"] = None
    X, _ = build_xy(df, CO2V_FEATURES_BASE, CO2V_TARGET, min_rows=50)
    assert "Engine_FuelType" not in "".join(X.columns)
    assert "dropping never-captured" in capsys.readouterr().out


def test_build_xy_raises_when_too_few_rows(df):
    with pytest.raises(ValueError, match="need >="):
        build_xy(df.head(3), CO2V_FEATURES_BASE, CO2V_TARGET, min_rows=200)
