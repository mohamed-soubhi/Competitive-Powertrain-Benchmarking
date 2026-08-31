"""Feature sets, leakage guard, and the modelling-matrix builder for CO2v.

Target: **CO2v** — VECTO declared specific CO2 (g/km), the whole-vehicle number.

Two feature sets:

* ``CO2V_FEATURES_BASE`` — fields present in every reporting year (mass, vehicle
  class, powertrain, fuel, flags). Works on 2019 / 2020 / 2023.
* ``CO2V_FEATURES_RICH`` — adds the engine ratings + axle config that only the
  2019–2020 source carries. Higher ceiling, narrower coverage.

Leakage: anything that is itself a CO2 / fuel-consumption measurement is banned
as a feature for CO2v — the engine-cycle CO2 (WHTC/WHSC g/kWh) feeds the VECTO
calculation that produces CO2v, so using it would be predicting CO2 from CO2.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

CO2V_TARGET = "CO2v"

CATEGORICAL = ["VehicleGroup", "powertrain_class", "Engine_FuelType", "AxleConfiguration"]
BOOL_FEATURES = ["ZeroEmissionVehicle", "HybridElectricHDV", "DualFuelVehicle", "VocationalVehicle"]

CO2V_FEATURES_BASE = [
    "GrossVehicleMass_t",
    "CurbMassChassis_kg",
    "VehicleGroup",
    "powertrain_class",
    "Engine_FuelType",
    *BOOL_FEATURES,
]

CO2V_FEATURES_RICH = [
    *CO2V_FEATURES_BASE,
    "Engine_RatedPower_kw",
    "Engine_Displacement_ltr",
    "Engine_RatedSpeed_rpm",
    "AxleConfiguration",
]

# columns that must never be a feature for the given target (target leakage)
LEAKY: dict[str, set[str]] = {
    "CO2v": {
        "WHTC_CO2_gkwh", "WHSC_CO2_gkwh",
        "COL_CO2_gtkm", "COL_CO2_gkm", "COL_FuelConsumption_l100km",
        "MS_SpecificCO2Emissions", "CO2v",
    },
}

# vehicle groups that carry a comparable VECTO CO2v
CO2_VEHICLE_GROUPS = (4, 5, 9, 10)


def assert_no_leakage(features: Sequence[str], target: str) -> None:
    bad = set(features) & LEAKY.get(target, set())
    if target in features:
        bad.add(target)
    if bad:
        raise ValueError(f"target leakage: {sorted(bad)} must not be features for {target!r}")


def build_xy(
    df: pd.DataFrame,
    features: Sequence[str],
    target: str = CO2V_TARGET,
    *,
    vehicle_groups: Sequence[int] | None = CO2_VEHICLE_GROUPS,
    min_rows: int = 200,
) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot modelling matrix from a cleaned HDV frame.

    - rejects known target leakage
    - restricts to ``vehicle_groups`` (comparable CO2v) unless ``None``
    - drops rows with a missing target
    - drops feature columns that are entirely empty (loudly)
    - one-hot encodes the categoricals (with an explicit NaN level)
    - median-fills the remaining numeric gaps
    """
    assert_no_leakage(features, target)
    if target not in df.columns:
        raise KeyError(f"target {target!r} not in dataframe")

    d = df
    if vehicle_groups is not None:
        d = d[d["VehicleGroup"].isin(list(vehicle_groups))]
    d = d.dropna(subset=[target])
    if len(d) < min_rows:
        raise ValueError(f"only {len(d)} rows with target {target!r}; need >= {min_rows}")

    present = [f for f in features if f in d.columns]
    empty = [c for c in present if d[c].notna().sum() == 0]
    if empty:
        print(f"  build_xy: dropping never-captured feature(s) {empty}")
    used = [c for c in present if c not in empty]
    if not used:
        raise ValueError("no usable feature columns after dropping empty ones")

    frame = d[used].copy()
    for c in BOOL_FEATURES:
        if c in frame.columns:
            frame[c] = frame[c].astype("float64")

    cats = [c for c in CATEGORICAL if c in used]
    X = pd.get_dummies(frame, columns=cats, dummy_na=True)
    X = X.astype("float64")
    X = X.fillna(X.median(numeric_only=True))
    y = d[target].astype("float64")
    return X.reset_index(drop=True), y.reset_index(drop=True)
