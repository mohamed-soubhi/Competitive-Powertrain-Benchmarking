"""Feature sets for the LUZA powertrain models (FIX_PLAN.md S7).

Single source of truth for which columns feed which model, with the leaky and
the never-captured columns called out explicitly.

Why this file exists
--------------------
``ml_prediction.py`` built its "range model" like this::

    df["est_range_km"] = df["battery_useable_kwh"] * 1000 / df["efficiency_wh_km"]
    features = [..., "efficiency_wh_km", ...]     # <-- target is an exact
    rf.fit(df[features], df["est_range_km"])      #     function of a feature

so R2 was ~1.0 by construction and told us nothing. There is also **no measured
range column** in the dataset (``range_real_km`` / ``range_wltp_km`` are never
captured -> 0 non-null).

So the "range model" is retargeted: predict **efficiency (Wh/km)** from physical
descriptors. That is a real regression with a real target.

``weight_kg`` is likewise **never captured** (0/30 non-null). It is left out of
every feature set rather than imputed with a constant, which is what the old
``fillna(2000)`` did.
"""

from __future__ import annotations

import pandas as pd

# --- retargeted "range" model: predict efficiency from physical descriptors ---
EFFICIENCY_FEATURES = [
    "battery_useable_kwh",
    "power_kw",
    "torque_nm",
    "top_speed_kmh",
    "architecture_v",
]
EFFICIENCY_TARGET = "efficiency_wh_km"

ACCEL_FEATURES = ["power_kw", "torque_nm", "battery_useable_kwh", "top_speed_kmh"]
ACCEL_TARGET = "accel_0_100_s"

CHARGING_FEATURES = ["battery_useable_kwh", "power_kw", "architecture_v"]
CHARGING_TARGET = "charging_dc_kw"

# columns that must never feed a model with the given target (target leakage)
LEAKY: dict[str, set[str]] = {
    # efficiency is one factor of est_range_km; est_range is one factor of efficiency
    "efficiency_wh_km": {"est_range_km", "range_real_km", "range_wltp_km", "range_rated_km"},
    "est_range_km": {"efficiency_wh_km", "battery_useable_kwh", "battery_nominal_kwh"},
}

# never populated in the current dataset — using them means imputing a constant
NEVER_CAPTURED = {"weight_kg", "range_real_km", "range_wltp_km", "range_rated_km", "price"}


def binary_label_by_median(
    df: pd.DataFrame, column: str, below_is_positive: bool = True
) -> tuple[pd.Series, float]:
    """Split ``column`` at its own median instead of a hard-coded constant.

    The old classifier used ``efficiency_wh_km < 140`` — an arbitrary cut that
    happened to put ~2/3 of the fleet on one side. Splitting at the median gives
    a defined, roughly 50/50 target. Returns ``(labels, threshold)`` with rows
    missing ``column`` dropped. ``below_is_positive`` -> label 1 means "more
    efficient" (lower Wh/km).
    """
    s = pd.to_numeric(df[column], errors="coerce")
    thr = float(s.median())
    on_side = s < thr if below_is_positive else s > thr
    labels = on_side[s.notna()].astype(int)
    labels.name = f"{column}_below_median" if below_is_positive else f"{column}_above_median"
    return labels, thr


def assert_no_leakage(features: list[str], target: str) -> None:
    """Raise ``ValueError`` if any feature is a known leak for ``target``."""
    bad = set(features) & LEAKY.get(target, set())
    if target in features:
        bad.add(target)
    if bad:
        raise ValueError(f"target leakage: {sorted(bad)} must not be features for {target!r}")


def build_xy(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    min_rows: int = 8,
) -> tuple[pd.DataFrame, pd.Series]:
    """Assemble a modelling matrix from a cleaned EV frame.

    - rejects known target leakage up front
    - drops feature columns that are entirely empty (never-captured), loudly
    - drops rows with a missing target; median-fills remaining feature gaps
    """
    assert_no_leakage(features, target)
    if target not in df.columns:
        raise KeyError(f"target {target!r} not in dataframe")

    present = [f for f in features if f in df.columns]
    num = df[present + [target]].apply(pd.to_numeric, errors="coerce")

    empty = [c for c in present if num[c].notna().sum() == 0]
    if empty:
        print(f"  build_xy: dropping never-captured feature(s) {empty}")
    used = [c for c in present if c not in empty]
    if not used:
        raise ValueError("no usable feature columns after dropping empty ones")

    num = num.dropna(subset=[target])
    if len(num) < min_rows:
        raise ValueError(f"only {len(num)} rows with target {target!r}; need >= {min_rows}")

    X = num[used].fillna(num[used].median())
    y = num[target]
    return X, y
