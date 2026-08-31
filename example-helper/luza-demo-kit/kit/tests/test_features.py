"""Feature-set / leakage-guard contracts (S7)."""

import pytest
from sklearn.linear_model import LinearRegression

from luza.features import (
    EFFICIENCY_FEATURES,
    EFFICIENCY_TARGET,
    assert_no_leakage,
    binary_label_by_median,
    build_xy,
)
from luza.modeleval import evaluate_regressor


def _ev():
    from luza.dataio import load_ev_specs
    from luza.paths import CLEAN_DIR

    return load_ev_specs(CLEAN_DIR / "ev_database_2026-08-29_reclean.csv")


def test_efficiency_is_the_target_not_a_feature():
    assert EFFICIENCY_TARGET == "efficiency_wh_km"
    assert "efficiency_wh_km" not in EFFICIENCY_FEATURES
    assert not ({"range_real_km", "range_wltp_km", "weight_kg"} & set(EFFICIENCY_FEATURES))


def test_assert_no_leakage_flags_the_old_range_setup():
    leaky_feats = ["battery_useable_kwh", "power_kw", "efficiency_wh_km"]
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(leaky_feats, "est_range_km")
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage([*EFFICIENCY_FEATURES, "efficiency_wh_km"], "efficiency_wh_km")
    assert_no_leakage(EFFICIENCY_FEATURES, "efficiency_wh_km")  # clean -> no raise


def test_build_xy_drops_never_captured_weight_and_keeps_target_out():
    df = _ev().copy()
    df["all_nan_feature"] = float("nan")
    X, y = build_xy(df, [*EFFICIENCY_FEATURES, "all_nan_feature"], EFFICIENCY_TARGET)
    assert "all_nan_feature" not in X.columns    # 100% NaN feature is dropped
    assert "efficiency_wh_km" not in X.columns   # target never leaks in
    assert len(X) == len(y) >= 20
    assert X.notna().all().all()
    assert y.name == "efficiency_wh_km"


def test_honest_efficiency_model_is_not_the_old_fake_r2():
    X, y = build_xy(_ev(), EFFICIENCY_FEATURES, EFFICIENCY_TARGET)
    rep = evaluate_regressor(LinearRegression(), X, y, n_repeats=5)
    # old pipeline reported ~1.0; a real out-of-fold number is well under that
    assert rep.loo_r2 < 0.999
    assert rep.naive_r2 < 0.999


def test_median_split_replaces_the_hardcoded_140_threshold():
    labels, thr = binary_label_by_median(_ev(), "efficiency_wh_km")
    assert 120 < thr < 220                       # data-driven, not 140
    counts = labels.value_counts()
    assert set(counts.index) == {0, 1}
    assert abs(counts[0] - counts[1]) <= 2       # ~50/50 by construction
    assert labels.name == "efficiency_wh_km_below_median"
