"""Hypothetical-predictor honesty guards (S10)."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from luza.predict import (
    HypotheticalPrediction,
    bootstrap_prediction_interval,
    flag_out_of_envelope,
    predict_hypotheticals,
    training_envelope,
)


def _train(n=60, seed=0):
    rng = np.random.default_rng(seed)
    power = rng.uniform(100, 300, n)
    battery = rng.uniform(40, 90, n)
    X = pd.DataFrame({"power_kw": power, "battery_useable_kwh": battery})
    y = pd.Series(0.5 * power + 1.2 * battery + rng.normal(0, 3, n), name="target")
    return X, y


def test_envelope_captures_training_min_max():
    X, _ = _train()
    env = training_envelope(X)
    assert env["power_kw"]["min"] == X["power_kw"].min()
    assert env["power_kw"]["max"] == X["power_kw"].max()
    assert env["power_kw"]["p5"] < env["power_kw"]["p95"]


def test_flag_out_of_envelope_catches_extrapolation():
    X, _ = _train()
    env = training_envelope(X)
    assert flag_out_of_envelope({"power_kw": 200, "battery_useable_kwh": 65}, env) == []
    assert flag_out_of_envelope({"power_kw": 900, "battery_useable_kwh": 65}, env) == ["power_kw"]
    assert flag_out_of_envelope({"power_kw": 900, "battery_useable_kwh": 5}, env) == [
        "battery_useable_kwh",
        "power_kw",
    ]


def test_bootstrap_interval_brackets_point_and_is_reproducible():
    X, y = _train()
    x_new = {"power_kw": 200, "battery_useable_kwh": 65}
    p1, lo1, hi1 = bootstrap_prediction_interval(LinearRegression(), X, y, x_new, n_boot=200)
    p2, lo2, hi2 = bootstrap_prediction_interval(LinearRegression(), X, y, x_new, n_boot=200)
    assert (p1, lo1, hi1) == (p2, lo2, hi2)          # fixed random_state
    assert lo1 < p1 < hi1


def test_interval_is_positive_width_and_finite_far_outside_envelope():
    # RF clamps outside its training range, so interval WIDTH is not a reliable
    # extrapolation signal (that is what flag_out_of_envelope is for). What must
    # hold: the interval is still finite, ordered, and non-degenerate.
    X, y = _train()
    rf = RandomForestRegressor(n_estimators=40, random_state=0)
    p, lo, hi = bootstrap_prediction_interval(
        rf, X, y, {"power_kw": 2000, "battery_useable_kwh": 400}, n_boot=150
    )
    assert np.isfinite([p, lo, hi]).all()
    assert lo <= p <= hi and hi > lo


def test_predict_hypotheticals_labels_illustrative_and_flags():
    X, y = _train()
    scenarios = [
        {"power_kw": 180, "battery_useable_kwh": 60},
        {"power_kw": 500, "battery_useable_kwh": 120},   # both out of envelope
    ]
    res = predict_hypotheticals(LinearRegression(), X, y, scenarios, n_boot=100)
    assert all(isinstance(r, HypotheticalPrediction) and r.illustrative for r in res)
    assert res[0].out_of_envelope == []
    assert res[1].out_of_envelope == ["battery_useable_kwh", "power_kw"]
    d = res[0].as_dict()
    assert set(d) == {
        "inputs", "point", "lo", "hi", "interval_alpha", "out_of_envelope", "illustrative",
    }
