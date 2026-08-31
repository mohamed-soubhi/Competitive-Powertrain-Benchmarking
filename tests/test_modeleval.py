"""Offline tests for the honest-evaluation helpers."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from powerbench.modeleval import (
    RegressionReport,
    evaluate_regressor,
    oof_scatter_sample,
    training_envelope,
)


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 4))
    y = X @ np.array([2.0, -1.0, 0.5, 0.0]) + rng.normal(scale=0.3, size=1500)
    return X, y


def test_evaluate_regressor_report_shape(xy):
    X, y = xy
    rep = evaluate_regressor(LinearRegression(), X, y, sample_cap=1000, n_splits=5)
    assert isinstance(rep, RegressionReport)
    d = rep.as_dict()
    assert d["n"] == 1500 and d["n_used_for_cv"] == 1000 and d["n_features"] == 4
    assert d["kfold_r2_mean"] > 0.9                     # recoverable linear signal
    assert d["kfold_mae"] < d["baseline_mae"]           # beats predict-the-median
    assert d["naive_r2"] >= d["kfold_r2_mean"] - 0.05   # naive is >= CV (optimistic)


def test_evaluate_regressor_rejects_nan():
    X = np.array([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]])
    with pytest.raises(ValueError, match="NaN or inf"):
        evaluate_regressor(LinearRegression(), X, np.array([1.0, 2.0, 3.0]))


def test_oof_scatter_sample_lengths(xy):
    X, y = xy
    a, o = oof_scatter_sample(LinearRegression(), X, y, sample_cap=200)
    assert len(a) == len(o) == 200


def test_training_envelope_bounds(xy):
    import pandas as pd
    X, _ = xy
    env = training_envelope(pd.DataFrame(X, columns=list("abcd")))
    assert set(env) == set("abcd")
    assert env["a"]["min"] <= env["a"]["p1"] <= env["a"]["p99"] <= env["a"]["max"]
