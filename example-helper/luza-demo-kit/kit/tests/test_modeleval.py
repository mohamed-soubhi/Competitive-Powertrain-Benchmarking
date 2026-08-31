"""Honest-evaluation harness contracts (S6).

New file. Exercises ``luza.modeleval`` only. No callers yet; ``ml_prediction.py``
adopts it in S12.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from luza.modeleval import (
    ClassificationReport,
    RegressionReport,
    baseline_mae,
    evaluate_classifier,
    evaluate_regressor,
    loo_predictions,
    naive_r2,
    repeated_kfold_r2,
)


def _linear_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * X[:, 2] + 3.0
    return X, y


def _noise_data(n=40, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 5)), rng.normal(size=n)


def test_loo_recovers_a_clean_linear_signal():
    X, y = _linear_data()
    rep = evaluate_regressor(LinearRegression(), X, y, n_repeats=5)
    assert rep.loo_r2 > 0.99
    assert rep.kfold_r2_mean > 0.99
    assert rep.loo_mae < 1e-6 or rep.loo_mae < 0.05


def test_harness_exposes_overfitting_that_naive_r2_hides():
    X, y = _noise_data()
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rep = evaluate_regressor(rf, X, y, n_repeats=5)
    # train-on-test looks like the model learned something...
    assert rep.naive_r2 > 0.4
    # ...out-of-fold shows it did not.
    assert rep.loo_r2 < 0.2
    assert rep.loo_r2 < rep.naive_r2


def test_baseline_mae_is_median_prediction_error():
    y = np.array([1.0, 2.0, 2.0, 10.0])
    assert baseline_mae(y) == pytest.approx(np.mean(np.abs(y - 2.0)))


def test_report_is_frozen_and_serialisable():
    X, y = _linear_data(n=20)
    rep = evaluate_regressor(LinearRegression(), X, y, n_repeats=3)
    assert isinstance(rep, RegressionReport)
    with pytest.raises(Exception):
        rep.loo_r2 = 0.0  # frozen dataclass
    d = rep.as_dict()
    assert set(d) == {
        "n", "n_features", "loo_r2", "loo_mae", "loo_rmse",
        "kfold_r2_mean", "kfold_r2_std", "naive_r2", "baseline_mae",
    }
    assert d["n"] == 20 and d["n_features"] == 3
    assert isinstance(rep.summary(), str)


def test_small_n_does_not_crash_kfold():
    X, y = _linear_data(n=8)
    mean, std = repeated_kfold_r2(LinearRegression(), X, y, n_splits=5, n_repeats=3)
    assert np.isfinite(mean) and std >= 0.0


def test_guards_reject_nan_and_tiny_input():
    X, y = _linear_data(n=10)
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate_regressor(LinearRegression(), Xn, y)
    with pytest.raises(ValueError):
        loo_predictions(LinearRegression(), X[:2], y[:2])


def test_naive_r2_is_optimistic_for_rf():
    X, y = _linear_data()
    rf = RandomForestRegressor(n_estimators=30, random_state=0)
    assert naive_r2(rf, X, y) > 0.9


# --- classifier harness (S8) ---


def _separable_binary(n=60, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y


def test_classifier_report_shape_and_balanced_baseline():
    X, y = _separable_binary()
    rep = evaluate_classifier(
        RandomForestClassifier(n_estimators=40, random_state=0), X, y, n_repeats=4
    )
    assert isinstance(rep, ClassificationReport)
    assert sum(rep.class_balance.values()) == rep.n == 60
    assert np.array(rep.confusion).shape == (2, 2)
    assert 0.0 <= rep.cv_f1_macro <= 1.0
    assert 0.4 <= rep.majority_baseline_acc <= 0.6           # ~balanced
    assert rep.cv_accuracy > rep.majority_baseline_acc        # model beats baseline
    assert set(rep.as_dict()) == {
        "n", "n_features", "class_balance", "majority_class",
        "majority_baseline_acc", "cv_accuracy", "cv_f1_macro",
        "cv_f1_std", "confusion",
    }


def test_classifier_reports_majority_baseline_on_imbalanced_data():
    y = np.array([0] * 54 + [1] * 6)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    rep = evaluate_classifier(
        RandomForestClassifier(n_estimators=20, random_state=0), X, y, n_repeats=3
    )
    assert rep.majority_class == 0
    assert rep.majority_baseline_acc == pytest.approx(54 / 60)
    assert rep.class_balance == {0: 54, 1: 6}


def test_classifier_guards_on_single_class_and_tiny_class():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    with pytest.raises(ValueError):
        evaluate_classifier(RandomForestClassifier(), X, np.zeros(20, dtype=int))
    y = np.array([0] * 19 + [1])
    with pytest.raises(ValueError, match="stratified"):
        evaluate_classifier(RandomForestClassifier(), X, y)
