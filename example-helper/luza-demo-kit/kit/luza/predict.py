"""Hypothetical-vehicle prediction with honesty guards (FIX_PLAN.md S10).

The old ``predict_hypothetical`` fed four made-up spec sheets into a Random
Forest and printed the point predictions as if they were facts. Two problems:

1. **Extrapolation.** A tree model predicts a flat line outside the range it was
   trained on. Three of the four hypotheticals had a ``battery_useable_kwh`` or
   ``power_kw`` above anything in the 30-car training set, so those numbers are
   the model clamping, not reasoning.
2. **No uncertainty.** A single number hides that with n=30 the spread is large.

This module adds:
- ``training_envelope`` / ``flag_out_of_envelope`` — which inputs are outside the
  min..max the model actually saw.
- ``bootstrap_prediction_interval`` — resample the training rows, refit, and
  report a percentile interval, not just a point.
- every result carries ``illustrative=True``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone

RANDOM_STATE = 42


@dataclass(frozen=True)
class HypotheticalPrediction:
    inputs: dict
    point: float
    lo: float
    hi: float
    interval_alpha: float
    out_of_envelope: list[str]
    illustrative: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


def training_envelope(X: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-column min/max/p5/p95 of the training features."""
    num = X.apply(pd.to_numeric, errors="coerce")
    return {
        c: {
            "min": float(num[c].min()),
            "max": float(num[c].max()),
            "p5": float(num[c].quantile(0.05)),
            "p95": float(num[c].quantile(0.95)),
        }
        for c in num.columns
    }


def flag_out_of_envelope(row: dict, envelope: dict[str, dict[str, float]]) -> list[str]:
    """Feature names whose value falls outside the training min..max."""
    out: list[str] = []
    for name, bounds in envelope.items():
        if name not in row or row[name] is None:
            continue
        v = float(row[name])
        if v < bounds["min"] or v > bounds["max"]:
            out.append(name)
    return sorted(out)


def bootstrap_prediction_interval(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    x_new: dict,
    n_boot: int = 400,
    alpha: float = 0.1,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    """(point, lo, hi): refit ``estimator`` on ``n_boot`` bootstrap resamples.

    ``point`` is the median bootstrap prediction; ``lo``/``hi`` are the
    ``alpha/2`` and ``1 - alpha/2`` percentiles.
    """
    X_arr = np.asarray(X.apply(pd.to_numeric, errors="coerce"), dtype=float)
    y_arr = np.asarray(getattr(y, "values", y), dtype=float).ravel()
    cols = list(X.columns)
    x_vec = np.array([[float(x_new[c]) for c in cols]], dtype=float)

    n = X_arr.shape[0]
    rng = np.random.default_rng(random_state)
    preds = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        model = clone(estimator).fit(X_arr[idx], y_arr[idx])
        preds[b] = model.predict(x_vec)[0]

    lo = float(np.percentile(preds, 100 * alpha / 2))
    hi = float(np.percentile(preds, 100 * (1 - alpha / 2)))
    return float(np.median(preds)), lo, hi


def predict_hypotheticals(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    scenarios: list[dict],
    alpha: float = 0.1,
    n_boot: int = 400,
    random_state: int = RANDOM_STATE,
) -> list[HypotheticalPrediction]:
    """Bootstrap interval + envelope flags for each scenario dict."""
    env = training_envelope(X)
    results: list[HypotheticalPrediction] = []
    for sc in scenarios:
        point, lo, hi = bootstrap_prediction_interval(
            estimator, X, y, sc, n_boot=n_boot, alpha=alpha, random_state=random_state
        )
        results.append(
            HypotheticalPrediction(
                inputs=dict(sc),
                point=point,
                lo=lo,
                hi=hi,
                interval_alpha=alpha,
                out_of_envelope=flag_out_of_envelope(sc, env),
            )
        )
    return results
