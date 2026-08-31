"""Honest regression evaluation for a large tabular dataset.

The LUZA-kit original did Leave-One-Out on ~30 rows. Here n is ~10^5–10^6, so
LOO is neither possible nor needed: the headline is **shuffled K-fold CV**
(out-of-fold), fitted on a bounded random subsample for speed, with

* ``naive_r2``      — train == test on the same subsample, kept only to show the
                      optimism gap.
* ``baseline_mae``  — MAE of always predicting the training median. Any real
                      model must clear this.
* ``kfold_r2_std``  — spread of the fold scores.

Pure / numeric: no plotting, no file IO.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score

RANDOM_STATE = 42


@dataclass(frozen=True)
class RegressionReport:
    n: int
    n_used_for_cv: int
    n_features: int
    kfold_r2_mean: float
    kfold_r2_std: float
    kfold_mae: float
    kfold_rmse: float
    naive_r2: float
    baseline_mae: float
    target_mean: float
    target_std: float

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"n={self.n} (cv on {self.n_used_for_cv}) | "
            f"CV R2={self.kfold_r2_mean:.3f}±{self.kfold_r2_std:.3f} | "
            f"CV MAE={self.kfold_mae:.1f} (median-baseline {self.baseline_mae:.1f}) | "
            f"naive R2={self.naive_r2:.3f}"
        )


def _subsample(n: int, cap: int, seed: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, cap, replace=False)


def _xy(X, y) -> tuple[np.ndarray, np.ndarray]:
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y, dtype=float).ravel()
    if Xa.ndim == 1:
        Xa = Xa.reshape(-1, 1)
    if not np.isfinite(Xa).all() or not np.isfinite(ya).all():
        raise ValueError("X / y contain NaN or inf; clean before evaluating")
    return Xa, ya


def evaluate_regressor(
    estimator,
    X,
    y,
    *,
    sample_cap: int = 60_000,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> RegressionReport:
    Xa, ya = _xy(X, y)
    idx = _subsample(len(ya), sample_cap, random_state)
    Xs, ys = Xa[idx], ya[idx]
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    r2 = cross_val_score(clone(estimator), Xs, ys, cv=cv, scoring="r2")
    neg_mae = cross_val_score(clone(estimator), Xs, ys, cv=cv, scoring="neg_mean_absolute_error")
    neg_rmse = cross_val_score(clone(estimator), Xs, ys, cv=cv, scoring="neg_root_mean_squared_error")

    fitted = clone(estimator).fit(Xs, ys)
    naive = float(r2_score(ys, fitted.predict(Xs)))
    baseline = float(mean_absolute_error(ys, np.full_like(ys, np.median(ys))))

    return RegressionReport(
        n=int(len(ya)),
        n_used_for_cv=int(len(ys)),
        n_features=int(Xa.shape[1]),
        kfold_r2_mean=float(r2.mean()),
        kfold_r2_std=float(r2.std()),
        kfold_mae=float(-neg_mae.mean()),
        kfold_rmse=float(-neg_rmse.mean()),
        naive_r2=naive,
        baseline_mae=baseline,
        target_mean=float(ys.mean()),
        target_std=float(ys.std()),
    )


def oof_scatter_sample(
    estimator,
    X,
    y,
    *,
    sample_cap: int = 6_000,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple[list[float], list[float]]:
    """(actual, out-of-fold predicted) on a small sample, for an actual-vs-pred plot."""
    Xa, ya = _xy(X, y)
    idx = _subsample(len(ya), sample_cap, random_state)
    Xs, ys = Xa[idx], ya[idx]
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = cross_val_predict(clone(estimator), Xs, ys, cv=cv)
    return [float(v) for v in ys], [float(v) for v in oof]


def permutation_importance_df(
    estimator,
    X,
    y,
    *,
    sample_cap: int = 20_000,
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
):
    """DataFrame(feature, importance_mean, importance_std) sorted desc. Needs pandas."""
    import pandas as pd
    from sklearn.inspection import permutation_importance

    Xa, ya = _xy(X, y)
    cols = list(getattr(X, "columns", [f"f{i}" for i in range(Xa.shape[1])]))
    idx = _subsample(len(ya), sample_cap, random_state)
    Xs, ys = Xa[idx], ya[idx]
    model = clone(estimator).fit(Xs, ys)
    r = permutation_importance(model, Xs, ys, n_repeats=n_repeats, random_state=random_state,
                               scoring="r2")
    return (
        pd.DataFrame({"feature": cols, "importance_mean": r.importances_mean,
                      "importance_std": r.importances_std})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def training_envelope(X) -> dict[str, dict[str, float]]:
    """min / max / p1 / p99 per numeric column — for out-of-range what-if flags."""
    import pandas as pd

    d = X if hasattr(X, "columns") else pd.DataFrame(X)
    num = d.apply(pd.to_numeric, errors="coerce")
    return {
        c: {
            "min": float(num[c].min()), "max": float(num[c].max()),
            "p1": float(num[c].quantile(0.01)), "p99": float(num[c].quantile(0.99)),
        }
        for c in num.columns
    }
