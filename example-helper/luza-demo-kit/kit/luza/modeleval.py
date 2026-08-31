"""Honest model evaluation (FIX_PLAN.md S6).

The original scripts scored models with ``r2_score(y, model.predict(X))`` — i.e.
they evaluated on the same rows the model was fit on. With 30 vehicles and a
50-tree Random Forest that reports R2 ~ 0.95 regardless of whether the model
learned anything generalisable.

This module replaces that with out-of-fold estimates:

- ``loo_*``      Leave-One-Out CV: every row predicted by a model that never saw
                 it. The primary honest number for a dataset this small.
- ``kfold_r2_*`` Repeated K-Fold R2 (mean +/- std over many shuffles): shows how
                 unstable the estimate is.
- ``naive_r2``   the old train-on-test number, kept ONLY so dashboards/readmes
                 can show the gap.
- ``baseline_mae`` MAE of always predicting the training median — the bar any
                 real model must clear.

Pure / numeric only: no plotting, no file IO.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

RANDOM_STATE = 42


@dataclass(frozen=True)
class RegressionReport:
    """Out-of-fold performance summary for one regressor on one (X, y)."""

    n: int
    n_features: int
    loo_r2: float
    loo_mae: float
    loo_rmse: float
    kfold_r2_mean: float
    kfold_r2_std: float
    naive_r2: float
    baseline_mae: float

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"n={self.n} | LOO R2={self.loo_r2:.3f} MAE={self.loo_mae:.3g} "
            f"RMSE={self.loo_rmse:.3g} | {self.kfold_r2_mean:.3f}"
            f"+/-{self.kfold_r2_std:.3f} kfold R2 | naive(train=test) R2="
            f"{self.naive_r2:.3f} | median-baseline MAE={self.baseline_mae:.3g}"
        )


def _as_xy(X, y) -> tuple[np.ndarray, np.ndarray]:
    X_arr = np.asarray(getattr(X, "values", X), dtype=float)
    y_arr = np.asarray(getattr(y, "values", y), dtype=float).ravel()
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"X has {X_arr.shape[0]} rows, y has {y_arr.shape[0]}")
    if not np.isfinite(X_arr).all() or not np.isfinite(y_arr).all():
        raise ValueError("X / y contain NaN or inf; clean before evaluating")
    return X_arr, y_arr


def loo_predictions(estimator, X, y) -> np.ndarray:
    """Leave-One-Out out-of-fold predictions (one per row, model never saw it)."""
    X_arr, y_arr = _as_xy(X, y)
    if X_arr.shape[0] < 3:
        raise ValueError("need >=3 rows for LOO CV")
    return cross_val_predict(clone(estimator), X_arr, y_arr, cv=LeaveOneOut())


def repeated_kfold_r2(
    estimator,
    X,
    y,
    n_splits: int = 5,
    n_repeats: int = 20,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Mean and std of R2 over ``n_repeats`` shuffled K-fold splits.

    ``n_splits`` is clamped so every fold has >=2 rows, so this stays usable on
    the tiny LUZA dataset without the caller having to special-case it.
    """
    X_arr, y_arr = _as_xy(X, y)
    n = X_arr.shape[0]
    n_splits = max(2, min(n_splits, n // 2))
    scores = cross_val_score(
        clone(estimator),
        X_arr,
        y_arr,
        cv=RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        ),
        scoring="r2",
    )
    return float(scores.mean()), float(scores.std())


def naive_r2(estimator, X, y) -> float:
    """The old train-on-test R2. Optimistic by construction — contrast only."""
    X_arr, y_arr = _as_xy(X, y)
    model = clone(estimator).fit(X_arr, y_arr)
    return float(r2_score(y_arr, model.predict(X_arr)))


def baseline_mae(y) -> float:
    """MAE of always predicting the median of ``y``. The bar to beat."""
    y_arr = np.asarray(getattr(y, "values", y), dtype=float).ravel()
    return float(mean_absolute_error(y_arr, np.full_like(y_arr, np.median(y_arr))))


def evaluate_regressor(
    estimator,
    X,
    y,
    n_splits: int = 5,
    n_repeats: int = 20,
) -> RegressionReport:
    """Full honest report for ``estimator`` on ``(X, y)``."""
    X_arr, y_arr = _as_xy(X, y)
    oof = loo_predictions(estimator, X_arr, y_arr)
    kf_mean, kf_std = repeated_kfold_r2(
        estimator, X_arr, y_arr, n_splits=n_splits, n_repeats=n_repeats
    )
    return RegressionReport(
        n=int(X_arr.shape[0]),
        n_features=int(X_arr.shape[1]),
        loo_r2=float(r2_score(y_arr, oof)),
        loo_mae=float(mean_absolute_error(y_arr, oof)),
        loo_rmse=float(root_mean_squared_error(y_arr, oof)),
        kfold_r2_mean=kf_mean,
        kfold_r2_std=kf_std,
        naive_r2=naive_r2(estimator, X_arr, y_arr),
        baseline_mae=baseline_mae(y_arr),
    )


@dataclass(frozen=True)
class ClassificationReport:
    """Stratified out-of-fold performance for a binary classifier.

    The old ``classify_efficiency`` reported ``(pred == y).mean()`` on the
    training rows with a hard-coded 140 Wh/km cut. On an imbalanced split that
    accuracy is beaten by "always predict the majority class" — which is why
    ``majority_baseline_acc`` and ``cv_f1`` are reported alongside it.
    """

    n: int
    n_features: int
    class_balance: dict
    majority_class: int
    majority_baseline_acc: float
    cv_accuracy: float
    cv_f1_macro: float
    cv_f1_std: float
    confusion: list  # [[tn, fp], [fn, tp]] in label-sorted order

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"n={self.n} | balance={self.class_balance} | "
            f"CV acc={self.cv_accuracy:.3f} (majority baseline "
            f"{self.majority_baseline_acc:.3f}) | "
            f"CV F1-macro={self.cv_f1_macro:.3f}+/-{self.cv_f1_std:.3f} | "
            f"confusion={self.confusion}"
        )


def _as_xy_binary(X, y) -> tuple[np.ndarray, np.ndarray]:
    X_arr, _ = _as_xy(X, np.zeros(len(getattr(X, "values", X))))
    y_arr = np.asarray(getattr(y, "values", y)).ravel().astype(int)
    classes = np.unique(y_arr)
    if classes.size != 2:
        raise ValueError(f"expected 2 classes, got {classes.tolist()}")
    return X_arr, y_arr


def evaluate_classifier(
    estimator,
    X,
    y,
    n_splits: int = 5,
    n_repeats: int = 20,
    random_state: int = RANDOM_STATE,
) -> ClassificationReport:
    """Stratified-CV report: accuracy, F1-macro (mean/std), confusion, baseline."""
    X_arr, y_arr = _as_xy_binary(X, y)
    classes, counts = np.unique(y_arr, return_counts=True)
    k = int(min(n_splits, counts.min()))
    if k < 2:
        raise ValueError(
            f"smallest class has {int(counts.min())} rows; need >=2 for stratified CV"
        )
    oof = cross_val_predict(
        clone(estimator),
        X_arr,
        y_arr,
        cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state),
    )
    f1s = cross_val_score(
        clone(estimator),
        X_arr,
        y_arr,
        cv=RepeatedStratifiedKFold(
            n_splits=k, n_repeats=n_repeats, random_state=random_state
        ),
        scoring="f1_macro",
    )
    return ClassificationReport(
        n=int(y_arr.size),
        n_features=int(X_arr.shape[1]),
        class_balance={int(c): int(n) for c, n in zip(classes, counts)},
        majority_class=int(classes[counts.argmax()]),
        majority_baseline_acc=float(counts.max() / counts.sum()),
        cv_accuracy=float(accuracy_score(y_arr, oof)),
        cv_f1_macro=float(f1s.mean()),
        cv_f1_std=float(f1s.std()),
        confusion=confusion_matrix(y_arr, oof, labels=classes).tolist(),
    )
