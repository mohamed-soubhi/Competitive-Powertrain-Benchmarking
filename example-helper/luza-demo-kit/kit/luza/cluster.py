"""Market segmentation via K-Means (FIX_PLAN.md S9).

Fixes in this module vs the old ``cluster_vehicles``:

- **k is chosen, not assumed.** The old code hard-coded ``n_clusters=3``. Here a
  silhouette scan over k = 2..6 picks the k with the best mean silhouette.
- **Labels are a total order, so they cannot overlap.** The old code named
  clusters with independent ``if`` tests (``avg_power > 300`` -> "Performance",
  ``avg_battery < 60`` -> "Compact", else "Mainstream") — a cluster could match
  none or several. Here clusters are ranked by their centroid on one axis
  (default ``power_kw``) and assigned tiers 1..k, low to high.
- inertia, silhouette and cluster sizes are reported, not just names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
DEFAULT_K_RANGE = (2, 3, 4, 5, 6)


@dataclass(frozen=True)
class SegmentResult:
    k: int
    features: list[str]
    labels: list[int]          # raw KMeans cluster id per row
    tier: list[int]            # 1..k, ranked low->high by ``rank_by`` centroid
    sizes: dict                # {tier: count}
    inertia: float
    silhouette: float
    silhouette_by_k: dict      # {k: mean silhouette}
    row_index: list            # df index values used (rows with a usable target)
    rank_axis: str

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        picks = ", ".join(f"k={k}:{s:.3f}" for k, s in sorted(self.silhouette_by_k.items()))
        return (
            f"k={self.k} (by silhouette; {picks}) | sizes={self.sizes} | "
            f"inertia={self.inertia:.1f} | silhouette={self.silhouette:.3f} | "
            f"tiers ranked by {self.rank_axis}"
        )


def _prep(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str], list]:
    present = [f for f in features if f in df.columns]
    num = df[present].apply(pd.to_numeric, errors="coerce")
    empty = [c for c in present if num[c].notna().sum() == 0]
    used = [c for c in present if c not in empty]
    if len(used) < 2:
        raise ValueError("need >=2 usable feature columns for clustering")
    num = num[used].dropna(how="all")
    X = num.fillna(num.median())
    return X, used, list(X.index)


def silhouette_scan(
    X_scaled: np.ndarray,
    k_values=DEFAULT_K_RANGE,
    random_state: int = RANDOM_STATE,
) -> dict[int, float]:
    """Mean silhouette for each valid k. Skips k that collapse to one cluster."""
    n = X_scaled.shape[0]
    out: dict[int, float] = {}
    for k in k_values:
        if not 2 <= k < n:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X_scaled)
        if np.unique(km.labels_).size < 2:
            continue
        out[int(k)] = float(silhouette_score(X_scaled, km.labels_))
    return out


def fit_segments(
    df: pd.DataFrame,
    features: list[str],
    k: int | None = None,
    k_range=DEFAULT_K_RANGE,
    rank_by: str = "power_kw",
    random_state: int = RANDOM_STATE,
) -> SegmentResult:
    """Scale features, pick k by silhouette (unless ``k`` given), rank into tiers."""
    X, used, idx = _prep(df, features)
    X_scaled = StandardScaler().fit_transform(X)

    scan = silhouette_scan(X_scaled, k_range, random_state)
    if not scan:
        raise ValueError("no k in k_range produced >=2 non-empty clusters")
    best_k = int(k) if k is not None else max(scan, key=scan.get)

    km = KMeans(n_clusters=best_k, n_init=10, random_state=random_state).fit(X_scaled)
    labels = km.labels_

    axis = rank_by if rank_by in used else used[0]
    axis_means = pd.Series(X[axis].to_numpy(), index=labels).groupby(level=0).mean()
    order = list(axis_means.sort_values().index)               # cluster ids, low -> high
    tier_of = {cid: t for t, cid in enumerate(order, start=1)}
    tiers = [tier_of[c] for c in labels]

    return SegmentResult(
        k=best_k,
        features=used,
        labels=labels.tolist(),
        tier=tiers,
        sizes={t: int(np.sum(np.asarray(tiers) == t)) for t in range(1, best_k + 1)},
        inertia=float(km.inertia_),
        silhouette=float(silhouette_score(X_scaled, labels)),
        silhouette_by_k=scan,
        row_index=idx,
        rank_axis=axis,
    )
