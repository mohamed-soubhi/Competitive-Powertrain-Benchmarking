"""K-Means segmentation contracts (S9)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from luza.cluster import SegmentResult, fit_segments, silhouette_scan


def _blob_df(centers=3, n=90, seed=0):
    X, _ = make_blobs(n_samples=n, centers=centers, cluster_std=0.6, random_state=seed)
    return pd.DataFrame({"power_kw": X[:, 0] * 40 + 200, "battery_useable_kwh": X[:, 1] * 10 + 70})


def _ev():
    from luza.dataio import load_ev_specs
    from luza.paths import CLEAN_DIR

    return load_ev_specs(CLEAN_DIR / "ev_database_2026-08-29_reclean.csv")


def test_silhouette_scan_recovers_true_cluster_count():
    df = _blob_df(centers=3)
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(df.values)
    scan = silhouette_scan(Xs, (2, 3, 4, 5, 6))
    assert set(scan) <= {2, 3, 4, 5, 6}
    assert max(scan, key=scan.get) == 3


def test_fit_segments_tiers_are_a_total_order_no_overlap():
    res = fit_segments(_blob_df(centers=3), ["power_kw", "battery_useable_kwh"])
    assert isinstance(res, SegmentResult)
    assert res.k == 3
    assert set(res.tier) == {1, 2, 3}                      # every tier used, none extra
    assert sum(res.sizes.values()) == len(res.tier) == len(res.row_index)
    assert res.inertia > 0 and res.silhouette > 0.5


def test_tiers_rank_low_to_high_on_rank_axis():
    df = _blob_df(centers=3)
    res = fit_segments(df, ["power_kw", "battery_useable_kwh"], rank_by="power_kw")
    s = pd.Series(df["power_kw"].to_numpy(), index=res.tier).groupby(level=0).mean()
    assert s.loc[1] < s.loc[res.k]                          # tier 1 = lowest power
    assert res.rank_axis == "power_kw"


def test_explicit_k_overrides_silhouette_pick():
    res = fit_segments(_blob_df(centers=3), ["power_kw", "battery_useable_kwh"], k=4)
    assert res.k == 4
    assert set(res.tier) == {1, 2, 3, 4}


def test_runs_on_real_ev_data():
    res = fit_segments(_ev(), ["battery_useable_kwh", "power_kw", "charging_dc_kw", "architecture_v"])
    assert 2 <= res.k <= 6
    assert sum(res.sizes.values()) == len(res.row_index) >= 20
    assert len(res.silhouette_by_k) >= 2


def test_needs_two_usable_features():
    df = pd.DataFrame({"power_kw": [1.0, 2, 3, 4], "dead": [np.nan] * 4})
    with pytest.raises(ValueError, match="usable feature"):
        fit_segments(df, ["power_kw", "dead"])
