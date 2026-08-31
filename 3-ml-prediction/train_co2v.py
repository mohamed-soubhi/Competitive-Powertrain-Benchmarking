#!/usr/bin/env python3
"""Train and honestly evaluate CO2v (VECTO declared g/km) regressors.

Two feature sets:
  base  — mass + vehicle class + powertrain + fuel + flags. All reporting years.
  rich  — base + engine ratings + axle config. 2019–2020 only (the 2023 viewer
          table has no engine data).

For each: HistGradientBoosting + a Linear baseline, scored with shuffled 5-fold
CV on a bounded subsample, next to the train-on-test (naive) number and the
predict-the-median baseline. Writes a JSON report + one joblib model per set.

    uv run python 3-ml-prediction/train_co2v.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

from powerbench.dataio import load_hdv  # noqa: E402
from powerbench.features import (  # noqa: E402
    CO2V_FEATURES_BASE,
    CO2V_FEATURES_RICH,
    CO2V_TARGET,
    build_xy,
)
from powerbench.modeleval import (  # noqa: E402
    evaluate_regressor,
    oof_scatter_sample,
    permutation_importance_df,
    training_envelope,
)
from powerbench.paths import ML_OUTPUT_DIR, ensure_dirs  # noqa: E402

log = logging.getLogger("powerbench.ml.co2v")

RICH_YEARS = (2019, 2020)
OUT_JSON = ML_OUTPUT_DIR / "co2v_models.json"


def hgb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=42
    )


def train_one(tag: str, df, features: list[str]) -> dict:
    X, y = build_xy(df, features, CO2V_TARGET)
    log.info("[%s] X=%s  target n=%d  mean=%.0f", tag, X.shape, len(y), y.mean())

    out: dict = {"tag": tag, "n_rows": int(len(y)), "n_features": int(X.shape[1]),
                 "feature_columns": list(X.columns), "models": {}}

    for name, est in (("HistGradientBoosting", hgb()), ("Linear", LinearRegression())):
        t0 = time.time()
        rep = evaluate_regressor(est, X, y)
        out["models"][name] = rep.as_dict()
        log.info("[%s] %-20s %s  (%.0fs)", tag, name, rep.summary(), time.time() - t0)

    actual, oof = oof_scatter_sample(hgb(), X, y, sample_cap=4000)
    out["scatter"] = {"actual": actual, "oof": oof}

    imp = permutation_importance_df(hgb(), X, y, sample_cap=15000, n_repeats=5)
    out["importance"] = imp.head(15).to_dict(orient="records")

    out["envelope"] = training_envelope(X)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    ensure_dirs()
    df = load_hdv()
    log.info("loaded %d rows", len(df))

    results = {
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": CO2V_TARGET,
        "sets": {
            "base": train_one("base", df, CO2V_FEATURES_BASE),
            "rich": train_one("rich", df[df["MS_Year"].isin(RICH_YEARS)], CO2V_FEATURES_RICH),
        },
    }
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("wrote %s", OUT_JSON.name)

    for tag, s in results["sets"].items():
        h = s["models"]["HistGradientBoosting"]
        print(f"\n{tag}: CV R2={h['kfold_r2_mean']:.3f}±{h['kfold_r2_std']:.3f}  "
              f"CV MAE={h['kfold_mae']:.1f} vs median-baseline {h['baseline_mae']:.1f}  "
              f"(naive R2={h['naive_r2']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
