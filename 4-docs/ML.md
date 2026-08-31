# ML step — predicting CO2v

## Question

Given a heavy-duty vehicle's spec sheet, how well can its **whole-vehicle CO2**
(`CO2v`, VECTO declared g/km) be predicted — and which specs carry the signal?

Scope: vehicle groups **4 / 5 / 9 / 10** (the VECTO classes with a comparable
`CO2v`), all three reporting years where the feature is available.

## Two feature sets

| Set | Features | Coverage |
|-----|----------|----------|
| **base** | GVW, curb mass, `VehicleGroup`, `powertrain_class`, `Engine_FuelType`, ZE / hybrid / dual-fuel / vocational flags | every year (2019 / 2020 / 2023) |
| **rich** | base **+** `Engine_RatedPower_kw`, `Engine_Displacement_ltr`, `Engine_RatedSpeed_rpm`, `AxleConfiguration` | 2019–2020 only (the 2023 viewer table has no engine data) |

Categoricals are one-hot encoded (explicit NaN level); numeric gaps median-filled.

## Leakage guard

`powerbench/features.LEAKY` bans every CO2 / fuel-consumption column as a feature
for `CO2v`:

```
WHTC_CO2_gkwh, WHSC_CO2_gkwh, COL_CO2_gtkm, COL_CO2_gkm,
COL_FuelConsumption_l100km, MS_SpecificCO2Emissions
```

The engine-cycle CO2 (WHTC/WHSC g/kWh) is an **input to the VECTO calculation
that produces `CO2v`** — using it would be predicting CO2 from CO2.
`build_xy()` calls `assert_no_leakage()` and raises before building the matrix.

## Honest evaluation

n is ~10^5–10^6, so Leave-One-Out (the LUZA-kit default) is dropped. Instead:

- **Headline:** shuffled 5-fold cross-validation, scored **out-of-fold**, on a
  bounded 60 000-row random subsample (for speed).
- **`naive_r2`:** train-on-test on the same subsample — kept only to show the
  optimism gap.
- **`baseline_mae`:** MAE of always predicting the training median. A useful
  model must clear this.
- **`kfold_r2_std`:** spread across the five folds.

## Results (out-of-fold)

| Feature set | CV R² | CV MAE (g/km) | median-baseline MAE | naive R² |
|-------------|-------|---------------|---------------------|----------|
| **rich** (+ engine ratings, 2019–20) | **0.60** ± 0.01 | **27.8** | 49.4 | 0.65 |
| **base** (mass + class + powertrain, all years) | **0.45** ± 0.02 | 45.2 | 62.2 | 0.48 |
| Linear (either set) | ~0.31 | — | — | — |

Reading it:

- The signal is **non-linear** — a linear model reaches only R² ≈ 0.31, the
  gradient-boosted model 0.45 / 0.60.
- **Curb mass and vehicle group** carry most of the base signal; **engine
  displacement and power** give the rich set its extra ~15 R² points — physically
  sensible (a bigger, more powerful engine in a heavier truck emits more per km).
- The **naive ↔ CV gap is small** (0.03–0.05), so the model is not overfitting.
- `Engine_FuelType_nan` shows up in the base importances because 2023 rows have
  no fuel type — the model uses it as a mild "is this a 2023 record" proxy. Noted,
  not hidden.

## Files

| Path | Role |
|------|------|
| `powerbench/features.py` | feature sets, `LEAKY`, `assert_no_leakage`, `build_xy` |
| `powerbench/modeleval.py` | `evaluate_regressor` → frozen `RegressionReport`, `oof_scatter_sample`, `permutation_importance_df`, `training_envelope` |
| `3-ml-prediction/train_co2v.py` | trains both sets (HGB + Linear), writes `3-ml-prediction/output/co2v_models.json` |
| app **ML tab** | metric cards, MAE comparison, actual-vs-OOF scatter, permutation importance, and a **what-if** predictor (refit in-process so it always matches the running sklearn) |

## Reproduce

```bash
uv run python 3-ml-prediction/train_co2v.py     # ~90 s -> co2v_models.json
uv run pytest -q tests/test_features.py tests/test_modeleval.py
uv run streamlit run app/streamlit_app.py        # ML tab
```

Or from the app: **Pipeline** tab → "Train models only" (or the full
mine → load → train button).

## What-if predictor

Sliders for the raw inputs → predicted `CO2v` ± the model's CV MAE. Any input
outside the training min/max raises an **extrapolation** warning
(`training_envelope`). Every number is labelled illustrative — the model sees
only these fields; real `CO2v` also depends on aerodynamics, tyres, gearbox and
auxiliaries.

## Caveats

- 2019–2020 and 2023 use **different VECTO versions**; the `base` model mixes
  them, so treat its cross-year use as directional.
- `CO2v` itself is a *simulated* declared figure, not a measured tailpipe number.
- The bounded-subsample CV is a speed/rigour trade-off; full-data numbers move by
  < 0.01 R² in spot checks.
