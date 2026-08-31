# 🚛 Competitive Powertrain Benchmarking Dashboard

EU heavy-duty vehicle (truck/bus) powertrain benchmarking and CO2 simulation platform for **Horse Powertrain**.

[![Tests](https://img.shields.io/badge/tests-76%20passed-success?style=flat-square&logo=pytest)](tests/)
[![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square&logo=python)](pyproject.toml)
[![DuckDB](https://img.shields.io/badge/storage-DuckDB%20%2B%20Parquet-yellow?style=flat-square&logo=duckdb)](powerbench/dataio.py)
[![Streamlit](https://img.shields.io/badge/app-Streamlit-red?style=flat-square&logo=streamlit)](app/streamlit_app.py)
[![GitHub Pages](https://img.shields.io/badge/demo-GitHub%20Pages-blueviolet?style=flat-square&logo=github)](https://mohamed-soubhi.github.io/Competitive-Powertrain-Benchmarking/)

---

## 🌐 Online Demo & Deployment

- **Interactive GitHub Pages App**: [https://mohamed-soubhi.github.io/Competitive-Powertrain-Benchmarking/](https://mohamed-soubhi.github.io/Competitive-Powertrain-Benchmarking/)
- **Streamlit Community Cloud**: Deploy directly via `app/streamlit_app.py` or test the full pipeline locally.

---

## 📋 System Architecture

```mermaid
flowchart LR
    subgraph Mining [1. Live Mining]
        EEA[(EEA Discodata\nSQL-over-HTTP)] --> F1[fetch_eea_hdv.py\n2019-2020]
        EEA --> F2[fetch_eea_hdv_viewer.py\n2023]
        F1 & F2 --> RAW[Raw JSON Snapshots\n+ .prov.txt sidecars]
    end

    subgraph Pipeline [2. Validation & Ingestion]
        RAW --> REC[reclean.py]
        REC --> PYD[Pydantic HDVRow Gate\nPhysical Bounds]
        PYD --> DER[OEM Canonicalisation\nPowertrain Classifier]
        DER --> DUCK[(DuckDB + Parquet\n756k rows + manifest)]
    end

    subgraph ML [3. ML Engine]
        DUCK --> TR[train_co2v.py]
        TR --> HGB[HistGradientBoosting\n5-Fold CV / MAE 27.8]
        HGB --> MOD[co2v_models.json]
    end

    subgraph UI [4. Interactive UI]
        DUCK & MOD --> ST[Streamlit App\n8 Interactive Tabs]
    end
```

---

## 🚀 Application Tabs & Features

| Tab | Purpose | Key Visualizations / Controls |
|---|---|---|
| **① Pipeline** | Live mining & dataset orchestration | Multi-year picker, live Discodata streamer, no-network re-clean, one-click retrain |
| **② Overview** | High-level fleet composition | Fleet volume metrics, powertrain breakdown, OEM representation, filtered data preview |
| **③ Distributions** | Fleet parameter spread | Histograms and per-OEM box plots (CO2v, Power kW, Displacement L, GVW, RPM) |
| **④ Correlations** | Engineering trade-off analysis | Pearson heatmap (identifies negative link between engine size and brake specific CO2) |
| **⑤ Benchmark** | OEM competitive ranking | Box plots (median + IQR whiskers), multi-year trend lines, relative Δ % ranking |
| **⑥ ML** | CO2v predictive modeling & what-if | Out-of-fold scatter, permutation feature importance, interactive parameter sliders |
| **⑦ Metrics** | Regulatory & test cycle reference | Deep dive into WHTC/WHSC dynamometer cycles vs whole-truck VECTO CO2v vs g/t-km |
| **⑧ Provenance** | Auditability & data lineage | Cryptographic SHA-256 manifest, source table references, rejection counters |

---

## 🛠️ Setup & Quickstart

Managed with [`uv`](https://github.com/astral-sh/uv) (fast Python package and project manager):

```bash
# 1. Clone repository
git clone https://github.com/mohamed-soubhi/Competitive-Powertrain-Benchmarking.git
cd Competitive-Powertrain-Benchmarking

# 2. Sync dependencies (runtime + dev + streamlit)
uv sync --group app --group dev

# 3. Run complete test suite (76 offline tests)
uv run pytest -v

# 4. Launch the Streamlit dashboard
uv run streamlit run app/streamlit_app.py
```

---

## ⛏️ Data Mining & Pipeline Commands

```bash
# Dry run: Inspect generated T-SQL queries without sending network requests
uv run python 1-mining/fetch_eea_hdv.py --dry-run -v

# Mine 2019–2020 VECTO data from Discodata
uv run python 1-mining/fetch_eea_hdv.py

# Mine 2023 pre-joined viewer table
uv run python 1-mining/fetch_eea_hdv_viewer.py

# Validate raw snapshots and build DuckDB + Parquet store
uv run python 2-pipeline/reclean.py

# Train and honestly evaluate CO2v regression models (5-Fold CV)
uv run python 3-ml-prediction/train_co2v.py
```

---

## 🧠 Machine Learning Methodology

- **Target**: `CO2v` — VECTO declared specific CO2 emissions (g/km).
- **Target Leakage Guard**: Dynamometer test-cycle emissions (`WHTC_CO2_gkwh`, `WHSC_CO2_gkwh`), freight efficiency (`COL_CO2_gtkm`), and member state reported values (`MS_SpecificCO2Emissions`) are strictly banned from model inputs.
- **Evaluation**: Shuffled 5-fold cross-validation on 60,000 bounded subsamples:
  - **Rich Feature Set** (Mass + Class + Engine Specs, 2019–2020): **CV $R^2 = 0.595 \pm 0.006$**, **CV MAE = 27.8 g/km** (vs 49.4 median baseline).
  - **Base Feature Set** (Mass + Class + Powertrain, all years): **CV $R^2 = 0.448 \pm 0.018$**, **CV MAE = 45.2 g/km** (vs 62.2 median baseline).
  - **Non-Linear Signal**: HistGradientBoosting decisively outperforms linear regression ($R^2 \approx 0.312$).
- **Dominant Drivers**: Chassis curb mass and vehicle group explain most whole-vehicle emissions variance; displacement and rated speed provide significant secondary lift.

---

## 📜 Regulatory Reference

- **Euro VI Engine Approval (Reg. 595/2009)**: `WHTC` (transient dynamometer cycle) and `WHSC` (13 steady-state modes) measure brake-specific engine efficiency in `g/kWh`.
- **HDV CO2 Monitoring (Reg. (EU) 2018/956)**: Mandates whole-vehicle simulation in `VECTO` to determine declared `CO2v` in `g/km`.
- **Transport Freight Efficiency**: `COL_CO2_gtkm` measures grams of CO2 per metric tonne of freight hauled per kilometer ($g/t\cdot km$).

---

## 🧪 Test Suite

All 76 unit tests run completely offline with no network dependencies:

```bash
uv run pytest -q
```
