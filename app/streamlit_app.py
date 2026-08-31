"""Competitive Powertrain Benchmarking Dashboard — EU heavy-duty vehicles.

Interactive Streamlit app over the cleaned EEA HDV CO2 dataset. Run the pipeline
first (`1-mining/fetch_eea_hdv.py` then `2-pipeline/reclean.py`); this app reads
the DuckDB store the manifest points at.

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from powerbench.benchmark import (  # noqa: E402
    CO2_VEHICLE_GROUPS,
    METRIC_LABELS,
    co2_trend,
    filter_frame,
    improvement_ranking,
    metric_by_dimension,
    numeric_correlations,
)
from powerbench.dataio import load_hdv, provenance_line, read_manifest  # noqa: E402
from powerbench.paths import ML_OUTPUT_DIR  # noqa: E402
from powerbench.oem import POWERTRAIN_CLASSES  # noqa: E402
from powerbench.theme import (  # noqa: E402
    BRAND_ORDER,
    app_css,
    brand_color_map,
    corr_kw,
    corr_textfont,
    fold_brand,
    plotly_layout,
    tokens,
)

st.set_page_config(page_title="Powertrain Benchmarking — EU HDV", page_icon="🚛", layout="wide")

DARK = False          # rebound from the sidebar toggle below
T = tokens(False)
CMAP = brand_color_map(False)

CO2_METRICS = ["CO2v", "WHTC_CO2_gkwh", "WHSC_CO2_gkwh", "COL_CO2_gtkm", "COL_FuelConsumption_l100km"]
CORR_COLS = [
    "Engine_RatedPower_kw", "Engine_Displacement_ltr", "Engine_RatedSpeed_rpm",
    "GrossVehicleMass_t", "CurbMassChassis_kg", "WHTC_CO2_gkwh", "WHSC_CO2_gkwh", "CO2v",
]
SHORT = {
    "Engine_RatedPower_kw": "power kW", "Engine_Displacement_ltr": "displ. L",
    "Engine_RatedSpeed_rpm": "rated rpm", "GrossVehicleMass_t": "GVW t",
    "CurbMassChassis_kg": "curb kg", "WHTC_CO2_gkwh": "WHTC g/kWh",
    "WHSC_CO2_gkwh": "WHSC g/kWh", "CO2v": "CO2v g/km",
}

GLOSSARY_MD = """
## What the CO2 and spec fields mean

### Engine test-cycle CO2 — a bench test of the engine alone
The engine (not the whole truck) runs a fixed ~30-minute speed/load profile on a
dynamometer for EU type approval (Reg. 595/2009, Euro VI).

| Field | Cycle | Meaning |
|---|---|---|
| **WHTC g/kWh** | World Harmonised **Transient** Cycle | constantly changing speed + load, mimics stop-go driving, includes a cold start. Always a little higher than WHSC. |
| **WHSC g/kWh** | World Harmonised **Steady-state** Cycle | 13 fixed load points held steady — the engine's "best case". |

- **Unit `g/kWh`** = grams of CO2 per kWh of work at the crankshaft — a
  *fuel-efficiency* number for the engine, independent of truck weight, aero,
  tyres or mission.
- Typical diesel HDV: WHTC ~600–660, WHSC ~590–620. **Lower = more efficient.**
- ~99% populated, but **2019–2020 only** (the 2023 viewer table doesn't carry them).
- **Benchmark caveat:** g/kWh *falls* as engine displacement rises (big engines run
  nearer their sweet spot), so a raw g/kWh league table flatters OEMs that sell
  heavy long-haul tractors. Use it as a feature, not a ranking.

### Whole-vehicle CO2 — VECTO simulation
The truck as a system (engine + gearbox + axles + aero + tyres + auxiliaries) is
simulated by the EU tool **VECTO** over standard mission profiles.

| Field | Meaning |
|---|---|
| **CO2v** (g/km) | VECTO **declared specific CO2** for the vehicle's main mission profile. The headline whole-truck number. ~93% populated across all three years → the **primary benchmark and ML target**. Typical rigid ~660–810 g/km. 2023 uses a newer VECTO version, so cross-year moves are directional, not exact. |
| **COL_CO2_gtkm** (g/tonne-km) | one specific mission slot (long-haul, one payload). CO2 per tonne of freight moved one km — the real efficiency of *hauling goods*, normalised for truck size. Only ~2–3% populated → shown where present, not modelled in v1. |
| **COL_CO2_gkm** (g/km) | same mission, not payload-normalised. |
| **COL_FuelConsumption_l100km** | same mission, litres per 100 km. ~3% populated. |

The `LHL / LHR / RDL / RDR / UDL / UDR …` families in the raw source are the other
mission profiles (Long Haul, Regional Delivery, Urban Delivery × Loaded /
Representative payload); v1 does not pull them.

### Member-State reported
| Field | Meaning |
|---|---|
| **MS_SpecificCO2Emissions** | CO2 figure the registering country reported (not the OEM/VECTO chain). ~3–5% populated, mixed units, weak link to CO2v (r ≈ −0.19). Shown for completeness, **not modelled**. |

### Physical spec fields (model features, not CO2)
| Field | Meaning |
|---|---|
| **Engine_RatedPower_kw** | max engine power, kW (~330 avg; 115–566). |
| **Engine_Displacement_ltr** | swept volume, litres (~11–13 for line-haul diesels). |
| **Engine_RatedSpeed_rpm** | rpm at rated power (~1600–1900); lower-revving big engines tend to be more efficient. |
| **GrossVehicleMass_t** | technically permissible max laden mass, tonnes (GVW). Segment proxy. |
| **CurbMassChassis_kg** | empty chassis mass, kg. |
| **MS_TechnPermMaxLadenMass** | same concept as GVW, in kg (viewer rows converted from tonnes). |
| **VehicleGroup** | VECTO class 1–17. v1 keeps 4 / 5 / 9 / 10 for CO2 work; group 5 (4×2 rigid ~18–26 t) dominates. |
| **VehicleSubgroup** | e.g. `5-LH` = group 5, long-haul. |
| **AxleConfiguration** | `4x2`, `6x2`, `6x4`, `8x4` (driven × total wheel-ends). |
| **LegislativeClass** | `N3` (>12 t) / `N2` (3.5–12 t). |
| **country** | registration Member State (2023 rows only). |

**In one line:** g/kWh = engine efficiency on a bench · CO2v (g/km) = whole
simulated truck · g/t·km = efficiency of moving freight. We benchmark on **CO2v**
and use the rest as features.
"""

CO2V_MODELS_JSON = ML_OUTPUT_DIR / "co2v_models.json"


@st.cache_data(show_spinner=False)
def load_ml_report() -> dict | None:
    if not CO2V_MODELS_JSON.exists():
        return None
    return json.loads(CO2V_MODELS_JSON.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Fitting the what-if model…")
def whatif_model(tag: str, _key: str):
    """Refit the CO2v model in-process (avoids sklearn-version pickle issues)."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    from powerbench.features import CO2V_FEATURES_BASE, CO2V_FEATURES_RICH, CO2V_TARGET, build_xy
    from powerbench.modeleval import training_envelope

    d = load_hdv()
    feats = CO2V_FEATURES_RICH if tag == "rich" else CO2V_FEATURES_BASE
    if tag == "rich":
        d = d[d["MS_Year"].isin([2019, 2020])]
    X, y = build_xy(d, feats, CO2V_TARGET)
    if len(X) > 80_000:
        samp = X.sample(80_000, random_state=42)
        X, y = samp, y.loc[samp.index]
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                          max_leaf_nodes=31, random_state=42)
    model.fit(X.to_numpy(float), y.to_numpy(float))
    rep = load_ml_report() or {}
    cv_mae = rep.get("sets", {}).get(tag, {}).get(
        "models", {}).get("HistGradientBoosting", {}).get("kfold_mae")
    return {"model": model, "features": list(X.columns),
            "envelope": training_envelope(X), "cv_mae": cv_mae}


def whatif_vector(bundle: dict, raw: dict) -> "pd.DataFrame":
    """Build a one-row frame matching the model's one-hot feature columns."""
    cols = set(bundle["features"])
    row = {c: 0.0 for c in bundle["features"]}
    for k, v in raw.items():
        if v is None:
            continue
        if k in cols:                                        # plain numeric / bool feature
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                pass
        cands = {f"{k}_{v}"}                                 # categorical one-hot
        try:
            fv = float(v)
            cands |= {f"{k}_{fv}", f"{k}_{int(fv)}"}
        except (TypeError, ValueError):
            pass
        for c in cands & cols:
            row[c] = 1.0
    return pd.DataFrame([row])[bundle["features"]]


def styled(fig: go.Figure, **ov) -> go.Figure:
    fig.update_layout(**plotly_layout(DARK, **ov))
    return fig


def bars(fig: go.Figure) -> go.Figure:
    fig.update_traces(marker_line_width=0, marker_cornerradius=4)
    return fig


def how_to_read(text: str) -> None:
    """High-contrast 'how to read this' line (st.caption renders too faint)."""
    st.markdown(f"<div style='color:{T['text']};font-size:0.9rem;'>▸ {text}</div>",
                unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading cleaned dataset…")
def get_data(_key: str) -> pd.DataFrame:
    df = load_hdv()
    df["brand"] = df["brand"].map(fold_brand)
    return df


def manifest_key() -> str:
    m = read_manifest() or {}
    d = m.get("datasets", {}).get("hdv", {})
    return f"{d.get('file', '?')}:{d.get('sha256', '?')[:12]}"


# ------------------------------------------------------------------- live pipeline
FETCH_VEHICLE = ROOT / "1-mining" / "fetch_eea_hdv.py"
FETCH_VIEWER = ROOT / "1-mining" / "fetch_eea_hdv_viewer.py"
RECLEAN = ROOT / "2-pipeline" / "reclean.py"
TRAIN_ML = ROOT / "3-ml-prediction" / "train_co2v.py"
YEARS_VEHICLE = (2019, 2020)   # -> CO2_HeavyDutyVehicles
YEARS_VIEWER = (2023,)         # -> HDV_2023_viewer

st.session_state.setdefault("mining", False)
st.session_state.setdefault("last_run", None)


def run_stream(args: list[str]):
    """Yield stdout+stderr lines from ``python <args>`` run at the repo root."""
    proc = subprocess.Popen(
        [sys.executable, *[str(a) for a in args]],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line.rstrip("\n")
    proc.wait()
    yield f"__EXIT__ {proc.returncode}"


def build_stages(sel: list[int], mine: bool, train: bool = True) -> list[tuple[str, list]]:
    stages: list[tuple[str, list]] = []
    if mine:
        v = sorted(set(sel) & set(YEARS_VEHICLE))
        if v:
            stages.append((f"1 · Mine {v} — CO2_HeavyDutyVehicles",
                           [FETCH_VEHICLE, "--years", *[str(y) for y in v]]))
        if set(sel) & set(YEARS_VIEWER):
            stages.append(("1 · Mine [2023] — HDV_2023_viewer", [FETCH_VIEWER, "--years", "2023"]))
    stages.append(("2 · Validate + load DuckDB", [RECLEAN]))
    if train:
        stages.append(("3 · Train CO2v models (~90 s)", [TRAIN_ML]))
    return stages


def execute(stages: list[tuple[str, list]]) -> None:
    st.session_state.mining = True
    ok = True
    try:
        for label, args in stages:
            with st.status(label, expanded=True) as status:
                box, buf, code, t0 = st.empty(), [], 0, time.time()
                for line in run_stream(args):
                    if line.startswith("__EXIT__"):
                        code = int(line.split()[1]); break
                    buf.append(line)
                    box.code("\n".join(buf[-400:]) or "…")
                dt = time.time() - t0
                if code == 0:
                    status.update(label=f"✓ {label}  ·  {dt:.0f}s", state="complete")
                else:
                    status.update(label=f"✗ {label}  ·  exit {code}", state="error")
                    ok = False
                    break
    finally:
        st.session_state.mining = False
        st.session_state.last_run = time.strftime("%Y-%m-%d %H:%M:%S")
        get_data.clear()
        load_ml_report.clear()
        whatif_model.clear()
    if ok:
        st.success("Pipeline finished — reloading data.")
        st.rerun()
    else:
        st.error("Pipeline stopped on a failed stage. See the log above.")


def render_pipeline() -> None:
    st.subheader("Run the mining pipeline")
    st.markdown(
        "Fetches live from **discodata.eea.europa.eu** (EEA HDV CO2 monitoring), "
        "validates, and rebuilds the local DuckDB store. Nothing here is pre-downloaded."
    )
    busy = st.session_state.mining

    _m = read_manifest() or {}
    _hdv = _m.get("datasets", {}).get("hdv", {})
    _ml = load_ml_report() or {}
    s1, s2 = st.columns(2)
    s1.metric("Dataset rows", f"{_hdv.get('rows', 0):,}",
              help=f"from {', '.join(x['file'] for x in _hdv.get('source_snapshots', [])) or '—'}")
    s2.metric("Model trained", (_ml.get("written_at") or "—").replace("T", " ").rstrip("Z"))
    st.caption(f"Manifest written {_m.get('written_at', '—')}")

    st.markdown("**Stages:** ① mine (live EEA Discodata) → ② validate + load DuckDB → ③ train CO2v models")
    pick = st.multiselect(
        "Reporting years to mine", [*YEARS_VEHICLE, *YEARS_VIEWER],
        default=[*YEARS_VEHICLE, *YEARS_VIEWER],
        help="2019–2020 come from CO2_HeavyDutyVehicles; 2023 from HDV_2023_viewer.",
    )
    train = st.checkbox("Retrain the CO2v models after loading", value=True, disabled=busy,
                        help="Adds ~90 s. Off = rebuild the dataset only.")
    confirm = st.checkbox(
        "I understand this sends queries to the EEA Discodata endpoint", disabled=busy,
    )
    c1, c2, c3 = st.columns(3)
    go_mine = c1.button("⛏️  Full: mine → load → train", disabled=busy or not confirm or not pick,
                        width="stretch")
    go_reclean = c2.button("♻️  Re-load + train (no network)", disabled=busy, width="stretch",
                           help="Re-validate the existing raw snapshots, then train")
    go_train = c3.button("🧠  Train models only", disabled=busy, width="stretch",
                         help="Re-run 3-ml-prediction/train_co2v.py on the current dataset")
    if busy:
        st.info("A run is in progress — this page is busy until it finishes.")
    if st.session_state.last_run:
        st.caption(f"Last run this session: {st.session_state.last_run}")

    if go_mine:
        execute(build_stages(pick, mine=True, train=train))
    elif go_train:
        execute([("3 · Train CO2v models (~90 s)", [TRAIN_ML])])
    elif go_reclean:
        execute(build_stages(pick, mine=False, train=train))


# --------------------------------------------------------------------------- load
try:
    df = get_data(manifest_key())
except FileNotFoundError:
    st.title("🚛 Competitive Powertrain Benchmarking — EU Heavy-Duty Vehicles")
    st.warning("No cleaned dataset yet. Run the pipeline below to build it.")
    render_pipeline()
    st.stop()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    DARK = st.toggle("🌙 Dark mode", value=False)
    st.header("Filters")
    years = sorted(int(y) for y in df["MS_Year"].dropna().unique())
    sel_years = st.multiselect("Reporting year", years, default=years)

    all_brands = [b for b in CMAP if b in set(df["brand"])]
    sel_brands = st.multiselect("Manufacturer", all_brands, default=all_brands)

    pts = sorted(df["powertrain_class"].dropna().unique())
    sel_pts = st.multiselect("Powertrain", pts, default=pts)

    groups = sorted(int(g) for g in df["VehicleGroup"].dropna().unique())
    co2_default = [g for g in groups if g in CO2_VEHICLE_GROUPS]
    sel_groups = st.multiselect(
        "Vehicle group", groups, default=co2_default,
        help="Groups 4/5/9/10 carry VECTO-simulated CO2. Others report none.",
    )

    metric = st.selectbox(
        "CO2 / efficiency metric", CO2_METRICS,
        format_func=lambda m: METRIC_LABELS.get(m, m),
    )
    dim = st.radio("Compare by", ["brand", "oem_group", "powertrain_class"], horizontal=True)

    st.divider()
    DOCS_HTML_PATH = ROOT / "docs" / "documentation.html"
    if DOCS_HTML_PATH.exists():
        st.download_button(
            "📚 Download HTML Docs",
            data=DOCS_HTML_PATH.read_bytes(),
            file_name="powertrain_benchmarking_documentation.html",
            mime="text/html",
            help="Download standalone HTML documentation",
            width="stretch",
        )

T = tokens(DARK)
CMAP = brand_color_map(DARK)
st.markdown(app_css(DARK), unsafe_allow_html=True)

fdf = filter_frame(
    df, years=sel_years, brands=sel_brands, powertrains=sel_pts, vehicle_groups=sel_groups
)
metric_label = METRIC_LABELS.get(metric, metric)

st.title("🚛 Competitive Powertrain Benchmarking — EU Heavy-Duty Vehicles")
st.caption(
    "Source: EEA HDV CO2 monitoring (Regulation (EU) 2018/956), reporting years "
    f"{years[0]}–{years[-1]}. {len(fdf):,} of {len(df):,} rows match the current filters."
)
if fdf.empty:
    st.warning("No rows match the filters.")
    st.stop()

tab_pipe, tab_over, tab_dist, tab_corr, tab_bench, tab_ml, tab_docs, tab_prov = st.tabs(
    ["Pipeline", "Overview", "Distributions", "Correlations", "Benchmark", "ML", "Documentation", "Provenance"]
)

# --------------------------------------------------------------------------- overview
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vehicles", f"{len(fdf):,}")
    c2.metric("Manufacturers", fdf["brand"].nunique())
    c3.metric(f"Median {metric}", f"{fdf[metric].median():.1f}" if fdf[metric].notna().any() else "—")
    ze = (fdf["powertrain_class"] == "BEV / Zero-emission").sum()
    c4.metric("Zero-emission", f"{ze:,}", help="BEV / FCEV rows in the filtered set")

    left, right = st.columns(2)
    with left:
        vc = fdf["powertrain_class"].value_counts().reset_index()
        vc.columns = ["powertrain_class", "n"]
        fig = px.bar(vc, x="n", y="powertrain_class", orientation="h", text="n")
        fig.update_traces(marker_color=T["accent"], textposition="outside",
                          textfont_color=T["text"], cliponaxis=False)
        st.plotly_chart(bars(styled(fig, title="Powertrain composition", yaxis_title="",
                                    xaxis_title="vehicles", showlegend=False)), width="stretch")
        how_to_read(
            "The EU HDV fleet here is almost entirely diesel; gas (CNG/LNG) is the only "
            "sizeable alternative. Zero-emission and hybrid volumes are negligible before "
            "2021 — an electrification trend needs the later reporting years."
        )
    with right:
        gc = fdf.groupby(["MS_Year", "brand"]).size().reset_index(name="n")
        fig = px.bar(gc, x="brand", y="n", color="MS_Year", barmode="group",
                     color_discrete_sequence=[T["accent"], T["good"]])
        st.plotly_chart(bars(styled(fig, title="Vehicles by manufacturer & year",
                                    xaxis_title="", yaxis_title="vehicles")), width="stretch")
        how_to_read(
            "Counts are certified vehicle *variants* reported to the EEA, not registrations "
            "or sales. 2020 is a partial reporting period — compare rates, not raw counts."
        )
    st.markdown("**Sample of matching vehicles**")
    _cols = [c for c in [
        "MS_Year", "brand", "name", "powertrain_class", "Engine_FuelType",
        "VehicleGroup", "VehicleSubgroup", "country", "GrossVehicleMass_t",
        "Engine_RatedPower_kw", "CO2v", "WHTC_CO2_gkwh",
    ] if c in fdf.columns]
    _prev = fdf[_cols].head(500).copy()
    for _b in _prev.columns[_prev.dtypes == bool]:            # bool -> Yes/No text
        _prev[_b] = _prev[_b].map({True: "Yes", False: "No"})
    # drop columns that are entirely empty for the current selection
    _empty = [c for c in _prev.columns
              if _prev[c].replace("", pd.NA).isna().all()]
    _prev = _prev.drop(columns=_empty)
    if _empty:
        st.caption("Hidden (no data for this selection): " + ", ".join(_empty))
    st.dataframe(_prev, width="stretch", height=320, hide_index=True)

# --------------------------------------------------------------------------- benchmark
with tab_bench:
    if fdf[metric].notna().sum() < 10:
        st.info(f"`{metric}` has too few values in this selection ({fdf[metric].notna().sum()}). "
                "Pick another metric or widen the filters.")
    else:
        rank = metric_by_dimension(fdf, metric, dim, min_count=20)   # for order + n
        keep = list(rank[dim])
        sub = fdf[fdf[dim].isin(keep)].dropna(subset=[metric])
        order = list(rank[dim])                                       # ascending mean
        counts = dict(zip(rank[dim], rank["n"]))
        fig = px.box(
            sub, x=metric, y=dim,
            color=dim if dim == "brand" else None,
            color_discrete_map=CMAP if dim == "brand" else None,
            category_orders={dim: order}, points=False,
        )
        if dim != "brand":
            fig.update_traces(marker_color=T["accent"], line_color=T["accent"])
        fig.update_traces(
            hovertemplate="%{y}<br>median %{x:.1f}  ·  Q1 %{q1:.1f}  ·  Q3 %{q3:.1f}<extra></extra>"
        )
        fig.update_yaxes(autorange="reversed", ticktext=[f"{b}  (n={counts.get(b, 0):,})" for b in order],
                         tickvals=order)
        st.plotly_chart(styled(fig, title=f"{metric_label} by {dim} — lower is better",
                               xaxis_title=metric_label, yaxis_title="", showlegend=False),
                        width="stretch")
        how_to_read(
            f"Box = interquartile range of {metric_label}; line = median; whiskers = 1.5×IQR. "
            "Left-most (lowest median) OEM leads; box width shows how consistent the range is. "
            "Mix still matters — an OEM heavy in long-haul tractors (group 5) sits higher; "
            "narrow the vehicle-group filter for a like-for-like read."
        )

        tr = co2_trend(fdf, metric, dim, min_count=20)
        if tr["MS_Year"].nunique() > 1:
            fig = px.line(
                tr, x="MS_Year", y="mean", color=dim, markers=True,
                color_discrete_map=CMAP if dim == "brand" else None,
            )
            fig.update_traces(line_width=2.5, marker_size=9)
            fig.update_xaxes(tickvals=sorted(tr["MS_Year"].unique()))
            st.plotly_chart(styled(fig, title=f"{metric_label} over time",
                                   xaxis_title="reporting year", yaxis_title=metric_label),
                            width="stretch")

            yrs = sorted(tr["MS_Year"].unique())
            imp = improvement_ranking(fdf, metric, dim, first_year=yrs[0], last_year=yrs[-1], min_count=20)
            if not imp.empty:
                fig = px.bar(imp, x="delta", y=dim, orientation="h", text="pct")
                fig.update_traces(
                    texttemplate="%{text:.1f}%", textfont_color=T["text"],
                    textposition="outside", cliponaxis=False,
                    marker_color=[T["good"] if d < 0 else T["bad"] for d in imp["delta"]],
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(bars(styled(
                    fig, title=f"Change {yrs[0]}→{yrs[-1]}  (blue = CO2 fell)",
                    xaxis_title=f"Δ {metric_label}", yaxis_title="", showlegend=False,
                )), width="stretch")
                how_to_read(
                    "Bars left of zero improved; the % label is the relative change. "
                    "2019–2020 and 2023 use different VECTO versions, so read cross-year "
                    "moves as directional, not exact."
                )
        else:
            st.info("Only one reporting year in the selection — no trend to plot.")

# --------------------------------------------------------------------------- distributions
with tab_dist:
    num_cols = [c for c in dict.fromkeys([metric, *CORR_COLS])
                if c in fdf and fdf[c].notna().sum() > 5]
    split = st.checkbox("Split by manufacturer", value=False)
    how_to_read(
        "One chart per numeric field, stacked. Histogram = spread across the filtered "
        "fleet; the split view is a per-OEM box (median line, IQR box) ordered by median. "
        "Long right tails on power / displacement are the heavy long-haul tractors."
    )
    _brand_order = [b for b in BRAND_ORDER if b in set(fdf["brand"])]
    _pt_order = [p for p in POWERTRAIN_CLASSES if p in set(fdf["powertrain_class"])]
    for field in num_cols:
        if split:
            fig = px.box(fdf.dropna(subset=[field]), x="brand", y=field, color="brand",
                         color_discrete_map=CMAP, points=False,
                         category_orders={"brand": _brand_order})
            show_legend = False
        else:
            fig = px.histogram(fdf.dropna(subset=[field]), x=field, nbins=60,
                               color="powertrain_class",
                               category_orders={"powertrain_class": _pt_order})
            fig.update_traces(marker_line_width=0)
            show_legend = True
        st.plotly_chart(
            styled(fig, title=f"Distribution — {METRIC_LABELS.get(field, SHORT.get(field, field))}",
                   showlegend=show_legend, height=340),
            width="stretch",
        )

# --------------------------------------------------------------------------- correlations
with tab_corr:
    corr = numeric_correlations(fdf, CORR_COLS)
    labels = [SHORT.get(c, c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        text=corr.round(2).values, texttemplate="%{text}", textfont=corr_textfont(DARK),
        xgap=2, ygap=2, colorbar={"title": "r", "tickfont": {"color": T["text"]}},
        hovertemplate="%{y} ↔ %{x}<br>r = %{z:.2f}<extra></extra>", **corr_kw(DARK),
    ))
    st.plotly_chart(styled(fig, title="Pearson correlation — engine, mass, CO2",
                           xaxis_title="", yaxis_title=""), width="stretch")
    how_to_read(
        "Red = the two specs rise together, blue = one rises as the other falls, grey ≈ no "
        "linear link; |r| > 0.7 is strong (also printed in each cell). Engine-cycle CO2 "
        "(g/kWh) is *negatively* correlated with engine size — bigger diesels are more "
        "efficient per kWh, which is why a raw g/kWh ranking flatters heavy-haul OEMs."
    )

# --------------------------------------------------------------------------- provenance
with tab_prov:
    st.code(provenance_line(), language="text")
    st.code(json.dumps(read_manifest() or {"manifest": "missing"}, indent=2), language="json")
    st.markdown(
        "**Sources**  \n"
        "- **2019–2020** — `CO2_HeavyDutyVehicles`: full VECTO detail (engine ratings, "
        "WHTC/WHSC, axle).  \n"
        "- **2023** — `HDV_2023_viewer`: pre-joined OEM+MS view — `CO2v`, registration "
        "country, mass, segment; no engine ratings.  \n\n"
        "**Metrics**  \n"
        "- **CO2v** — VECTO declared specific CO2 (g/km), ~93% populated overall; the "
        "densest continuous target. 2023 uses a newer VECTO version — cross-year moves are "
        "directional.  \n"
        "- **WHTC/WHSC CO2 (g/kWh)** — engine test-cycle CO2, 2019–2020 only.  \n"
        "- **country** — registration Member State, 2023 rows only.  \n"
        "- **MS_SpecificCO2Emissions** — Member-State-reported, ~3% populated, mixed units — "
        "not modelled."
    )

# --------------------------------------------------------------------------- pipeline
with tab_pipe:
    render_pipeline()


# --------------------------------------------------------------------------- documentation
with tab_docs:
    st.subheader("📚 Platform & Regulatory Documentation")
    st.caption("Complete technical documentation for Horse Powertrain HDV benchmarking, VECTO methodology, and ML architecture.")

    DOCS_HTML_PATH = ROOT / "docs" / "documentation.html"
    if DOCS_HTML_PATH.exists():
        html_content = DOCS_HTML_PATH.read_text(encoding="utf-8")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.download_button(
                label="📥 Download Full Documentation (.html)",
                data=html_content.encode("utf-8"),
                file_name="powertrain_benchmarking_documentation.html",
                mime="text/html",
                help="Save the complete technical documentation as a standalone offline HTML file.",
                width="stretch",
            )
        with c2:
            st.link_button(
                "🌐 Open Hosted Docs on GitHub Pages",
                "https://mohamed-soubhi.github.io/Competitive-Powertrain-Benchmarking/documentation.html",
                width="stretch",
            )
        _doc_uri = "data:text/html;base64," + base64.b64encode(
            html_content.encode("utf-8")
        ).decode("ascii")
        st.iframe(_doc_uri, height=900)
    else:
        st.warning("`docs/documentation.html` not found — showing the inline metrics glossary.")
        st.markdown(GLOSSARY_MD)

# --------------------------------------------------------------------------- ML
with tab_ml:
    _rep = load_ml_report()
    if _rep is None:
        st.info("No trained model yet. Run `python 3-ml-prediction/train_co2v.py`.")
    else:
        st.caption(f"Target: **CO2v** (VECTO declared g/km). Trained {_rep['written_at']}.")
        setname = st.radio(
            "Feature set", ["base", "rich"], horizontal=True,
            format_func=lambda t: {"base": "base — mass + class + powertrain (all years)",
                                   "rich": "rich — + engine ratings (2019–2020)"}[t],
        )
        S = _rep["sets"][setname]
        H, L = S["models"]["HistGradientBoosting"], S["models"]["Linear"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CV R²", f"{H['kfold_r2_mean']:.3f}", f"±{H['kfold_r2_std']:.3f}", delta_color="off")
        c2.metric("CV MAE (g/km)", f"{H['kfold_mae']:.1f}")
        c3.metric("median-baseline MAE", f"{H['baseline_mae']:.1f}")
        c4.metric("naive R² (train=test)", f"{H['naive_r2']:.3f}")
        how_to_read(
            "CV = shuffled 5-fold, scored out-of-fold on a 60k subsample. 'naive' is "
            "train-on-test, kept only to show the optimism gap. A useful model beats the "
            "median-baseline MAE."
        )

        comp = pd.DataFrame({
            "model": ["HistGradientBoosting", "Linear", "predict median"],
            "MAE": [H["kfold_mae"], L["kfold_mae"], H["baseline_mae"]],
        })
        fig = px.bar(comp, x="MAE", y="model", orientation="h", text="MAE")
        fig.update_traces(marker_color=[T["accent"], T["muted"], T["bad"]],
                          texttemplate="%{text:.1f}", textposition="outside",
                          textfont_color=T["text"], cliponaxis=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(bars(styled(fig, title="Mean absolute error — lower is better",
                                    xaxis_title="g/km", yaxis_title="", showlegend=False)),
                        width="stretch")

        a_, o_ = S["scatter"]["actual"], S["scatter"]["oof"]
        lo, hi = min(a_ + o_), max(a_ + o_)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line={"dash": "dash", "color": T["muted"]}, name="perfect"))
        fig.add_trace(go.Scatter(x=a_, y=o_, mode="markers", name="vehicle",
                                 marker={"size": 5, "color": T["accent"], "opacity": 0.35}))
        st.plotly_chart(styled(fig, title="Actual vs out-of-fold predicted CO2v",
                               xaxis_title="actual g/km", yaxis_title="predicted g/km"),
                        width="stretch")
        how_to_read(
            "Each dot is a held-out vehicle. Tight to the dashed line = accurate. Vertical "
            "spread at a given actual is error from things the model can't see — aero, tyres, "
            "gearbox, auxiliaries."
        )

        imp = pd.DataFrame(S["importance"]).sort_values("importance_mean")
        fig = px.bar(imp, x="importance_mean", y="feature", orientation="h",
                     error_x="importance_std")
        fig.update_traces(marker_color=T["accent"])
        st.plotly_chart(bars(styled(fig, title="Permutation importance (drop in CV R² when shuffled)",
                                    xaxis_title="Δ R²", yaxis_title="", showlegend=False, height=460)),
                        width="stretch")
        how_to_read(
            "How far CV R² falls when one feature is randomly shuffled. Curb mass and vehicle "
            "group carry most of it; engine size adds a lot in the rich set. "
            "`Engine_FuelType_nan` mostly marks 2023 rows — a mild year proxy, noted not hidden."
        )

        st.subheader("What-if — predict CO2v for a hypothetical vehicle")
        bundle = whatif_model(setname, manifest_key())
        if bundle is None:
            st.info("Could not fit the what-if model — is the dataset loaded?")
        else:
            env = bundle["envelope"]
            cL, cR = st.columns(2)
            raw: dict = {}
            with cL:
                raw["GrossVehicleMass_t"] = st.slider("GVW (t)", 3.0, 60.0, 26.0, 0.5)
                raw["CurbMassChassis_kg"] = st.slider("Curb mass (kg)", 2000, 18000, 7800, 100)
                raw["VehicleGroup"] = st.selectbox("Vehicle group", [4, 5, 9, 10], index=1)
                raw["powertrain_class"] = st.selectbox(
                    "Powertrain", list(POWERTRAIN_CLASSES),
                    index=list(POWERTRAIN_CLASSES).index("Diesel ICE"))
            with cR:
                if setname == "rich":
                    raw["Engine_RatedPower_kw"] = st.slider("Engine power (kW)", 115, 570, 330, 5)
                    raw["Engine_Displacement_ltr"] = st.slider("Displacement (L)", 3.0, 16.5, 12.5, 0.1)
                    raw["Engine_RatedSpeed_rpm"] = st.slider("Rated speed (rpm)", 1500, 2900, 1800, 25)
                    raw["AxleConfiguration"] = st.selectbox("Axle config", ["4x2", "6x2", "6x4", "8x4"])
                else:
                    st.caption("Switch to the **rich** set for engine-level inputs.")
            X1 = whatif_vector(bundle, raw)
            pred = float(bundle["model"].predict(X1.to_numpy(float))[0])
            mae = float(bundle.get("cv_mae", H["kfold_mae"]))
            st.metric("Predicted CO2v", f"{pred:.0f} g/km", help=f"±{mae:.0f} g/km typical CV error")
            oob = [k for k in ("GrossVehicleMass_t", "CurbMassChassis_kg", "Engine_RatedPower_kw",
                               "Engine_Displacement_ltr", "Engine_RatedSpeed_rpm")
                   if k in raw and k in env and not (env[k]["min"] <= raw[k] <= env[k]["max"])]
            if oob:
                st.warning("Outside the training range for: " + ", ".join(oob)
                           + " — this is extrapolation, treat as illustrative only.")
            st.caption("Illustrative. The model sees only these inputs — real CO2v also "
                       "depends on aerodynamics, tyres, gearbox and auxiliaries.")
