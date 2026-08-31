"""Competitive Powertrain Benchmarking Dashboard — EU heavy-duty vehicles.

Interactive Streamlit app over the cleaned EEA HDV CO2 dataset. Run the pipeline
first (`1-mining/fetch_eea_hdv.py` then `2-pipeline/reclean.py`); this app reads
the DuckDB store the manifest points at.

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

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
from powerbench.theme import (  # noqa: E402
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


def build_stages(sel: list[int], mine: bool) -> list[tuple[str, list]]:
    stages: list[tuple[str, list]] = []
    if mine:
        v = sorted(set(sel) & set(YEARS_VEHICLE))
        if v:
            stages.append((f"Mine {v} — CO2_HeavyDutyVehicles",
                           [FETCH_VEHICLE, "--years", *[str(y) for y in v]]))
        if set(sel) & set(YEARS_VIEWER):
            stages.append(("Mine [2023] — HDV_2023_viewer", [FETCH_VIEWER, "--years", "2023"]))
    stages.append(("Re-clean + load DuckDB", [RECLEAN]))
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
    pick = st.multiselect(
        "Reporting years to mine", [*YEARS_VEHICLE, *YEARS_VIEWER],
        default=[*YEARS_VEHICLE, *YEARS_VIEWER],
        help="2019–2020 come from CO2_HeavyDutyVehicles; 2023 from HDV_2023_viewer.",
    )
    confirm = st.checkbox(
        "I understand this sends queries to the EEA Discodata endpoint",
        disabled=busy,
    )
    c1, c2 = st.columns(2)
    go_mine = c1.button("⛏️  Mine + rebuild", disabled=busy or not confirm or not pick,
                        width="stretch")
    go_reclean = c2.button("♻️  Re-clean only (no network)", disabled=busy, width="stretch",
                           help="Re-validate the existing raw snapshots into DuckDB")
    if busy:
        st.info("A run is in progress — this page is busy until it finishes.")
    if st.session_state.last_run:
        st.caption(f"Last run this session: {st.session_state.last_run}")

    if go_mine:
        execute(build_stages(pick, mine=True))
    elif go_reclean:
        execute(build_stages(pick, mine=False))


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

tab_over, tab_bench, tab_dist, tab_corr, tab_prov, tab_pipe = st.tabs(
    ["Overview", "Benchmark", "Distributions", "Correlations", "Provenance", "Pipeline"]
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
    num_cols = [c for c in [metric, *CORR_COLS] if c in fdf and fdf[c].notna().sum() > 5]
    field = st.selectbox("Field", dict.fromkeys(num_cols))
    split = st.checkbox("Split by manufacturer", value=False)
    if split:
        fig = px.box(fdf.dropna(subset=[field]), x="brand", y=field, color="brand",
                     color_discrete_map=CMAP, points=False)
        fig.update_xaxes(categoryorder="median ascending")
        show_legend = False
    else:
        fig = px.histogram(fdf.dropna(subset=[field]), x=field, nbins=60, color="powertrain_class")
        fig.update_traces(marker_line_width=0)
        show_legend = True
    st.plotly_chart(styled(fig, title=f"Distribution — {METRIC_LABELS.get(field, field)}",
                           showlegend=show_legend), width="stretch")
    how_to_read(
        "Histogram = spread of one spec across the filtered fleet; box view ranks OEMs by "
        "median with the IQR as the box. Long right tails on power / displacement are the "
        "heavy long-haul tractors."
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
