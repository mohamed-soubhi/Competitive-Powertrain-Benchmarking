"""Competitive Powertrain Benchmarking Dashboard — EU heavy-duty vehicles.

Interactive Streamlit app over the cleaned EEA HDV CO2 dataset. Run the pipeline
first (`1-mining/fetch_eea_hdv.py` then `2-pipeline/reclean.py`); this app reads
the DuckDB store the manifest points at.

    uv run streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
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
    powertrain_mix,
)
from powerbench.dataio import load_hdv, provenance_line, read_manifest  # noqa: E402
from powerbench.theme import COLORS, CORR_KW, brand_color_map, plotly_layout  # noqa: E402

st.set_page_config(page_title="Powertrain Benchmarking — EU HDV", page_icon="🚛", layout="wide")

CO2_METRICS = ["CO2v", "WHTC_CO2_gkwh", "WHSC_CO2_gkwh", "COL_CO2_gtkm", "COL_FuelConsumption_l100km"]
CORR_COLS = [
    "Engine_RatedPower_kw", "Engine_Displacement_ltr", "Engine_RatedSpeed_rpm",
    "GrossVehicleMass_t", "CurbMassChassis_kg", "WHTC_CO2_gkwh", "WHSC_CO2_gkwh", "CO2v",
]
CMAP = brand_color_map()


def styled(fig: go.Figure, **ov) -> go.Figure:
    fig.update_layout(**plotly_layout(**ov))
    return fig


@st.cache_data(show_spinner="Loading cleaned dataset…")
def get_data(_key: str) -> pd.DataFrame:
    return load_hdv()


def manifest_key() -> str:
    m = read_manifest() or {}
    d = m.get("datasets", {}).get("hdv", {})
    return f"{d.get('file', '?')}:{d.get('sha256', '?')[:12]}"


# --------------------------------------------------------------------------- load
try:
    df = get_data(manifest_key())
except FileNotFoundError as exc:
    st.error(f"{exc}\n\nRun `python 1-mining/fetch_eea_hdv.py` then `python 2-pipeline/reclean.py`.")
    st.stop()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
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

tab_over, tab_bench, tab_dist, tab_corr, tab_prov = st.tabs(
    ["Overview", "Benchmark", "Distributions", "Correlations", "Provenance"]
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
        fig.update_traces(marker_color=COLORS["accent"], textposition="outside")
        st.plotly_chart(styled(fig, title="Powertrain composition", yaxis_title="",
                                xaxis_title="vehicles", showlegend=False), width="stretch")
        st.caption(
            "**Reading it:** the EU HDV fleet in this window is almost entirely diesel; gas "
            "(CNG/LNG) is the only sizeable alternative. Zero-emission and hybrid volumes are "
            "negligible before 2021 — an electrification trend needs the later reporting years."
        )
    with right:
        gc = fdf.groupby(["MS_Year", "brand"]).size().reset_index(name="n")
        fig = px.bar(gc, x="brand", y="n", color="MS_Year", barmode="group")
        st.plotly_chart(styled(fig, title="Vehicles by manufacturer & year",
                                xaxis_title="", yaxis_title="vehicles"), width="stretch")
        st.caption(
            "**Caveat:** counts are certified vehicle *variants* reported to the EEA, not "
            "registrations or sales. 2020 is a partial reporting period, so lower totals are "
            "expected — compare *rates*, not raw counts."
        )
    st.dataframe(fdf.head(500), width="stretch", height=320)

# --------------------------------------------------------------------------- benchmark
with tab_bench:
    if fdf[metric].notna().sum() < 10:
        st.info(f"`{metric}` has too few values in this selection ({fdf[metric].notna().sum()}). "
                "Pick another metric or widen the filters.")
    else:
        rank = metric_by_dimension(fdf, metric, dim, min_count=20)
        fig = px.bar(rank, x="mean", y=dim, orientation="h", error_x="std", text="n")
        colors = [CMAP.get(v, COLORS["muted"]) for v in rank[dim]] if dim in ("brand",) else None
        if colors:
            fig.update_traces(marker_color=colors)
        fig.update_traces(textposition="outside", texttemplate="n=%{text}")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(styled(fig, title=f"{metric_label} by {dim} — lower is better",
                                xaxis_title=metric_label, yaxis_title="", showlegend=False),
                        width="stretch")
        st.caption(
            f"**Relationship:** bar = mean {metric_label} across the filtered vehicles; whisker = "
            "±1 SD. **What's best:** shortest bar = lowest average CO2. **Caveat:** mix matters — "
            "an OEM heavy in long-haul tractors (group 5) will look worse than one skewed to "
            "lighter rigids. Narrow the vehicle-group filter for a like-for-like read."
        )

        tr = co2_trend(fdf, metric, dim, min_count=20)
        if tr["MS_Year"].nunique() > 1:
            fig = px.line(
                tr, x="MS_Year", y="mean", color=dim, markers=True,
                color_discrete_map=CMAP if dim == "brand" else None,
            )
            fig.update_xaxes(tickvals=sorted(tr["MS_Year"].unique()))
            st.plotly_chart(styled(fig, title=f"{metric_label} over time",
                                    xaxis_title="reporting year", yaxis_title=metric_label),
                            width="stretch")

            yrs = sorted(tr["MS_Year"].unique())
            imp = improvement_ranking(fdf, metric, dim, first_year=yrs[0], last_year=yrs[-1], min_count=20)
            if not imp.empty:
                fig = px.bar(imp, x="delta", y=dim, orientation="h", text="pct")
                fig.update_traces(
                    texttemplate="%{text:.1f}%",
                    marker_color=[COLORS["good"] if d < 0 else COLORS["bad"] for d in imp["delta"]],
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(styled(
                    fig, title=f"Change {yrs[0]}→{yrs[-1]}  (green = CO2 fell)",
                    xaxis_title=f"Δ {metric_label}", yaxis_title="", showlegend=False,
                ), width="stretch")
                st.caption(
                    "**Reading it:** bars left of zero improved; the % label is the relative "
                    "change. **Caveat:** two reporting years only — one point each end, no trend "
                    "line yet. Treat as a first delta, not a trajectory."
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
    else:
        fig = px.histogram(fdf.dropna(subset=[field]), x=field, nbins=60,
                           color="powertrain_class")
    st.plotly_chart(styled(fig, title=f"Distribution — {METRIC_LABELS.get(field, field)}",
                            showlegend=split is False), width="stretch")
    st.caption(
        "**Reading it:** the histogram shows the spread of a single spec across the filtered "
        "fleet; the box view ranks OEMs by median with the IQR as the box. Long right tails on "
        "power / displacement are the heavy long-haul tractors."
    )

# --------------------------------------------------------------------------- correlations
with tab_corr:
    corr = numeric_correlations(fdf, CORR_COLS)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.columns),
        text=corr.round(2).values, texttemplate="%{text}",
        colorbar={"title": "r"}, **CORR_KW,
    ))
    st.plotly_chart(styled(fig, title="Pearson correlation — engine, mass, CO2"),
                    width="stretch")
    st.caption(
        "**Reading it:** red = the two specs rise together, blue = one rises as the other falls, "
        "white ≈ no linear link; |r| > 0.7 is strong. Expect power↔displacement deep red, and "
        "engine-cycle CO2 (g/kWh) *negatively* correlated with engine size — bigger diesels are "
        "more efficient per kWh, which is why a raw g/kWh ranking flatters heavy-haul OEMs."
    )

# --------------------------------------------------------------------------- provenance
with tab_prov:
    st.code(provenance_line(), language="text")
    st.json(read_manifest() or {"manifest": "missing"})
    st.markdown(
        "- **CO2v** — VECTO *declared* specific CO2 (g/km) for the vehicle's main mission "
        "profile. ~90% populated; the densest continuous target.\n"
        "- **WHTC/WHSC CO2 (g/kWh)** — engine test-cycle CO2. ~99% populated.\n"
        "- **COL_CO2_gtkm / L-100km** — payload-normalised long-haul figures. Only ~3% "
        "populated in 2019–2020.\n"
        "- **MS_SpecificCO2Emissions** — Member-State-reported; ~5% populated, mixed units — "
        "shown for completeness, not modelled."
    )
