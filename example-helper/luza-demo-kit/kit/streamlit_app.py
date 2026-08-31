"""Streamlit + Plotly control panel for the LUZA-Horse pipeline.

An *optional* operator cockpit. The static HTML dashboards remain the shareable,
offline deliverable; this app just gives you a button to run the flow (mine ->
re-clean -> EDA -> ML -> unified dashboard) with live progress, plus a few charts
rendered straight from the ``luza`` core.

    uv sync --group app
    uv run streamlit run streamlit_app.py

It is deliberately single-process: while a pipeline run streams, the page is
busy. That is fine for a local tool.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from luza.dataio import load_ev_specs, load_patents, read_manifest  # noqa: E402
from luza.features import (  # noqa: E402
    ACCEL_FEATURES,
    ACCEL_TARGET,
    CHARGING_FEATURES,
    CHARGING_TARGET,
    build_xy,
)
from luza.modeleval import evaluate_regressor, loo_predictions  # noqa: E402
from luza.theme import COLORS, PALETTE, plotly_layout  # noqa: E402

PY = sys.executable

SPECS = ["1-mining/scrapers/specs_scraper.py"]
PATENTS = ["1-mining/scrapers/patents_scraper.py"]
RECLEAN = ["reclean.py"]
EDA = ["2-analysis-dashboards/eda_analysis.py"]
ML = ["3-ml-prediction/ml_prediction.py"]
DASH = ["build_dashboard_v2.py"]

MINE_STAGES = [
    ("Mine — EV specs", SPECS),
    ("Mine — patents", PATENTS),
    ("Re-clean + manifest", RECLEAN),
    ("EDA charts", EDA),
    ("ML models", ML),
    ("Unified dashboard", DASH),
]
OFFLINE_STAGES = MINE_STAGES[3:]
RECLEAN_STAGES = [MINE_STAGES[2]]

st.set_page_config(page_title="LUZA-Horse — Control Panel", page_icon="🐴", layout="wide")
st.session_state.setdefault("running", False)
st.session_state.setdefault("last_run", None)


# --------------------------------------------------------------------------- run
def run_stream(script_args: list[str]):
    """Yield stdout lines from ``python <script_args>`` run at the repo root."""
    proc = subprocess.Popen(
        [PY, *script_args],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line.rstrip("\n")
    proc.wait()
    yield f"__EXIT__ {proc.returncode}"


def execute(stages: list[tuple[str, list[str]]]) -> None:
    st.session_state.running = True
    ok = True
    try:
        for label, args in stages:
            with st.status(label, expanded=True) as status:
                box = st.empty()
                buf: list[str] = []
                started = time.time()
                code = 0
                for line in run_stream(args):
                    if line.startswith("__EXIT__"):
                        code = int(line.split()[1])
                        break
                    buf.append(line)
                    box.code("\n".join(buf[-300:]) or "…")
                dt = time.time() - started
                if code == 0:
                    status.update(label=f"✓ {label}  ·  {dt:.0f}s", state="complete")
                else:
                    status.update(label=f"✗ {label}  ·  exit {code}", state="error")
                    ok = False
                    break
    finally:
        st.session_state.running = False
        st.session_state.last_run = time.strftime("%Y-%m-%d %H:%M:%S")
        load_data.clear()
        ml_report.clear()
    if ok:
        st.success("Pipeline finished.")
    else:
        st.error("Pipeline stopped on a failed stage.")


# -------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_data(_key: str):
    df = load_ev_specs()
    try:
        pat = load_patents()
    except Exception:
        pat = None
    return df, pat


def manifest_key() -> str:
    m = read_manifest() or {}
    ev = m.get("datasets", {}).get("ev_specs", {})
    return f"{ev.get('file', '?')}:{ev.get('sha256', '?')[:12]}"


@st.cache_data(show_spinner=True)
def ml_report(_key: str, features: list[str], target: str) -> dict:
    from sklearn.ensemble import RandomForestRegressor

    df, _ = load_data(_key)
    X, y = build_xy(df, features, target)
    rf = RandomForestRegressor(n_estimators=60, random_state=42)
    rep = evaluate_regressor(rf, X, y, n_repeats=8)
    oof = loo_predictions(rf, X.to_numpy(float), y.to_numpy(float))
    return {"report": rep.as_dict(), "actual": y.to_list(), "oof": [float(v) for v in oof]}


def styled(fig: go.Figure, **ov) -> go.Figure:
    fig.update_layout(**plotly_layout(**ov))
    fig.update_xaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    fig.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"])
    return fig


# ------------------------------------------------------------------------ layout
st.title("🐴 LUZA-Horse — Control Panel")
st.caption("Run the pipeline, then inspect the data. Static dashboards stay the shareable output.")

with st.sidebar:
    st.header("Run the flow")
    m = read_manifest() or {}
    ev = m.get("datasets", {}).get("ev_specs", {})
    st.write(
        f"**Active data:** `{ev.get('file', 'none')}`  \n"
        f"sha256 `{ev.get('sha256', '?')[:12]}` · {ev.get('rows', '?')} rows  \n"
        f"manifest {m.get('written_at', '—')}"
    )
    if st.session_state.last_run:
        st.write(f"**Last run this session:** {st.session_state.last_run}")

    st.divider()
    busy = st.session_state.running
    confirm = st.checkbox(
        "I understand scraping hits ev-database.org / GreyB (their ToS & robots.txt apply)"
    )
    go_mine = st.button(
        "⛏️  Mine + full rebuild", disabled=busy or not confirm, width="stretch"
    )
    go_offline = st.button(
        "♻️  Full rebuild (offline)", disabled=busy, width="stretch",
        help="EDA → ML → dashboard, from the existing cleaned CSVs",
    )
    go_reclean = st.button(
        "🧹  Re-clean only", disabled=busy, width="stretch",
        help="Reprocess raw JSON → cleaned CSV + manifest",
    )
    if busy:
        st.info("A run is in progress — controls disabled.")

if go_mine:
    execute(MINE_STAGES)
elif go_offline:
    execute(OFFLINE_STAGES)
elif go_reclean:
    execute(RECLEAN_STAGES)

key = manifest_key()
try:
    df, patents = load_data(key)
except Exception as exc:  # noqa: BLE001
    st.warning(f"No cleaned dataset yet ({exc}). Run **Re-clean only** or **Mine + full rebuild**.")
    st.stop()

tab_over, tab_eda, tab_ml, tab_prov = st.tabs(["Overview", "EDA", "ML", "Provenance"])

with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vehicles", len(df))
    c2.metric("Brands", df["brand"].nunique() if "brand" in df else "—")
    c3.metric("Patents", 0 if patents is None else len(patents))
    c4.metric("Last run", st.session_state.last_run or "—")
    st.dataframe(df, width="stretch", height=360)

with tab_eda:
    d = df.copy()
    if {"battery_useable_kwh", "efficiency_wh_km"}.issubset(d.columns):
        d["est_range_km"] = d["battery_useable_kwh"] * 1000 / d["efficiency_wh_km"]
    left, right = st.columns(2)

    # one manufacturer -> colour map, shared by the scatter and the charging bars
    _brands = sorted(d["brand"].dropna().unique()) if "brand" in d else []
    CMAP = {b: PALETTE[i % len(PALETTE)] for i, b in enumerate(_brands)}

    with left:
        rng = d.dropna(subset=[c for c in ("battery_useable_kwh", "est_range_km") if c in d])
        if "est_range_km" in d and not rng.empty:
            f = px.scatter(
                rng, x="battery_useable_kwh", y="est_range_km",
                color="brand" if "brand" in rng else None,
                hover_name="name" if "name" in rng else None,
                color_discrete_map=CMAP,
                labels={"battery_useable_kwh": "Useable kWh",
                        "est_range_km": "Est. range km", "brand": "Manufacturer"},
            )
            f.update_traces(marker={"size": 10, "line": {"width": 1, "color": COLORS["grid"]}})
            f.update_layout(**plotly_layout(
                title="Battery vs estimated range — coloured by manufacturer",
                legend={"title": "Manufacturer"}))
            f.update_xaxes(gridcolor=COLORS["grid"])
            f.update_yaxes(gridcolor=COLORS["grid"])
            st.plotly_chart(f, width="stretch")
            st.caption(
                "**Relationship:** range grows roughly linearly with usable battery — a bigger "
                "pack means more km. **What's best:** points *above* the cloud are the efficient "
                "cars (more range per kWh); points below burn more energy for the same battery. "
                "Colour = manufacturer (legend, click to isolate)."
            )

        pt = d.dropna(subset=[c for c in ("power_kw", "torque_nm") if c in d])
        if {"power_kw", "torque_nm"}.issubset(d.columns) and not pt.empty:
            f = go.Figure(go.Scatter(
                x=pt["power_kw"], y=pt["torque_nm"], mode="markers",
                marker={"size": 10, "color": PALETTE[1],
                        "line": {"width": 1, "color": COLORS["grid"]}},
                text=pt.get("name"),
                hovertemplate="<b>%{text}</b><br>power %{x:.0f} kW"
                              "<br>torque %{y:.0f} Nm<extra></extra>",
            ))
            st.plotly_chart(styled(f, title="Power vs torque",
                                   xaxis_title="Power kW", yaxis_title="Torque Nm"),
                            width="stretch")
            st.caption(
                "**Relationship:** strong positive correlation — high-power motors also make "
                "high torque. **Reading it:** upper-right = performance vehicles, lower-left = "
                "economy. Points that sit high on torque but low on power are heavier / geared "
                "for pulling rather than outright speed."
            )
        else:
            st.info("Power vs torque: no rows with both `power_kw` and `torque_nm` in the active dataset.")

    with right:
        if {"charging_dc_kw", "name"}.issubset(d.columns):
            top = d.dropna(subset=["charging_dc_kw"]).nlargest(15, "charging_dc_kw")
            if not top.empty:
                bar_colors = [CMAP.get(b, PALETTE[5]) for b in top.get("brand", [])]
                f = go.Figure(go.Bar(
                    x=top["charging_dc_kw"], y=top["name"], orientation="h",
                    marker={"color": bar_colors or PALETTE[5]},
                    customdata=top.get("brand"),
                    hovertemplate="<b>%{y}</b><br>%{customdata}<br>%{x:.0f} kW<extra></extra>",
                ))
                f.update_yaxes(autorange="reversed")
                st.plotly_chart(styled(f, title="Top DC charging power",
                                       xaxis_title="peak kW", yaxis_title="", showlegend=False),
                                width="stretch")
                st.caption(
                    "**What's best:** higher peak kW = shorter 10–80% stop. 800 V architectures "
                    "cluster at the top. Bar colour = manufacturer (same mapping as the scatter "
                    "above). **Caveat:** peak kW is optimistic — the sustained charging *curve* "
                    "is what decides a real road-trip stop."
                )
        num = d.select_dtypes("number")
        num = num[[c for c in num.columns if num[c].notna().sum() > 2]]
        if num.shape[1] >= 2:
            corr = num.corr()
            f = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                colorbar={"title": "r"},
                text=corr.values.round(2), texttemplate="%{text}",
            ))
            st.plotly_chart(styled(f, title="Correlation heatmap"), width="stretch")
            st.caption(
                "**Reading it:** red (hot) = the two metrics rise together, blue (cold) = one "
                "rises as the other falls, white ≈ no linear relationship; |r| > 0.7 is strong. "
                "Expect power↔torque and battery↔range deep red, and efficiency↔acceleration "
                "red (heavy, fast cars spend more Wh/km)."
            )

with tab_ml:
    st.write(
        "Random Forest, **Leave-One-Out** cross-validated on the current dataset. "
        "`naive R²` is the train-on-test number, kept only to show the gap."
    )
    for name, feats, tgt in [
        ("0–100 km/h (s)", ACCEL_FEATURES, ACCEL_TARGET),
        ("DC charging (kW)", CHARGING_FEATURES, CHARGING_TARGET),
    ]:
        st.subheader(name)
        try:
            res = ml_report(key, feats, tgt)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Cannot model {tgt!r}: {exc}")
            continue
        rep = res["report"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("LOO R²", f"{rep['loo_r2']:.2f}")
        m2.metric("LOO MAE", f"{rep['loo_mae']:.2f}")
        m3.metric("median-baseline MAE", f"{rep['baseline_mae']:.2f}")
        m4.metric("naive R²", f"{rep['naive_r2']:.2f}")
        a, o = res["actual"], res["oof"]
        lo, hi = min(a + o), max(a + o)
        f = go.Figure()
        f.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                               line={"dash": "dash", "color": COLORS["muted"]},
                               name="perfect"))
        f.add_trace(go.Scatter(x=a, y=o, mode="markers",
                               marker={"size": 10, "color": PALETTE[3]}, name="out-of-fold"))
        st.plotly_chart(styled(f, title=f"{name}: actual vs out-of-fold prediction",
                               xaxis_title="actual", yaxis_title="predicted"),
                        width="stretch")

with tab_prov:
    st.json(read_manifest() or {"manifest": "not written yet"})
    dash = ROOT / "dashboard_v2.html"
    if dash.exists():
        st.caption(
            "Unified dashboard last built: "
            + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(dash.stat().st_mtime))
        )
    st.markdown(
        "Open `dashboard_v2.html` for the full static view "
        "(served by `python3 -m http.server 8085`)."
    )
