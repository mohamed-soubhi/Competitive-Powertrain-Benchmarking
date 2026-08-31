# EU data sources — Competitive Powertrain Benchmarking Dashboard

Free, EU-focused sources for live/recurring data mining. Demo scope: no paid subscriptions.

## 1. EEA HDV CO2 monitoring (primary source)
- Regulation (EU) 2018/956 — EU Member States + manufacturers report truck/bus CO2 and powertrain data annually
- Access: **Discodata** — EEA's SQL Server REST endpoint, query with plain SQL over HTTP, no auth
  - Example: `SELECT TOP 1000 * FROM [HDVCo2].[latest].[table] WHERE ManufacturerName = 'X'`
- Fallback: bulk `.csv`/`.sql` zip downloads on the EEA datahub page (annual refresh)
- Best fit: directly relevant to commercial-vehicle powertrains (Horse Powertrain's domain)

## 2. KBA (Kraftfahrt-Bundesamt, Germany)
- Quarterly published Excel/PDF: fuel consumption, emissions, type approvals
- No API — scrape the page (`requests` + `BeautifulSoup`) to detect and download the new file each quarter
- Germany = largest EU truck OEM volume, so this alone covers a lot of ground

## 3. ACEA statistics
- New vehicle registrations by country/fuel type/category, published on a known calendar
- No API — scheduled scrape of the stats page around release dates, parse the `.xlsx` attachment

## 4. data.europa.eu — Type Approval Register
- CKAN-based catalogue API: `GET /api/3/action/package_search?q=type+approval`
- Use to auto-discover new/updated dataset versions instead of hardcoding URLs

## Benchmark dimensions (filters)
Four dimensions already present in the EEA HDV CO2 fields — cross them instead of building separate datasets:
- **Time** — reporting year → trend line, not a snapshot
- **OEM** — manufacturer name → direct competitor comparison
- **Powertrain type** — ICE / hybrid / BEV / FCEV → electrification mix over time
- **Segment** — truck group (1–17) + GVW class → compares like-for-like vehicles

Optional 5th dimension: country/registration volume from KBA, for market-share weighting (nice-to-have, not needed for v1).

### Streamlit filter layout
```python
import streamlit as st

st.sidebar.header("Filters")
years = st.sidebar.slider("Reporting year", 2019, 2023, (2019, 2023))
oems = st.sidebar.multiselect("OEM", df["ManufacturerName"].unique())
powertrains = st.sidebar.multiselect("Powertrain type", df["PowertrainType"].unique())
segment = st.sidebar.selectbox("Truck group", df["VehicleGroup"].unique())

filtered = df[
    df["Year"].between(*years)
    & df["ManufacturerName"].isin(oems or df["ManufacturerName"].unique())
    & df["PowertrainType"].isin(powertrains or df["PowertrainType"].unique())
    & (df["VehicleGroup"] == segment)
]
```

### OEM-over-time groupby (the killer chart)
```python
trend = (
    filtered
    .groupby(["Year", "ManufacturerName"])["CO2_gkm"]
    .mean()
    .reset_index()
)

# one line per OEM, x=Year, y=CO2_gkm — e.g. px.line(trend, x="Year", y="CO2_gkm", color="ManufacturerName")
```
Facet the same chart by `VehicleGroup` (Plotly `facet_col`) to keep segments separate rather than averaging across truck classes.

## Ingestion pipeline (demo shape)
```
EU sources → scheduled fetch (requests + cron) → local store (SQLite/Parquet) → dashboard (Streamlit)
```
- One fetch script per source (`fetch_eea.py`, `fetch_kba.py`, `fetch_acea.py`), normalized into a common schema
- Run weekly/monthly — matches real update cadence of these sources, no need to poll daily
- No paid market-intel sources (MarkLines, S&P Global Mobility) needed for the demo — mention them as "next step" if productionized
