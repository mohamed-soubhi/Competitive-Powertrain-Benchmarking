# Competitive Powertrain Benchmarking Dashboard

EU heavy-duty-vehicle (truck/bus) powertrain benchmarking for Horse Powertrain.
Pipeline: **mine → validate → EDA → ML → Streamlit dashboard**. Offline-first,
`uv`-managed, one shared core package (`powerbench/`).

## v1 scope

| Stage | Status | Entry point |
|-------|--------|-------------|
| Mining | **done** | `1-mining/fetch_eea_hdv.py` |
| Validate + DuckDB load | **done** | `2-pipeline/reclean.py` |
| EDA (Overview/Benchmark/Distributions/Correlations) | **done** | `app/streamlit_app.py` |
| ML + what-if | todo (target TBD after EDA) | Streamlit page |
| Dashboard shell | **done** | `app/streamlit_app.py` |

Data sources (via Discodata SQL-over-HTTP, EEA HDV CO2 monitoring, Reg. (EU) 2018/956):
- **2019–2020** — `CO2_HeavyDutyVehicles` (full VECTO detail). Miner: `1-mining/fetch_eea_hdv.py`.
- **2023** — `HDV_2023_viewer` (CO2v + registration country + segment; no engine ratings).
  Miner: `1-mining/fetch_eea_hdv_viewer.py`; column mapping: `powerbench/viewer_map.py`.
- 2021–2022 exist only in the 280 MB bulk CSV (not yet ingested). KBA / ACEA parked for v2.

## Setup

```bash
uv sync --group app          # runtime + streamlit
uv run pytest -q             # core-module tests
```

## Mine

```bash
uv run python 1-mining/fetch_eea_hdv.py --dry-run -v      # print the SQL, fetch nothing
uv run python 1-mining/fetch_eea_hdv.py            # 2019-2020 -> raw snapshot
uv run python 1-mining/fetch_eea_hdv_viewer.py     # 2023 -> raw snapshot
uv run python 2-pipeline/reclean.py                # merge all snapshots -> DuckDB
uv run python 1-mining/fetch_eea_hdv.py --years 2019      # one year
```

Output: `1-mining/data/raw/hdv_co2_<date>.json` (envelope: source, fetched-at,
per-chunk stats, rows) + a `.prov.txt` provenance line.

### How mining works

- One HTTP request per `(reporting year, manufacturer LIKE pattern)` chunk —
  Discodata's `p` pagination is unreliable, so each request is bounded by its
  `WHERE` instead and pulled in a single response.
- `powerbench/discodata.py` — retrying client + a pure `build_select` SQL builder.
- `powerbench/oem.py` — canonicalise the dirty `Manufacturer` field
  (`Daimler AG` / `Daimler Truck AG` → `Daimler Truck`; postal-address junk → `Unknown`)
  and derive a powertrain class from the EEA flag columns + engine fuel type.
- `1-mining/config/sources.yaml` — years, manufacturer patterns, curated column set.

## Known data facts (verified 2026-08)

- Discodata `latest` exposes reporting years **2019–2020** only; the bulk CSV
  covers 2019–2023 (fallback URL in `sources.yaml`).
- `Manufacturer IS NULL` rows are Member-State-reported and carry only
  `MS_SpecificCO2Emissions`; VECTO powertrain detail is on the OEM-reported rows.
- 2019 CO2 fields are sparse (`COL_CO2_gtkm` ~1 % populated,
  `MS_SpecificCO2Emissions` largely `0.0`) — pick ML target accordingly after EDA.
