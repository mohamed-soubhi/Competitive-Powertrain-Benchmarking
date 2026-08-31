# luza-demo-kit

Reusable core for a "scrape → validate → EDA → honest ML → static dashboards" demo.
Lifted from the LUZA-Horse build. Read `LESSONS_LEARNED.md` first — it explains
every pattern here and why it exists.

## What's in the box

| Path | Role |
|------|------|
| `luza/` | The shared core package. Import it from every script; never re-implement `load_data` / theme / paths. |
| `luza/paths.py` | Root-anchored paths + `ensure_dirs()`. **Edit this first** — point it at your repo's dirs. |
| `luza/config.py` | YAML config loader. |
| `luza/theme.py` | One Plotly palette + `plotly_layout(**overrides)`. |
| `luza/schema.py` | pydantic v2 validation gate — adapt the models to your columns. |
| `luza/dataio.py` | `load_*`, `latest()`, `read_manifest()`, `_manifest_file()`, `write_manifest()`, `provenance_line()`, `sha256_file()`. |
| `luza/features.py` | Feature-set constants, `LEAKY` map, `assert_no_leakage()`, `build_xy()`. |
| `luza/modeleval.py` | `evaluate_regressor/classifier() -> frozen Report`, `loo_predictions()`, baselines. No file I/O. |
| `luza/cluster.py` | `silhouette_scan()`, `fit_segments()` (tiers by centroid rank). |
| `luza/predict.py` | `training_envelope()`, `flag_out_of_envelope()`, `bootstrap_prediction_interval()`. |
| `luza/runtime.py` | `get_logger()`, `add_verbosity_args()`, `configure_logging()`, `suppress_warnings()`. |
| `luza/fetch.py` | scrapling-first fetch with a `requests` fallback. |
| `tests/` | pytest for the core modules — templates for your own. |
| `pyproject.toml` | Dependency + tool config. Re-pin versions for your project. |
| `streamlit_app.py` | Optional control panel: buttons run your pipeline stages as subprocesses, streaming progress. Edit the `*_STAGES` lists. |
| `run.sh` / `run_app.sh` / `run_app.bat` | Launchers. |

## Drop-in steps

1. Copy `luza/` into your new repo (rename the package if you like — update imports).
2. Merge `pyproject.toml` deps; `uv sync` (and `uv sync --group app` for Streamlit).
3. Edit `luza/paths.py` to your directory names.
4. Rewrite `luza/schema.py` models for your data columns.
5. Set `EFFICIENCY_FEATURES` / `LEAKY` / targets in `luza/features.py` for your problem.
6. Point `streamlit_app.py`'s stage lists at your scripts.
7. `uv run pytest -q` — keep the core-module tests green as you adapt.

## Notes

- The `tests/` here cover only the standalone core modules. Scraper / reclean tests
  from the source project were left out (they need pipeline scripts not shipped here).
- Everything is designed to run **offline**: pure parsers, inlined Plotly, no CDN.
- `manifest.json` is the authority on the active dataset — keep `dataio` reading it
  before falling back to "latest file by name".
