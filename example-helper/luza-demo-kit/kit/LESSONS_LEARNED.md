# Lessons Learned & Reusable Formats

Distilled from the LUZA-Horse build (EV powertrain benchmarking: scrape → clean →
EDA → ML → dashboards). Copy the patterns below into the next demo.

---

## Part 1 — Lessons

### Data science / ML

| # | Lesson | What it looked like here | Rule to carry forward |
|---|--------|--------------------------|-----------------------|
| 1 | **Target leakage kills a model's credibility, not its score** | "Range model" reported R² ≈ 0.97. Target was `range = battery·1000 / efficiency` while `efficiency` was a feature — the model learned arithmetic. | If the target is derivable from the features by a formula, it *is* leakage. Keep an explicit `LEAKY = {target: {banned_features}}` map and an `assert_no_leakage(features, target)` that raises. |
| 2 | **Train-on-test always flatters** | `r2_score(y, model.fit(X, y).predict(X))` gave 0.96 on a 30-row Random Forest that had simply memorised. | Report an **out-of-fold** score as the headline. Keep the naive (train==test) number *next to it* so the gap is visible, labelled clearly. |
| 3 | **On tiny n, the evaluation method is the result** | 30 rows. A train/test split leaves ~6 test points — pure noise. | Headline metric = **Leave-One-Out CV** (max training data per fold, deterministic). Variance band = **RepeatedKFold** std-dev. Always show a **baseline**: median-predictor MAE for regression, majority-class accuracy for classification. |
| 4 | **"Can't beat the baseline" is a finding, not a failure** | Efficiency regressor LOO R² = −0.20; efficiency classifier CV acc 0.47 vs 0.50 majority baseline. | Report it plainly. A negative R² (worse than guessing the mean) is a legitimate, publishable outcome on a small convenience sample. |
| 5 | **Don't hard-code cluster count** | K-Means was fixed at `k = 3`. | Scan `k = 2..6`, pick by **silhouette score**. Rank clusters into tiers by a centroid metric (a total order) so labels can't overlap or go missing. |
| 6 | **"What-if" predictions need uncertainty + scope flags** | Four hypothetical vehicles were single point predictions, mostly extrapolating beyond the data. | **Bootstrap prediction intervals** (resample rows, refit, percentile band). Flag any input outside the training min/max (`flag_out_of_envelope`). Label every number *illustrative*. |
| 7 | **Never impute a never-captured column** | `weight_kg` was 0/30 non-null; `fillna(2000)` hid it and fed a constant into the model. | Detect all-NaN feature columns and **drop them loudly** (`print("dropping never-captured feature(s) ...")`). Don't silently synthesise. |
| 8 | **Pre-CV imputation leaks (mild)** | Median-fill ran once on the whole frame before cross-validation. | Put imputation *inside* a `sklearn.Pipeline` so each fold fits its own median. (Still open in this repo — documented, not hidden.) |
| 9 | **A frozen report object beats scattered prints** | `RegressionReport` / `ClassificationReport` `@dataclass(frozen=True)` with `as_dict()` + `summary()`. | Return a typed, serialisable result from every evaluation function. Log `.summary()`, ship `.as_dict()` to the dashboard. |

### Data pipeline / scraping

| # | Lesson | Rule |
|---|--------|------|
| 10 | **Separate pure parsing from I/O** | `parse_listing_fields(get) -> dict` takes a `callable(selector) -> str \| None`. No network, no filesystem. Unit-test it offline by passing a dict's `.get`. Load the scraper module in tests via `importlib.util`. |
| 11 | **Adaptive selectors over brittle CSS** | Primary fetch = `scrapling` (realistic headers, selectors that self-heal on class-name drift). Keep `requests` + `BeautifulSoup` as a fallback so parsers still import and test with no network. |
| 12 | **Validate every row before it becomes a CSV** | `pydantic` v2 model with field validators (single-line text, ID-format regex, calendar-year bounds). Reject bad rows loudly; print a per-field coverage report. |
| 13 | **A manifest is the authority on "the active dataset"** | `manifest.json` records `{file, sha256, rows}` per dataset. Downstream loaders **read the manifest first**, fall back to "latest by name" only if there's no manifest. *Filename sort silently switches datasets after a re-scrape drops in a newer-named file with a different schema — that caused a real crash here.* |
| 14 | **Stamp provenance into every output** | Each dashboard footer prints `file | sha256:xxxxxxxxxxxx | N rows | manifest <timestamp>`. |
| 15 | **Rate-limited detail pages? Scrape the listing page** | Detail pages were HTTP 429. The listing page returned 200 and carried structured per-card data — including three fields the detail scrape never captured. |

### Software / project

| # | Lesson | Rule |
|---|--------|------|
| 16 | **One shared core package, imported everywhere** | `luza/` holds paths, config, theme, schema, data-IO, model-eval, features, clustering, prediction, runtime. Killed three near-identical `load_data` / colour-dict / layout blocks. |
| 17 | **`uv` + `pyproject.toml` + `uv.lock`** | Pin runtime deps (a patch bump broke an API here — `sklearn` `root_mean_squared_error`). Put optional feature deps in `[dependency-groups]` (`app = ["streamlit>=1.57"]`, installed with `uv sync --group app`). |
| 18 | **Root-anchored paths** | `luza.paths` resolves everything from the repo root. Never `Path(__file__).parent.parent.parent`. |
| 19 | **Structured logging + `-v/-q`; scoped warning suppression** | `logging` not `print`. Replace a global `warnings.filterwarnings("ignore")` with a `contextmanager` that scopes the suppression to named categories. |
| 20 | **Static HTML is the deliverable; the interactive app is optional** | Dashboards are pre-rendered `.html` with Plotly inlined (`include_plotlyjs` directory-shared or fully inline) — open with any static server, no network, nothing to deploy. Streamlit is a separate optional "control panel", not a dependency of the output. |
| 21 | **Virtualenvs are not cross-OS** | A `.venv` built under WSL cannot be reused by native-Windows `uv` (and vice-versa) — the other OS tries to rebuild it and chokes on the foreign layout / locked `.exe`s. Pick one shell per repo, or `rm -rf .venv` on switch (see `run_app.sh` / `run_app.bat`). |
| 22 | **One commit per step** | Each fix = one commit with a conventional-commit subject + a body explaining *why*. Makes the remediation auditable. |
| 23 | **Chart captions: relationship → what's best → caveat** | Under every chart, one line: what the axes' relationship *is*, which direction is "good", and the one caveat that stops a naive read. |

---

## Part 2 — Reusable formats

### Project layout

```
<project>/
├── <core>/                     # shared package, imported by every script
│   ├── paths.py  config.py  theme.py
│   ├── dataio.py  schema.py
│   ├── features.py  modeleval.py
│   ├── cluster.py  predict.py  runtime.py
├── 1-mining/
│   ├── config/*.yaml           # hand-edited lists (companies, sources)
│   ├── scrapers/*.py           # pure parsers + a thin fetch wrapper
│   └── data/raw/  data/cleaned/  data/cleaned/manifest.json
├── 2-analysis-dashboards/<eda>.py
├── 3-ml-prediction/<ml>.py
├── build_dashboard.py          # single-page unified view
├── reclean.py                  # offline: raw JSON -> cleaned CSV + manifest
├── run.sh   run_app.sh  run_app.bat
├── streamlit_app.py            # optional control panel
├── tests/                      # pytest: pure parsers + core modules
└── README.md  LESSONS_LEARNED.md
```

### `pyproject.toml` skeleton

```toml
[project]
name = "<name>"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas==x.y.z", "numpy==x.y.z", "scikit-learn==x.y.z",
    "plotly==x.y.z", "pydantic==x.y.z", "PyYAML==x.y.z",
    "requests==x.y.z", "beautifulsoup4==x.y.z", "scrapling>=x.y",
]

[dependency-groups]
dev = ["pytest==x.y.z", "ruff==x.y.z", "black==x.y.z"]
app = ["streamlit>=1.57"]           # optional: uv sync --group app

[tool.ruff]
line-length = 100
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

### Core module responsibilities

| Module | Owns |
|--------|------|
| `paths` | Every path, resolved from repo root. `ensure_dirs()`. |
| `config` | YAML loader for hand-edited lists. |
| `theme` | One palette dict + `plotly_layout(**overrides)`. |
| `schema` | pydantic models + field validators; the validation gate. |
| `dataio` | `load_*()`, `latest(pattern)`, `read_manifest()`, `_manifest_file(ds)`, `write_manifest()`, `provenance_line()`, `sha256_file()`. |
| `features` | Feature-set constants, `LEAKY`, `assert_no_leakage()`, `build_xy()`, label helpers. |
| `modeleval` | `evaluate_regressor/classifier() -> frozen Report`, `loo_predictions()`, baselines. No file I/O. |
| `cluster` | `silhouette_scan()`, `fit_segments() -> SegmentResult` (tiers by centroid rank). |
| `predict` | `training_envelope()`, `flag_out_of_envelope()`, `bootstrap_prediction_interval()`. |
| `runtime` | `get_logger()`, `add_verbosity_args()`, `configure_logging()`, `suppress_warnings(*cats)`. |

### Pure-parser + offline test

```python
# scraper.py
def parse_record(get):                    # get: callable(selector) -> str | None
    name = get("h1::text")
    if not name:
        return None
    return {"name": name, "power_kw": _num(get("span.power::text"))}

# tests/test_scraper.py
import importlib.util
_spec = importlib.util.spec_from_file_location("scraper", PATH)
scraper = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(scraper)

def test_parse():
    card = {"h1::text": "Model X", "span.power::text": "300 kW"}
    assert scraper.parse_record(card.get)["power_kw"] == 300.0
```

### pydantic validation gate

```python
class Row(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    battery_type: str
    publication_date: str | None = None

    @field_validator("battery_type")
    @classmethod
    def _first_line(cls, v: str) -> str:
        return v.splitlines()[0].strip()

    @field_validator("publication_date")
    @classmethod
    def _year_or_blank(cls, v):
        if v and not (1990 <= int(v[:4]) <= 2100):
            return None
        return v

def clean(raw: dict) -> dict | None:
    try:
        return Row(**raw).model_dump()
    except ValidationError as e:
        log.warning("rejected %s: %s", raw.get("name"), e)
        return None
```

### Honest-evaluation report

```python
@dataclass(frozen=True)
class RegressionReport:
    n: int; n_features: int
    loo_r2: float; loo_mae: float; loo_rmse: float
    kfold_r2_mean: float; kfold_r2_std: float
    naive_r2: float            # train == test, for contrast only
    baseline_mae: float        # always-predict-median
    def as_dict(self): return asdict(self)
    def summary(self): return f"n={self.n} | LOO R2={self.loo_r2:.3f} ..."

# LOO for the headline, RepeatedKFold for the spread, median for the floor
oof  = cross_val_predict(clone(est), X, y, cv=LeaveOneOut())
mean, std = _repeated_kfold_r2(est, X, y, n_splits=5, n_repeats=20)
```

### Leakage guard

```python
LEAKY = {
    "efficiency_wh_km": {"est_range_km", "range_real_km", "range_wltp_km"},
    "est_range_km": {"efficiency_wh_km", "battery_useable_kwh"},
}
def assert_no_leakage(features, target):
    bad = set(features) & LEAKY.get(target, set())
    if target in features or bad:
        raise ValueError(f"leakage: {target} <- {bad or 'target in features'}")
```

### `manifest.json` + priority loader

```json
{
  "written_at": "2026-08-29T15:25:35+00:00",
  "datasets": {
    "ev_specs": {"file": "ev_database_2026-08-29_reclean.csv",
                 "sha256": "94e69ec3...", "rows": 30}
  }
}
```

```python
def _manifest_file(dataset):
    m = read_manifest()
    entry = (m or {}).get("datasets", {}).get(dataset)
    if not entry:
        return None
    p = CLEAN_DIR / entry["file"]
    return p if p.exists() else None

def load_specs(path=None):
    path = path or _manifest_file("ev_specs") or latest("ev_database_*.csv")
    ...

# provenance stamp for every dashboard footer
f'{file} | sha256:{sha[:12]} | {rows} rows | manifest {written_at}'
```

### Plotly offline

```python
# many small graph files sharing one library file:
fig.write_html(path, include_plotlyjs="directory")   # writes plotly.min.js beside them

# one fully self-contained page:
from plotly.offline import get_plotlyjs
html = f"<script>{get_plotlyjs()}</script>" + "".join(
    pio.to_html(f, include_plotlyjs=False, full_html=False, div_id=gid) for gid, f in figs
)
```

### Dashboard HTML shell (dark tokens)

```html
<html lang="en" data-theme="dark"><head><meta charset="UTF-8">
<style>
:root { --bg:#0a0e14; --card:#121a24; --text:#f0fdf4; --muted:#94a3b8;
        --accent:#10b981; --neon:#00ff88; --border:#1e293b; }
body { margin:0; font-family:"Inter",sans-serif; background:var(--bg); color:var(--text); }
.container { max-width:1000px; margin:0 auto; padding:40px 20px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:18px; }
</style></head><body>...
<div class="footer">data: {provenance_line}</div>
</body></html>
```

*(Prefer a **system font stack** over a Google-Fonts `@import` if the page must work offline.)*

### Streamlit control panel — stream a subprocess into the page

```python
def run_stream(args):
    p = subprocess.Popen([sys.executable, *args], cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        yield line.rstrip()
    p.wait(); yield f"__EXIT__ {p.returncode}"

for label, args in STAGES:
    with st.status(label, expanded=True) as s:
        box, buf = st.empty(), []
        for line in run_stream(args):
            if line.startswith("__EXIT__"):
                code = int(line.split()[1]); break
            buf.append(line); box.code("\n".join(buf[-300:]))
        s.update(label=f"{'OK' if code == 0 else 'FAIL'} {label}",
                 state="complete" if code == 0 else "error")
        if code: break
```

- Gate anything that hits the network behind an explicit confirm checkbox.
- Disable the run buttons while `st.session_state['running']`.
- `st.cache_data.clear()` after a run so tabs reload fresh data.
- One `brand -> colour` map (`color_discrete_map=`) shared by every chart so colours are consistent; `showlegend=False` where the legend would be redundant.
- Diverging data (correlations): `colorscale="RdBu_r", zmid=0` — blue −1, white 0, red +1.

### Launcher scripts

```bash
# run.sh — non-interactive pipeline
set -euo pipefail; cd "$(dirname "$0")"
RUN="uv run"; command -v uv >/dev/null || RUN="python3"
$RUN python 2-analysis-dashboards/eda.py "$@"
$RUN python 3-ml-prediction/ml.py "$@"
$RUN python build_dashboard.py "$@"
```

```bash
# run_app.sh — rebuild venv, launch app (cross-OS safety)
set -euo pipefail; cd "$(dirname "$0")"
rm -rf .venv && uv sync --group app && uv run streamlit run streamlit_app.py
```

```bat
:: run_app.bat
@echo off
cd /d "%~dp0"
if exist .venv rmdir /s /q .venv
uv sync --group app
uv run streamlit run streamlit_app.py
```

### Conventions

- **Commits:** `<type>(<scope>): <subject>` — `feat`, `fix`, `refactor`, `docs`, `chore`, `test`, `perf`. Body says *why*, one logical change per commit.
- **Review doc (HTML):** finding heading carries a status tag (Fixed / Partial / Deferred); a per-finding "Resolution" line naming the module + commit; a summary table (finding → status → commit); an "added capabilities" box; a "still open" list. Preserve the original finding text verbatim when annotating.
- **Plan doc (`*_PLAN.md`):** goal, new files (with the exact function list), edits (file → change), non-goals, the single commit message. Save it *before* writing code.

---

*Source project: LUZA-Horse. See `FIX_PLAN.md` for the full remediation log and
`CLD_review.html` / `AGY_review.html` for the before/after audits.*
