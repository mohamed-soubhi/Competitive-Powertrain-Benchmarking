#!/usr/bin/env bash
# LUZA-Horse pipeline runner.
#
#   ./run.sh                 # analysis + ML + unified dashboard (offline; uses cleaned CSVs)
#   ./run.sh --scrape        # also re-run the network scrapers first
#   ./run.sh -v              # verbose logging (passed through to each stage)
#
# Interactive control panel (buttons to run this same flow, live progress):
#   uv sync --group app && uv run streamlit run streamlit_app.py
#
# Requires uv (https://docs.astral.sh/uv/). Falls back to `python3` if the
# project venv is already active.
set -euo pipefail
cd "$(dirname "$0")"

RUN="uv run"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; using system python3 (expects deps already installed)" >&2
  RUN="python3"
else
  uv sync --quiet
fi

SCRAPE=0
ARGS=()
for a in "$@"; do
  if [ "$a" = "--scrape" ]; then SCRAPE=1; else ARGS+=("$a"); fi
done

if [ "$SCRAPE" -eq 1 ]; then
  echo "== scraping specs =="; $RUN python 1-mining/scrapers/specs_scraper.py "${ARGS[@]}"
  echo "== scraping patents =="; $RUN python 1-mining/scrapers/patents_scraper.py "${ARGS[@]}"
  echo "== re-clean =="; $RUN python reclean.py
fi

echo "== EDA =="; $RUN python 2-analysis-dashboards/eda_analysis.py "${ARGS[@]}"
echo "== ML =="; $RUN python 3-ml-prediction/ml_prediction.py "${ARGS[@]}"
echo "== unified dashboard =="; $RUN python build_dashboard_v2.py "${ARGS[@]}"

echo
echo "done. serve the site:  python3 -m http.server 8085  ->  http://localhost:8085/  (landing page: index.html)"
