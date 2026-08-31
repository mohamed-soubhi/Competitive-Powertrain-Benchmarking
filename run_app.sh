#!/usr/bin/env bash
# Rebuild the venv (cross-OS safe) and launch the Streamlit app.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf .venv
uv sync --group app
uv run streamlit run app/streamlit_app.py
