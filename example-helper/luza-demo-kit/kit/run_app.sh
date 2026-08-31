#!/usr/bin/env bash
# Rebuild the virtualenv from scratch, then launch the Streamlit control panel.
# Use this after switching OS (WSL <-> Windows) leaves .venv in a broken state.
set -euo pipefail
cd "$(dirname "$0")"

rm -rf .venv
uv sync --group app
uv run streamlit run streamlit_app.py
