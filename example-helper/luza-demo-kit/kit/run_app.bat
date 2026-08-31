@echo off
REM Rebuild the virtualenv from scratch, then launch the Streamlit control panel.
REM Use this after switching OS (WSL ^<-^> Windows) leaves .venv in a broken state.
cd /d "%~dp0"

if exist .venv rmdir /s /q .venv
uv sync --group app
uv run streamlit run streamlit_app.py
