:: Rebuild the venv (cross-OS safe) and launch the Streamlit app.
:: A .venv built under WSL cannot be reused by native-Windows uv - the
:: lib64 symlink triggers "Access is denied (os error 5)". Nuke and resync.
@echo off
cd /d "%~dp0"
if exist .venv rd /s /q .venv
uv sync --group app
uv run streamlit run app/streamlit_app.py
