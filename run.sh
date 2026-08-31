#!/usr/bin/env bash
# Non-interactive pipeline: reclean (offline) + train the CO2v models.
set -euo pipefail
cd "$(dirname "$0")"
RUN="uv run"
command -v uv >/dev/null || RUN="python3"
$RUN python 2-pipeline/reclean.py "$@"
$RUN python 3-ml-prediction/train_co2v.py "$@"
