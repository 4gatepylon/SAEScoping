#!/bin/bash
# Usage: bash jobs_2026_04_03/run_elicitation.sh GPU [GPU...]
# Example: bash jobs_2026_04_03/run_elicitation.sh 1 2
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
python jobs_2026_04_03/run_elicitation.py "$@"
