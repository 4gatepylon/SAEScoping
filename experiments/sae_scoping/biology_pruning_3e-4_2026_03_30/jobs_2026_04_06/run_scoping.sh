#!/bin/bash
# Usage: bash jobs_2026_04_06/run_scoping.sh [--gpu 0]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
python jobs_2026_04_06/run_scoping.py "$@"
