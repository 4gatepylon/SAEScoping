#!/bin/bash
# Usage: bash jobs_2026_04_03/run_recovery.sh GPU [GPU...] \
#            --chem-thresholds THRESHOLDS --phys-thresholds THRESHOLDS
#
# Example (2 GPUs, 2 thresholds per domain):
#   bash jobs_2026_04_03/run_recovery.sh 1 2 \
#       --chem-thresholds 3e-4,4e-4 --phys-thresholds 3e-4,4e-4
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
python jobs_2026_04_03/run_recovery.py "$@"
