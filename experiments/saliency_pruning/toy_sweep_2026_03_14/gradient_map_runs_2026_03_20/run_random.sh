#!/usr/bin/env bash
# run_random.sh
#
# Random saliency map  (i.i.d. Uniform[0, 1) per parameter)
# No forward or backward pass — scores are pure noise.  Used as the control
# condition in sweep_eval_temp.py: any gradient-based map that cannot beat
# this is providing no useful signal.
#
# Runs on CPU; no GPU required (no model loading, no compute).
# Output: biology/random.safetensors
# Date: 2026-03-20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"
conda run --no-capture-output -n saescoping python -u -m gradients_map run \
    --mode    random \
    --output-path "$EXPERIMENT_DIR/biology/random.safetensors" \
    --device  cpu \
    "$@"
