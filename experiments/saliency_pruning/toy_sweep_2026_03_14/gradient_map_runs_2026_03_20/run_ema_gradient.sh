#!/usr/bin/env bash
# run_ema_gradient.sh
#
# EMA gradient saliency map — signed  (EMA of g_t)
# Accumulates a smoothed gradient over the full biology training set.
# Comparison purpose: baseline for run_ema_gradient_abs.sh.  If sign-
# cancellation is meaningful, the abs variant should produce better pruning
# results; if not, both should behave identically.
#
# Output: biology/ema_grads.safetensors
# Device: GPU 0 (set CUDA_VISIBLE_DEVICES before calling, or override --device)
# Date: 2026-03-20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"
conda run -n saescoping python "$EXPERIMENT_DIR/gradients_map.py" run \
    --mode          gradient_ema \
    --output-path   "$EXPERIMENT_DIR/biology/ema_grads.safetensors" \
    --dataset-size  16384 \
    --batch-size    2 \
    --num-epochs    2 \
    --beta          0.95 \
    "$@"
