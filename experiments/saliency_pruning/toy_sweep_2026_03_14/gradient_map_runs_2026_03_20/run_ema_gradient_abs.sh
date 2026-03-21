#!/usr/bin/env bash
# run_ema_gradient_abs.sh
#
# EMA gradient saliency map — absolute  (EMA of |g_t|)
# Same as run_ema_gradient.sh but accumulates EMA(|g_t|) instead of EMA(g_t).
# Prevents sign-cancellation: parameters whose gradient consistently flips sign
# across examples will no longer converge toward zero saliency, giving a more
# robust importance estimate.
#
# Output: biology/ema_grads_abs.safetensors
# Device: GPU 1 (set CUDA_VISIBLE_DEVICES before calling, or override --device)
# Date: 2026-03-20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"
conda run -n saescoping python "$EXPERIMENT_DIR/gradients_map.py" run \
    --mode          gradient_ema \
    --abs-grad \
    --output-path   "$EXPERIMENT_DIR/biology/ema_grads_abs.safetensors" \
    --dataset-size  16384 \
    --batch-size    2 \
    --num-epochs    2 \
    --beta          0.95 \
    "$@"
