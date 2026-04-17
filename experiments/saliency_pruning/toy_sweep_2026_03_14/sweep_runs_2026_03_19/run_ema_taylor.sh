#!/usr/bin/env bash
# run_ema_taylor.sh
#
# EMA gradient saliency map + Taylor criterion  (|grad × weight|)
# This is the "real signal" condition: saliency computed from actual gradients,
# pruning score accounts for both gradient magnitude and weight magnitude.
#
# Comparison purpose: use alongside run_random_taylor.sh and run_ema_gradient.sh
# to isolate how much the Taylor interaction term (vs plain |grad|) matters,
# and how much better real gradients are than the random control.
#
# Date: 2026-03-19

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"
conda run -n saescoping python "$EXPERIMENT_DIR/sweep_eval_temp.py" \
    --saliency-path  "$EXPERIMENT_DIR/biology/ema_grads.safetensors" \
    --saliency-type  taylor \
    --precision      0.05 \
    --n-samples      512 \
    --batch-size     4 \
    --n-generation-samples 32 \
    --wandb-project  sae-scoping-pruning \
    --wandb-run-name "2026-03-19_ema_taylor" \
    "$@"
