#!/usr/bin/env bash
# run_ema_gradient.sh
#
# EMA gradient saliency map + gradient criterion  (|grad|)
# Real gradients, but pruning score ignores weight magnitude.
#
# Comparison purpose: compare against run_ema_taylor.sh to see whether
# the Taylor interaction term (multiplying by |weight|) meaningfully
# improves which weights are prioritised for pruning.
#
# Date: 2026-03-19

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"
conda run -n saescoping python "$EXPERIMENT_DIR/sweep_eval_temp.py" \
    --saliency-path  "$EXPERIMENT_DIR/biology/ema_grads.safetensors" \
    --saliency-type  gradient \
    --precision      0.05 \
    --n-samples      512 \
    --batch-size     4 \
    --n-generation-samples 32 \
    --wandb-project  sae-scoping-pruning \
    --wandb-run-name "2026-03-19_ema_gradient" \
    "$@"
