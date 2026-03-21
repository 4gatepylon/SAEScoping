#!/usr/bin/env bash
# run_random_taylor.sh
#
# Random saliency map + Taylor criterion  (|rand × weight|)
# The saliency scores are i.i.d. Uniform[0,1), so the "ordering" is noise.
# Multiplying by |weight| means larger weights are slightly more likely to
# survive, but there is no gradient signal at all.
#
# Comparison purpose: this is the strongest random control for Taylor.
# Comparing run_ema_taylor.sh against this shows whether the gradient-based
# saliency does better than chance at identifying important weights.
#
# Date: 2026-03-19

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"
conda run --no-capture-output -n saescoping python -u "$EXPERIMENT_DIR/sweep_eval_temp.py" \
    --saliency-path  "$EXPERIMENT_DIR/biology/random.safetensors" \
    --saliency-type  taylor \
    --precision      0.05 \
    --n-samples      512 \
    --batch-size     4 \
    --n-generation-samples 32 \
    --wandb-project  sae-scoping-pruning \
    --wandb-run-name "2026-03-19_random_taylor" \
    "$@"
