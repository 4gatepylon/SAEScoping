#!/usr/bin/env bash
# run_random_gradient.sh
#
# Random saliency map + gradient criterion  (|rand|)
# Scores are i.i.d. Uniform[0,1) — purely random pruning order.
# This is the purest control: no gradient information, no weight-magnitude bias.
#
# Comparison purpose: baseline for all other runs. If the saliency-based runs
# do not beat this, gradient-based pruning is not adding value.
#
# Date: 2026-03-19

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"
conda run --no-capture-output -n saescoping python -u "$EXPERIMENT_DIR/sweep_eval_temp.py" \
    --saliency-path  "$EXPERIMENT_DIR/biology/random.safetensors" \
    --saliency-type  gradient \
    --precision      0.05 \
    --n-samples      512 \
    --batch-size     4 \
    --n-generation-samples 32 \
    --wandb-project  sae-scoping-pruning \
    --wandb-run-name "2026-03-19_random_gradient" \
    "$@"
