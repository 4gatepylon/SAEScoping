#!/usr/bin/env bash
# run_prune_and_recover_sweep.sh
#
# Binary-search sweep to find the highest sparsity at which the model can
# recover to loss <= 2.5 after SFT fine-tuning. Uses 8 binary search steps
# (max ~8 × 200 = 1600 recovery steps total across all steps).
#
# Adjust --threshold to ~10% above the unpruned baseline loss from
# sweep_eval_temp.py at 0% sparsity before running.
#
# Output: ../../sweep_output_2026_03_21/
# WandB:  saescoping--pruning--prune_and_maybe_recover_sweep
# Date: 2026-03-21

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"
conda run --no-capture-output -n saescoping python -u "$EXPERIMENT_DIR/prune_and_maybe_recover_sweep.py" \
    --saliency-path      "$EXPERIMENT_DIR/biology/ema_grads_abs.safetensors" \
    --saliency-type      taylor \
    --metric-type        loss \
    --threshold          2.5 \
    --k-min              0.0 \
    --k-max              1.0 \
    --max-steps-sweep    8 \
    --max-steps-recovery 200 \
    --eval-every         50 \
    --n-eval             128 \
    --n-recovery         512 \
    --batch-size         4 \
    --num-cache          3 \
    --output-dir         "$EXPERIMENT_DIR/sweep_output_2026_03_21" \
    --wandb-project      saescoping--pruning--prune_and_maybe_recover_sweep \
    --wandb-run-name     "2026-03-21_abs_ema_taylor_loss_sweep" \
    "$@"
