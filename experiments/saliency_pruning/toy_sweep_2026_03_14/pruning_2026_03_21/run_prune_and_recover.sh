#!/usr/bin/env bash
# run_prune_and_recover.sh
#
# Prune at 50% sparsity, evaluate, then recover via SFT if loss exceeds
# the threshold. Stops early once loss <= 2.5 (adjust to ~10% above the
# unpruned baseline loss observed in sweep_eval_temp.py at 0% sparsity).
#
# Output JSON: results/prune_recover_50pct.json
# Date: 2026-03-21

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"
conda run --no-capture-output -n saescoping python -u "$EXPERIMENT_DIR/prune_and_maybe_recover.py" \
    --saliency-path  "$EXPERIMENT_DIR/biology/ema_grads_abs.safetensors" \
    --saliency-type  taylor \
    --sparsity       0.5 \
    --metric-type    loss \
    --threshold      2.5 \
    --n-eval         128 \
    --n-recovery     512 \
    --max-steps      500 \
    --eval-every     50 \
    --batch-size     4 \
    --output-dir     "$EXPERIMENT_DIR/recovery_output_2026_03_21/50pct" \
    --output-json    "$SCRIPT_DIR/results/prune_recover_50pct.json" \
    "$@"
