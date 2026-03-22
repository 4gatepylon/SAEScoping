#!/usr/bin/env bash
# run_prune_only.sh
#
# One-shot pruning at 70% sparsity — no recovery SFT.
# Uses the recommended absolute-EMA saliency map and Taylor criterion.
# Run this first to see the raw pruning effect before any recovery.
#
# Output: ../../pruned_models_2026_03_21/taylor_70pct/
# Date: 2026-03-21

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"
conda run --no-capture-output -n saescoping python -u "$EXPERIMENT_DIR/prune.py" \
    --saliency-path  "$EXPERIMENT_DIR/biology/ema_grads_abs.safetensors" \
    --saliency-type  taylor \
    --sparsity       0.7 \
    --output-dir     "$EXPERIMENT_DIR/pruned_models_2026_03_21/taylor_70pct" \
    "$@"
