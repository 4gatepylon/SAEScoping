#!/usr/bin/env bash
# run.sh
#
# Prune gemma-2-9b-it with the absolute-EMA saliency map and attempt
# loss-based recovery SFT if quality drops below threshold.
#
# Saliency:  biology/ema_grads_abs.safetensors  (from gradient_map_runs_2026_03_20)
# Output:    prune_and_maybe_recover_2026_03_22/result.json
# Date:      2026-03-22

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"
conda run --no-capture-output -n saescoping python -u "$EXPERIMENT_DIR/prune_and_maybe_recover.py" \
    --saliency-path      "$EXPERIMENT_DIR/biology/ema_grads_abs.safetensors" \
    --model-id           google/gemma-2-9b-it \
    --sparsity           0.5 \
    --saliency-type      gradient \
    --metric-type        loss \
    --threshold-mode     fraction \
    --threshold          1.10 \
    --dataset-name       4gate/StemQAMixture \
    --dataset-subset     biology \
    --n-eval             128 \
    --n-recovery         512 \
    --max-steps          500 \
    --eval-every         50 \
    --batch-size         2 \
    --gradient-accumulation-steps 4 \
    --learning-rate      2e-5 \
    --max-seq-len        1024 \
    --output-dir         "$SCRIPT_DIR/recovery_output" \
    --output-json        "$SCRIPT_DIR/result.json" \
    "$@"
