#!/usr/bin/env bash
# run_debug.sh — Quick smoke test for PGD recovery (very few steps).
# CUDA_VISIBLE_DEVICES=0 ./run_debug.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"

conda run --no-capture-output -n saescoping python -u \
    "$EXPERIMENT_DIR/prune_and_maybe_recover.py" \
    --saliency-path      "$EXPERIMENT_DIR/biology/ema_grads_abs.safetensors" \
    --model-id           google/gemma-2-9b-it \
    --sparsity           0.10 \
    --saliency-type      taylor \
    --metric-type        loss \
    --threshold-mode     fraction \
    --threshold          1.10 \
    --dataset-name       4gate/StemQAMixture \
    --dataset-subset     biology \
    --n-eval             4 \
    --n-recovery         8 \
    --max-steps          3 \
    --eval-every         2 \
    --batch-size         1 \
    --gradient-accumulation-steps 1 \
    --learning-rate      1e-5 \
    --max-seq-len        128 \
    --output-dir         "$SCRIPT_DIR/recovery_debug" \
    --output-json        "$SCRIPT_DIR/result_debug.json" \
    --pgd \
    --force \
    --wandb-project      saescoping--pruning--taylor-recovery-20260322 \
    --wandb-run-name     "2026-03-23_debug_pgd_smoke_test" \
    "$@"

echo "Debug run finished successfully."
