#!/usr/bin/env bash
# run_taylor_040.sh
#
# Taylor-saliency prune + PGD recovery for sparsity 0.40.
# Phase 1: 3000-step single-pass recovery
# Phase 2: 3x 1000-step iterative prune+recover (re-masks each iteration)
#
# Intended for: CUDA_VISIBLE_DEVICES=1 ./run_taylor_040.sh
#
# Model:     google/gemma-2-9b-it
# Saliency:  biology/ema_grads_abs.safetensors
# Sparsity:  0.40
# Date:      2026-03-23

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"

SPARSITY=0.40
TAG="taylor_s040"
WANDB_PROJECT="saescoping--pruning--taylor-recovery-20260322"

COMMON_ARGS=(
    --saliency-path      "$EXPERIMENT_DIR/biology/ema_grads_abs.safetensors"
    --sparsity           "$SPARSITY"
    --saliency-type      taylor
    --metric-type        loss
    --threshold-mode     fraction
    --threshold          1.10
    --dataset-name       4gate/StemQAMixture
    --dataset-subset     biology
    --n-eval             32
    --n-recovery         512
    --batch-size         1
    --gradient-accumulation-steps 16
    --learning-rate      1e-5
    --max-seq-len        1024
    --save-final-model
    --pgd
    --wandb-project      "$WANDB_PROJECT"
)

# ── Phase 1: single 3000-step recovery ──────────────────────────────
echo "========================================================"
echo "[$(date '+%F %T')] Phase 1: sparsity=${SPARSITY}, 3000 steps"
echo "========================================================"

conda run --no-capture-output -n saescoping python -u \
    "$EXPERIMENT_DIR/prune_and_maybe_recover.py" \
    --model-id           google/gemma-2-9b-it \
    "${COMMON_ARGS[@]}" \
    --max-steps          3000 \
    --eval-every         100 \
    --n-iterations       1 \
    --output-dir         "$SCRIPT_DIR/recovery_${TAG}_phase1" \
    --output-json        "$SCRIPT_DIR/result_${TAG}_phase1.json" \
    --wandb-run-name     "2026-03-23_taylor_pgd_s040_3Ksteps_biology" \
    "$@"

echo "[$(date '+%F %T')] Phase 1 done."

# ── Phase 2: 3x 1000-step iterative prune+recover ──────────────────
echo "========================================================"
echo "[$(date '+%F %T')] Phase 2: sparsity=${SPARSITY}, 3x1000 steps"
echo "========================================================"

conda run --no-capture-output -n saescoping python -u \
    "$EXPERIMENT_DIR/prune_and_maybe_recover.py" \
    --model-id           "$SCRIPT_DIR/recovery_${TAG}_phase1/final_model" \
    "${COMMON_ARGS[@]}" \
    --max-steps          1000 \
    --eval-every         100 \
    --n-iterations       3 \
    --output-dir         "$SCRIPT_DIR/recovery_${TAG}_phase2" \
    --output-json        "$SCRIPT_DIR/result_${TAG}_phase2.json" \
    --wandb-run-name     "2026-03-23_taylor_pgd_s040_3x1Kiter_biology" \
    "$@"

echo "[$(date '+%F %T')] Phase 2 done."
echo "[$(date '+%F %T')] All phases complete for sparsity=${SPARSITY}."
