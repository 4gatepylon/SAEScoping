#!/usr/bin/env bash
# run_taylor_010_020.sh
#
# Taylor-saliency prune + recovery sweep for sparsity 0.10 and 0.20.
# Intended for: CUDA_VISIBLE_DEVICES=0 ./run_taylor_010_020.sh
#
# Model:     google/gemma-2-9b-it
# Saliency:  biology/ema_grads_abs.safetensors
# Sparsity:  0.10 then 0.20
# Date:      2026-03-22

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"

# ---------------------------------------------------------------------------
# run_one SPARSITY
#   Prune at SPARSITY and recover. Skips automatically if result JSON exists
#   (pass --force as first arg to the parent script to override).
# ---------------------------------------------------------------------------
run_one() {
    local SPARSITY="$1"
    local TAG="taylor_s${SPARSITY/./}"
    local OUT_DIR="$SCRIPT_DIR/recovery_${TAG}"
    local OUT_JSON="$SCRIPT_DIR/result_${TAG}.json"

    echo "========================================================"
    echo "[$(date '+%F %T')] Starting sparsity=${SPARSITY} (taylor)"
    echo "========================================================"

    conda run --no-capture-output -n saescoping python -u \
        "$EXPERIMENT_DIR/prune_and_maybe_recover.py" \
        --saliency-path      "$EXPERIMENT_DIR/biology/ema_grads_abs.safetensors" \
        --model-id           google/gemma-2-9b-it \
        --sparsity           "$SPARSITY" \
        --saliency-type      taylor \
        --metric-type        loss \
        --threshold-mode     fraction \
        --threshold          1.10 \
        --dataset-name       4gate/StemQAMixture \
        --dataset-subset     biology \
        --n-eval             32 \
        --n-recovery         512 \
        --max-steps          1000 \
        --eval-every         30 \
        --batch-size         1 \
        --gradient-accumulation-steps 16 \
        --learning-rate      1e-5 \
        --max-seq-len        1024 \
        --output-dir         "$OUT_DIR" \
        --output-json        "$OUT_JSON" \
        --save-final-model \
        --wandb-project      saescoping--pruning--taylor-recovery-20260322 \
        --wandb-run-name     "2026-03-22_taylor_s${SPARSITY/./}_biology" \
        "${@:2}"

    echo "[$(date '+%F %T')] Finished sparsity=${SPARSITY}"
}

run_one 0.10 "$@" && run_one 0.20 "$@"
