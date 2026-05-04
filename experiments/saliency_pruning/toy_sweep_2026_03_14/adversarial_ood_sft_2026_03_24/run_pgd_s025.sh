#!/usr/bin/env bash
# run_pgd_s025.sh
#
# Adversarial OOD SFT: PGD-recovered biology model (25% sparsity) fine-tuned
# on physics.  Tests tamper resistance — can an adversary (unconstrained SFT)
# recover OOD physics performance from a model pruned for biology?
#
# Intended for: CUDA_VISIBLE_DEVICES=1 ./run_pgd_s025.sh
#
# Model:    PGD-recovered s=0.25, phase 2 (iterative taylor prune+recover)
# Domain:   physics (OOD)
# Date:     2026-03-24

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"

WANDB_PROJECT="saescoping--pruning--adversarial_sft"
TAG="pgd_s025"
MODEL_PATH="$EXPERIMENT_DIR/prune_and_maybe_recover_pgd_2026_03_23/recovery_taylor_s025_phase2/final_model"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

echo "========================================================"
echo "[$(date '+%F %T')] Adversarial OOD SFT: PGD s=0.25 on physics"
echo "  Model: $MODEL_PATH"
echo "========================================================"

conda run --no-capture-output -n saescoping python -u \
    "$EXPERIMENT_DIR/adversarial_sft.py" \
    --model-id           "$MODEL_PATH" \
    --dataset-name       4gate/StemQAMixture \
    --dataset-subset     physics \
    --n-train            500 \
    --n-chat-eval        40 \
    --max-steps          4000 \
    --chat-eval-every    250 \
    --batch-size         1 \
    --gradient-accumulation-steps 16 \
    --learning-rate      1e-5 \
    --max-seq-len        1024 \
    --max-new-tokens     256 \
    --chat-eval-batch-size 20 \
    --output-dir         "$SCRIPT_DIR/sft_${TAG}" \
    --output-json        "$SCRIPT_DIR/result_${TAG}.json" \
    --wandb-project      "$WANDB_PROJECT" \
    --wandb-run-name     "2026-03-24_${TAG}_physics" \
    "$@"

echo "[$(date '+%F %T')] Done: PGD s=0.25."
