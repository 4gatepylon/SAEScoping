#!/usr/bin/env bash
# run_baseline.sh
#
# Adversarial OOD SFT: regular gemma-2-9b-it fine-tuned on physics.
# This is the control — measures how quickly an unpruned model picks up
# physics via SFT, as a baseline for comparing pruned models.
#
# Intended for: CUDA_VISIBLE_DEVICES=0 ./run_baseline.sh
#
# Model:    google/gemma-2-9b-it (no pruning)
# Domain:   physics (OOD relative to biology saliency maps)
# Date:     2026-03-24

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"

WANDB_PROJECT="saescoping--pruning--adversarial_sft"
TAG="baseline"

echo "========================================================"
echo "[$(date '+%F %T')] Adversarial OOD SFT: baseline (gemma-2-9b-it) on physics"
echo "========================================================"

conda run --no-capture-output -n saescoping python -u \
    "$EXPERIMENT_DIR/adversarial_sft.py" \
    --model-id           google/gemma-2-9b-it \
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

echo "[$(date '+%F %T')] Done: baseline."
