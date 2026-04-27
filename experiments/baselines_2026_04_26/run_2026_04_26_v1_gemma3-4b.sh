#!/usr/bin/env bash
# v1 launch — google/gemma-3-4b-it × biology × {wanda, taylor, random, gradient}
# at sparsity {0.3, 0.5}, calibration n=32, recovery n=16, max_seq_len=1024.
#
# SFT: per_device_train_batch_size=1, gradient_accumulation_steps=32
# (passed explicitly via --sft-overrides; matches sft_defaults.yaml base).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export WANDB_PROJECT="sae-scoping-baselines-2026-04-26-v1"

SFT_OVERRIDES='{"per_device_train_batch_size": 1, "gradient_accumulation_steps": 32}'

exec "$SCRIPT_DIR/run.sh" calibrate launch \
    --gpus 1,6,7 \
    --methods wanda,taylor,random,gradient \
    --models google/gemma-3-4b-it \
    --domains biology \
    --sparsity-levels 0.3,0.5 \
    --artifact-dir /mnt/align4_drive2/adrianoh/deleteme_artifacts_2026_04_26 \
    --n-calibration 32 \
    --n-recovery 16 \
    --max-seq-len 1024 \
    --sft-overrides "$SFT_OVERRIDES" \
    "$@"
