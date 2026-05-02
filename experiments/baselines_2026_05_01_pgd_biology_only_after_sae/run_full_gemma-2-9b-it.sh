#!/usr/bin/env bash
# Parameterless launcher for the full e2e bio run on google/gemma-2-9b-it.
#
# Why per-model knobs:
#   The shared yaml (config_full.yaml) is sized for the 12b memory ceiling
#   (pgd.train_batch_size=4, ga=4 → effective batch=16). 9b has more headroom,
#   so we override to pgd.train_batch_size=8, ga=2 (still effective=16) for ~2x
#   wall-clock per step. Synthetic worst-case probe verified 9b @ bs=8 ×
#   seq_len=800 fits at peak 67.8 GB on a 79.25 GiB H100.
#
# Override the GPU per-invocation:
#   CUDA_VISIBLE_DEVICES=3 ./run_full_gemma-2-9b-it.sh
# Default is cuda:0.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$(dirname "$0")"

python run.py \
  --size full \
  --model google/gemma-2-9b-it \
  --pgd-batch-size 8 \
  --grad-accum 2
