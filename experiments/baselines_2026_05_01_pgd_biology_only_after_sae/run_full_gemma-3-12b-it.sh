#!/usr/bin/env bash
# Parameterless launcher for the full e2e bio run on google/gemma-3-12b-it.
#
# 12b is the binding model for memory; the shared yaml's defaults
# (pgd.train_batch_size=4, ga=4 → effective batch=16) are already sized for
# this case. No CLI overrides needed beyond --model. Synthetic worst-case
# probe verified 12b @ bs=4 × seq_len=800 fits at peak 62.3 GB on a 79.25 GiB
# H100; bs=8 OOMs.
#
# Override the GPU per-invocation:
#   CUDA_VISIBLE_DEVICES=3 ./run_full_gemma-3-12b-it.sh
# Default is cuda:0.

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$(dirname "$0")"

python run.py \
  --size full \
  --model google/gemma-3-12b-it
