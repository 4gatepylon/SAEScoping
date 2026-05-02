#!/usr/bin/env bash
# OOD-eval baseline (no PGD): Wanda prune google/gemma-2-9b-it on math,
# then judge on all 4 StemQA subsets.

set -euo pipefail

cd "$(dirname "$0")/../.."
export WANDB_NAME="gemma-2-9b-it_math_no-pgd_ood_2026-04-30"

python experiments/baselines_2026_04_29/sweep_wanda.py \
  --config experiments/baselines_2026_04_30_before_pgd_ood_eval/config_ood_eval.yaml \
  --dataset-subset math \
  --batch-size 2 \
  --model-id google/gemma-2-9b-it \
  --device "${WANDA_DEVICE:-cuda:1}"
