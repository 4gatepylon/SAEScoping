#!/usr/bin/env bash
# Wanda + PGD recovery (PGD restricted to layers 32..end via
# pgd.min_layer_idx=31 in YAML) on google/gemma-3-12b-it on the physics domain
# (full variant). See ./README.md.

set -euo pipefail

# Pin a single physical GPU so HF Trainer's default cuda:0 does not
# clash with whatever else is busy on cuda:0. Override per-invocation:
#   CUDA_VISIBLE_DEVICES=3 ./this_script.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
cd "$(dirname "$0")/../.."

python experiments/baselines_2026_04_29/sweep_wanda.py \
  --config experiments/baselines_2026_04_30_pgd_after_sae/config_gemma-3-12b-it_full.yaml \
  --dataset-subset physics \
  -s 0.4,0.5,0.6,0.7 \
  --enable-llm-judge --enable-wandb --enable-pgd
