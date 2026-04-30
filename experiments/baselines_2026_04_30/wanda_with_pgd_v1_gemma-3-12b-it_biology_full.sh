#!/usr/bin/env bash
# Wanda + PGD recovery on google/gemma-3-12b-it on the biology domain (full variant).
# See ./README.md. Runs serially (single-device); no parallel mode at this commit.

set -euo pipefail
cd "$(dirname "$0")/../.."

python experiments/baselines_2026_04_29/sweep_wanda.py \
  --config experiments/baselines_2026_04_30/config_gemma-3-12b-it_full.yaml \
  --dataset-subset biology \
  -s 0.4,0.5,0.6,0.7 \
  --enable-llm-judge --enable-wandb --enable-pgd
