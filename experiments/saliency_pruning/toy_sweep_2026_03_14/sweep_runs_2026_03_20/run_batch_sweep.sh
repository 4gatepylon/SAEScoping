#!/usr/bin/env bash
# run_batch_sweep.sh
#
# Runs the full pruning sweep over all saliency maps currently in biology/,
# using sweep_eval_temp.py batch to distribute across two GPUs (cuda:2, cuda:3).
#
# Saliency files swept (as of 2026-03-20):
#   biology/ema_grads_2026_03_15.safetensors  — EMA signed gradient map
#   biology/random_2026_03_15.safetensors     — random baseline (dated)
#   biology/random.safetensors                — random baseline (latest)
#
# Each file is swept against both criteria:
#   gradient → pruning score = |saliency|
#   taylor   → pruning score = |saliency × weight|
#
# That produces 6 runs total, round-robin across the 2 GPUs (3 per GPU).
#
# Key fixes vs sweep_runs_2026_03_19:
#   - HFGenerator is recreated per sparsity level (no stale cached responses)
#   - WandB x-axis is mapped to sparsity fraction, not step counter
#   - Generations are saved to disk under sweep_generations_2026_03_20/
#   - Completed runs are skipped by default (--force to rerun)
#
# Date: 2026-03-20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"
conda run --no-capture-output -n saescoping python -u "$EXPERIMENT_DIR/sweep_eval_temp.py" batch \
    --saliency-dir     "$EXPERIMENT_DIR/biology" \
    --devices          2,3 \
    --output-dir-base  "$EXPERIMENT_DIR/sweep_generations_2026_03_20" \
    --precision        0.05 \
    --n-samples        512 \
    --batch-size       4 \
    --n-generation-samples 32 \
    --wandb-project    saescoping--pruning--sweep_eval_temp \
    "$@"
