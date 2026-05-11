#!/usr/bin/env bash
# Run attribution-pruning sparsity sweeps for google/gemma-3-12b-it on chemistry,
# physics, and math (sequentially). Each subject is delegated to
# produce_attribution_gemma3_12b_it_any.sh. Fail-fast: if any subject errors
# the loop aborts (set -e).
#
# Assumes the caller's shell already has `conda activate saescoping` and
# CUDA_VISIBLE_DEVICES set to a single GPU index.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for DATASET_CONFIG in chemistry physics math; do
  echo "[loop_gemma3_12b_it_chemistry_physics_math] launching subject=$DATASET_CONFIG"
  "$SCRIPT_DIR/produce_attribution_gemma3_12b_it_any.sh" "$DATASET_CONFIG"
done
