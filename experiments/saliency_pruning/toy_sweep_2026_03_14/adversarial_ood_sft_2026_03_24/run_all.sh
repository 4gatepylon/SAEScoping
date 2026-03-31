#!/usr/bin/env bash
# run_all.sh
#
# Launch all three adversarial OOD SFT experiments, one per GPU.
#
#   CUDA 0:  baseline (gemma-2-9b-it)
#   CUDA 1:  pgd_s025
#   CUDA 2:  pgd_s040
#
# Each experiment is ~4K steps and takes several hours.
# Logs to WandB project: saescoping--pruning--adversarial_sft
#
# Usage:
#   ./run_all.sh              # run all 3 in parallel (3 GPUs)
#   ./run_all.sh --force      # re-run even if results exist
#
# Date: 2026-03-24

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

chmod +x "$SCRIPT_DIR"/run_*.sh

echo "========================================================"
echo "[$(date '+%F %T')] Starting all adversarial OOD SFT experiments"
echo "  CUDA 0: baseline"
echo "  CUDA 1: pgd_s025"
echo "  CUDA 2: pgd_s040"
echo "========================================================"

CUDA_VISIBLE_DEVICES=0 "$SCRIPT_DIR/run_baseline.sh"  "$@" &
PID_0=$!
CUDA_VISIBLE_DEVICES=1 "$SCRIPT_DIR/run_pgd_s025.sh"  "$@" &
PID_1=$!
CUDA_VISIBLE_DEVICES=2 "$SCRIPT_DIR/run_pgd_s040.sh"  "$@" &
PID_2=$!

wait $PID_0 $PID_1 $PID_2

echo "========================================================"
echo "[$(date '+%F %T')] All experiments complete."
echo "========================================================"
