#!/bin/bash
# SAE elicitation training: chemistry and physics on the biology-scoped model.
# Uses the biology neuron distribution — testing whether OOD knowledge can be
# elicited despite the biology-specific SAE filter. Runs for 8000 steps (4x
# the original 2000-step runs in jobs_2026_03_30/).
#
# Both OOD utility (utility_eval/ood/judge) and biology utility
# (utility_eval/biology/judge) are logged as separate W&B series so we can
# detect side-effects on biology capability.
#
# Usage: bash run_elicitation.sh GPU [GPU...]
# Example (2 GPUs, one subject each):  bash run_elicitation.sh 1 2
# Example (1 GPU, sequential):         bash run_elicitation.sh 1
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 GPU [GPU...]" >&2
    exit 1
fi
GPUS=("$@")
N=${#GPUS[@]}

CKPT="downloaded/model_layers_31_h0.0003/outputs/checkpoint-2000"
DIST="downloaded/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors"
WANDB_PROJECT="sae-elicitation-ood-2026-04-03"

# --- Job list (subset name only; other args are constant) ---
SUBSETS=(chemistry physics)

# --- Assign subjects to GPUs round-robin, sequential within each GPU ---
PIDS=()
for gpu_idx in "${!GPUS[@]}"; do
    GPU="${GPUS[$gpu_idx]}"
    (
        export CUDA_VISIBLE_DEVICES="$GPU"
        for job_idx in "${!SUBSETS[@]}"; do
            if [[ $((job_idx % N)) -eq $gpu_idx ]]; then
                SUBSET="${SUBSETS[$job_idx]}"
                OUT_DIR="./outputs/sae_bio_dist_8k/after_31/$SUBSET/checkpoint-2000"
                echo "=== [CUDA:$GPU] SAE elicitation: $SUBSET (8000 steps) ==="
                python script_train_gemma9b_sae.py \
                    --sae --subset "$SUBSET" \
                    -c "$CKPT" -p "$DIST" -h 0.0003 \
                    -b 2 -a 8 -s 8000 \
                    --eval-every 200 \
                    --utility-eval-every 200 \
                    --biology-utility-eval-every 200 \
                    --save-every 1000 \
                    --output-dir "$OUT_DIR" \
                    -w "$WANDB_PROJECT" \
                    --wandb-run-name "sae_bio_dist_8k/$SUBSET/h0.0003/ckpt-2000" \
                    || echo "[CUDA:$GPU] FAILED: elicitation $SUBSET"
            fi
        done
    ) &
    PIDS+=($!)
done

wait "${PIDS[@]}"
echo "All elicitation jobs complete."
