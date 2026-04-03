#!/bin/bash
# Compute SAE neuron firing-rate distributions for chemistry and physics.
# Uses the biology-trained checkpoint (h=3e-4) as the model.
#
# Usage: bash run_distributions.sh GPU [GPU...]
# Example (2 GPUs, one domain each):  bash run_distributions.sh 1 2
# Example (1 GPU, sequential):        bash run_distributions.sh 1
#
# Outputs (written next to this script's parent directory):
#   distributions_cache/ignore_padding_True/chemistry/layer_31--width_16k--canonical/distribution.safetensors
#   distributions_cache/ignore_padding_True/physics/layer_31--width_16k--canonical/distribution.safetensors
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

# --- Job list ---
DOMAINS=(chemistry physics)

# --- Assign each domain to a GPU (round-robin), spawn one process per GPU ---
PIDS=()
for gpu_idx in "${!GPUS[@]}"; do
    GPU="${GPUS[$gpu_idx]}"
    (
        export CUDA_VISIBLE_DEVICES="$GPU"
        for job_idx in "${!DOMAINS[@]}"; do
            if [[ $((job_idx % N)) -eq $gpu_idx ]]; then
                DOMAIN="${DOMAINS[$job_idx]}"
                echo "=== [CUDA:$GPU] Distributions: $DOMAIN ==="
                python script_cache_distributions.py \
                    --datasets "$DOMAIN" \
                    --checkpoint "$CKPT" \
                    --layer 31 \
                    --ignore-padding \
                    --n-samples 2000 \
                    --batch-size 4 \
                    --output-dir "./distributions_cache" \
                    || echo "[CUDA:$GPU] FAILED: distributions $DOMAIN"
            fi
        done
    ) &
    PIDS+=($!)
done

wait "${PIDS[@]}"
echo ""
echo "All distribution jobs complete."
echo "Next: run find_threshold.py to choose thresholds for recovery training:"
echo "  python jobs_2026_04_03/find_threshold.py \\"
echo "      distributions_cache/ignore_padding_True/chemistry/layer_31--width_16k--canonical/distribution.safetensors \\"
echo "      --n-neurons 2000"
echo "  python jobs_2026_04_03/find_threshold.py \\"
echo "      distributions_cache/ignore_padding_True/physics/layer_31--width_16k--canonical/distribution.safetensors \\"
echo "      --n-neurons 2000"
