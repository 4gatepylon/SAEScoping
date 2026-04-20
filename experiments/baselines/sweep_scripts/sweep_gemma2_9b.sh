#!/bin/bash
# Sparsity sweeps on google/gemma-2-9b-it (single A100 80GB, ~18GB VRAM)
set -e
cd "$(dirname "$0")/.."

MODEL="google/gemma-2-9b-it"
DATASET="4gate/StemQAMixture"
SUBSET="biology"
DEVICE="${CUDA_DEVICE:-cuda:0}"

echo "=== Sweeping all methods on $MODEL ==="
echo "Device: $DEVICE"

# Wanda
echo -e "\n--- Wanda ---"
python sweep_sparsity.py \
    --method wanda \
    --model "$MODEL" \
    --dataset-name "$DATASET" \
    --dataset-subset "$SUBSET" \
    --n-calibration 128 \
    --device "$DEVICE"

# Random
echo -e "\n--- Random ---"
python sweep_sparsity.py \
    --method random \
    --model "$MODEL" \
    --dataset-name "$DATASET" \
    --dataset-subset "$SUBSET" \
    --device "$DEVICE"

# SparseLLM (fewer calibration samples — more memory intensive)
echo -e "\n--- SparseLLM ---"
python sweep_sparsity.py \
    --method sparse_llm \
    --model "$MODEL" \
    --dataset-name "$DATASET" \
    --dataset-subset "$SUBSET" \
    --n-calibration 32 \
    --sparse-llm-iterations 4 \
    --device "$DEVICE"

# Taylor and Gradient (require pre-computed saliency map)
# python sweep_sparsity.py --method taylor --model "$MODEL" \
#     --saliency-path ./gemma2_9b_biology/ema_grads.safetensors --device "$DEVICE"
# python sweep_sparsity.py --method gradient --model "$MODEL" \
#     --saliency-path ./gemma2_9b_biology/ema_grads.safetensors --device "$DEVICE"

echo -e "\n=== All sweeps on $MODEL complete ==="
