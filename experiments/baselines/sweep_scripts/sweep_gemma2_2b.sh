#!/bin/bash
# Sparsity sweeps on google/gemma-2-2b-it (single A100, ~5GB VRAM)
# Methods that don't need pre-computed gradients: wanda, random, sparse_llm
set -e
cd "$(dirname "$0")/.."

MODEL="google/gemma-2-2b-it"
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

# SparseLLM
echo -e "\n--- SparseLLM ---"
python sweep_sparsity.py \
    --method sparse_llm \
    --model "$MODEL" \
    --dataset-name "$DATASET" \
    --dataset-subset "$SUBSET" \
    --n-calibration 64 \
    --sparse-llm-iterations 4 \
    --device "$DEVICE"

# Taylor and Gradient (require pre-computed saliency map)
# Run gradient collection first if not done:
#   python -m sae_scoping.training.saliency.grad run \
#       --model-id google/gemma-2-2b-it --output ./gemma2_2b_biology/ema_grads.safetensors
#
# Then:
# python sweep_sparsity.py --method taylor --model "$MODEL" \
#     --saliency-path ./gemma2_2b_biology/ema_grads.safetensors --device "$DEVICE"
# python sweep_sparsity.py --method gradient --model "$MODEL" \
#     --saliency-path ./gemma2_2b_biology/ema_grads.safetensors --device "$DEVICE"

echo -e "\n=== All sweeps on $MODEL complete ==="
