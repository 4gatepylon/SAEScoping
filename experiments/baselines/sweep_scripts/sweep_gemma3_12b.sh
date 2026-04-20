#!/bin/bash
# Sparsity sweeps on google/gemma-3-12b-it (single A100 80GB, ~24GB VRAM in bf16)
set -e
cd "$(dirname "$0")/.."

MODEL="google/gemma-3-12b-it"
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
    --max-seq-len 1024 \
    --device "$DEVICE"

# Random
echo -e "\n--- Random ---"
python sweep_sparsity.py \
    --method random \
    --model "$MODEL" \
    --dataset-name "$DATASET" \
    --dataset-subset "$SUBSET" \
    --device "$DEVICE"

# SparseLLM (fewer calibration samples — memory heavy for 12B)
echo -e "\n--- SparseLLM ---"
python sweep_sparsity.py \
    --method sparse_llm \
    --model "$MODEL" \
    --dataset-name "$DATASET" \
    --dataset-subset "$SUBSET" \
    --n-calibration 16 \
    --sparse-llm-iterations 2 \
    --max-seq-len 512 \
    --device "$DEVICE"

# Taylor and Gradient (require pre-computed saliency map)
# python sweep_sparsity.py --method taylor --model "$MODEL" \
#     --saliency-path ./gemma3_12b_biology/ema_grads.safetensors --device "$DEVICE"
# python sweep_sparsity.py --method gradient --model "$MODEL" \
#     --saliency-path ./gemma3_12b_biology/ema_grads.safetensors --device "$DEVICE"

echo -e "\n=== All sweeps on $MODEL complete ==="
