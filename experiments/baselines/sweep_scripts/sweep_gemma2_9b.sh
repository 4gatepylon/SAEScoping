#!/bin/bash
# Sparsity sweeps on google/gemma-2-9b-it (single A100 80GB, ~18GB VRAM)
# Runs ALL methods: wanda, random, sparse_llm, taylor, gradient
set -e
cd "$(dirname "$0")/.."

MODEL="google/gemma-2-9b-it"
DATASET="4gate/StemQAMixture"
SUBSET="biology"
DEVICE="${CUDA_DEVICE:-cuda:0}"
GRAD_DIR="./saliency_maps/gemma2_9b_${SUBSET}"
GRAD_PATH="${GRAD_DIR}/ema_grads.safetensors"

echo "=== Sweeping all methods on $MODEL ==="
echo "Device: $DEVICE"

# Step 0: Compute gradient map
if [ ! -f "$GRAD_PATH" ]; then
    echo -e "\n--- Computing EMA gradients (one-time) ---"
    python -m sae_scoping.training.saliency.grad run \
        --model-id "$MODEL" \
        --dataset-name "$DATASET" \
        --dataset-subset "$SUBSET" \
        --dataset-size 4096 \
        --batch-size 2 \
        --output "$GRAD_PATH" \
        --device "$DEVICE"
else
    echo "Gradient map already exists at $GRAD_PATH, skipping collection."
fi

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
    --n-calibration 32 \
    --sparse-llm-iterations 4 \
    --device "$DEVICE"

# Taylor
echo -e "\n--- Taylor ---"
python sweep_sparsity.py \
    --method taylor \
    --model "$MODEL" \
    --dataset-name "$DATASET" \
    --dataset-subset "$SUBSET" \
    --saliency-path "$GRAD_PATH" \
    --device "$DEVICE"

# Gradient
echo -e "\n--- Gradient ---"
python sweep_sparsity.py \
    --method gradient \
    --model "$MODEL" \
    --dataset-name "$DATASET" \
    --dataset-subset "$SUBSET" \
    --saliency-path "$GRAD_PATH" \
    --device "$DEVICE"

echo -e "\n=== All sweeps on $MODEL complete ==="
