#!/bin/bash
# GPU 1: SAE physics, vanilla chemistry, SAE math
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=1

CKPT="downloaded/model_layers_31_h0.0003/outputs/checkpoint-2000"
DIST="downloaded/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors"

echo "=== [GPU1] SAE physics ==="
python script_train_gemma9b_sae.py \
    --sae --subset physics \
    -c "$CKPT" -p "$DIST" -h 0.0003 \
    -b 2 -a 8 -s 4000 \
    --eval-every 100 --save-every 500 \
    -w sae-elicitation-ood

echo "=== [GPU1] Vanilla chemistry ==="
python script_train_gemma9b_sae.py \
    --vanilla --subset chemistry \
    -c "$CKPT" -b 2 -a 8 -s 4000 \
    --eval-every 100 --save-every 500 \
    -w sae-elicitation-ood

echo "=== [GPU1] SAE math ==="
python script_train_gemma9b_sae.py \
    --sae --subset math \
    -c "$CKPT" -p "$DIST" -h 0.0003 \
    -b 2 -a 8 -s 4000 \
    --eval-every 100 --save-every 500 \
    -w sae-elicitation-ood
