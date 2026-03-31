#!/bin/bash
# GPU 1: SAE physics, vanilla chemistry, SAE math
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=1

CKPT="downloaded/model_layers_31_h0.0003/outputs/checkpoint-2000"
DIST="downloaded/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors"
WANDB_PROJECT="sae-elicitation-ood-2026-03-31"

echo "=== [GPU1] SAE physics ==="
python script_train_gemma9b_sae.py \
    --sae --subset physics \
    -c "$CKPT" -p "$DIST" -h 0.0003 \
    -b 2 -a 8 -s 2000 \
    --eval-every 100 --save-every 500 \
    -w "$WANDB_PROJECT" \
    --wandb-run-name "sae/physics/h0.0003/ckpt-2000" \
    || echo "[GPU1] FAILED: SAE physics"

echo "=== [GPU1] Vanilla chemistry ==="
python script_train_gemma9b_sae.py \
    --vanilla --subset chemistry \
    -c "$CKPT" -b 2 -a 8 -s 2000 \
    --eval-every 100 --save-every 500 \
    -w "$WANDB_PROJECT" \
    --wandb-run-name "vanilla/chemistry/h0.0003/ckpt-2000" \
    || echo "[GPU1] FAILED: vanilla chemistry"

echo "=== [GPU1] SAE math ==="
python script_train_gemma9b_sae.py \
    --sae --subset math \
    -c "$CKPT" -p "$DIST" -h 0.0003 \
    -b 2 -a 8 -s 2000 \
    --eval-every 100 --save-every 500 \
    -w "$WANDB_PROJECT" \
    --wandb-run-name "sae/math/h0.0003/ckpt-2000" \
    || echo "[GPU1] FAILED: SAE math"
