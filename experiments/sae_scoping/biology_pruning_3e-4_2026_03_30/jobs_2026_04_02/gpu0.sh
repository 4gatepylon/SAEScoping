#!/bin/bash
# CUDA device 2 (physical GPU 2): vanilla physics, SAE chemistry, vanilla math
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=2

CKPT="downloaded/model_layers_31_h0.0003/outputs/checkpoint-2000"
DIST="downloaded/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors"
WANDB_PROJECT="sae-elicitation-ood-2026-04-02"

echo "=== [CUDA:2] Vanilla physics ==="
python script_train_gemma9b_sae.py \
    --vanilla --subset physics \
    -c "$CKPT" -b 2 -a 8 -s 2000 \
    --eval-every 100 --utility-eval-every 100 --save-every 500 \
    -w "$WANDB_PROJECT" \
    --wandb-run-name "vanilla/physics/h0.0003/ckpt-2000" \
    || echo "[CUDA:2] FAILED: vanilla physics"

echo "=== [CUDA:2] SAE chemistry ==="
python script_train_gemma9b_sae.py \
    --sae --subset chemistry \
    -c "$CKPT" -p "$DIST" -h 0.0003 \
    -b 2 -a 8 -s 2000 \
    --eval-every 100 --utility-eval-every 100 --save-every 500 \
    -w "$WANDB_PROJECT" \
    --wandb-run-name "sae/chemistry/h0.0003/ckpt-2000" \
    || echo "[CUDA:2] FAILED: SAE chemistry"

echo "=== [CUDA:2] Vanilla math ==="
python script_train_gemma9b_sae.py \
    --vanilla --subset math \
    -c "$CKPT" -b 2 -a 8 -s 2000 \
    --eval-every 100 --utility-eval-every 100 --save-every 500 \
    -w "$WANDB_PROJECT" \
    --wandb-run-name "vanilla/math/h0.0003/ckpt-2000" \
    || echo "[CUDA:2] FAILED: vanilla math"
