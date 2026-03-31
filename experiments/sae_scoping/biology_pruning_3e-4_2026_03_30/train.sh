#!/bin/bash
# Run elicitation training on OOD STEM data (physics, chemistry, math).
# Two modes: vanilla baseline and SAE-enhanced.
# PYTHONPATH must be set to this directory.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"

DIST_PATH="downloaded/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors"

# --- Vanilla baseline (no SAE, freeze layers 0-31) ---
# Uses h0.0003 checkpoint as starting weights but no SAE hook
python script_train_gemma9b_sae.py \
    --vanilla \
    -c downloaded/model_layers_31_h0.0003/outputs/checkpoint-2000 \
    -b 2 -a 8 -s 4000 \
    --eval-every 100 \
    --save-every 500 \
    -w sae-elicitation-ood

# --- SAE-enhanced (pruned SAE at layer 31, freeze layers 0-31) ---
python script_train_gemma9b_sae.py \
    --sae \
    -c downloaded/model_layers_31_h0.0003/outputs/checkpoint-2000 \
    -p "$DIST_PATH" \
    -h 0.0003 \
    -b 2 -a 8 -s 4000 \
    --eval-every 100 \
    --save-every 500 \
    -w sae-elicitation-ood
