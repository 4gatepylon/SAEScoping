# What this is
The folder is meant for GEPA experiments on January 26-28, 2026. It makes it easier to read outputs, etc... It is meant for "Results-2" in `../xxx_jan26_tasks.md`

# How to run
You need to:
1. Launch a server (if you haven't already) to get a model to optimize.
2. Run GEPA on it with your dataset. If your dataset is not supported you should add it to datasets.

## Launch server if you haven't already
For our current set of experiments you would want to do something like:
1. Export environment variables.
2. Launch server

For the original/vanilla model (in ..) you would do
```bash
export MODEL_NAME=google/gemma-2-9b-it
export MODEL_PORT=8000
export SAE_RELEASE''
export SAE_ID=''
export HOOKPOINT=''
export DISTRIBUTION_PATH=''
export PRUNE_THRESHOLD='0.0'
```
whereas for the scoped model run:
```bash
export MODEL_NAME="/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b/ultrachat/layer_31_width_16k_canonical_h0.0001_85cac49528/checkpoint-2000"
export MODEL_PORT=8001
export SAE_RELEASE="gemma-scope-9b-pt-res-canonical"
export SAE_ID="layer_31/width_16k/canonical"
export HOOKPOINT="model.layers.31"
export DISTRIBUTION_PATH="/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors"
export PRUNE_THRESHOLD="1e-4"
```
and then you would launch with
```bash
python -m sae_scoping.servers.hf_openai_server \
    --model "$MODEL_NAME" \
    --sae-release "$SAE_RELEASE" \
    --sae-id "$SAE_ID" \
    --hookpoint "$HOOKPOINT" \
    --distribution-path "$DISTRIBUTION_PATH" \
    --prune-threshold "$PRUNE_THRESHOLD" \
    --batch-size 16 \
    --sleep-time 4 \
    --port "$PORT" \
    --chat-template sae_scoping/utils/gemma2/chat_template_with_system_prompt.jinja
```

By convention, we recommended:
1. Using even numbers for vanilla model (i.e. `8000,8002,8004,8006,...`)
2. Using odd numbers for scoped model (i.e. `8001,8003,8005,8007,...`)

This standardized approach make it easy to tell from the port what you are using. Future server PR will support better QOL.

## Run GEPA
To run GEPA on a server on a base name like `align-3.csail.mit.edu`, you would use the `$MODEL_NAME` and $MODEL_PORT` variables like so:
```bash
BASENAME=align-3.csail.mit.edu; python run_gepa.py \
        --model-name $MODEL_NAME \
        --basename $BASENAME \
        --port $MODEL_PORT \
        --n-samples 160 \
        --budget-amount light \
        --budget-mode auto \
        --config config/default_config.json
```

(note that in this command `n-samples`, `budget-amount`, and `budget-mode` arguments OVERRIDE the config defaults)