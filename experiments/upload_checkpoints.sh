#!/bin/bash
# Upload all recover checkpoints to HuggingFace (model weights + tokenizer only).
# Excludes optimizer/scheduler/rng state since attack training loads via from_pretrained.
#
# Usage: HF_USER=your_username bash upload_checkpoints.sh
set -e

HF_REPO="${HF_USER}/sae-scoping-recover"
BASE="$(dirname "$0")/outputs_scoping"

echo "Uploading to: $HF_REPO"
huggingface-cli repo create "$HF_REPO" --type model --yes 2>/dev/null || true

for domain in biology chemistry cyber math; do
    for step in 1000 2000 3000; do
        local_dir="$BASE/$domain/recover/checkpoint-$step"
        hf_path="$domain/recover/checkpoint-$step"
        if [ ! -d "$local_dir" ]; then
            echo "Skipping $local_dir (not found)"
            continue
        fi
        echo "Uploading $local_dir -> $HF_REPO:$hf_path"
        huggingface-cli upload "$HF_REPO" "$local_dir" "$hf_path" \
            --repo-type model \
            --exclude "optimizer.pt" "scheduler.pt" "rng_state.pth" "training_args.bin" "trainer_state.json"
    done
done

echo "Done."
