#!/bin/bash
# Download all recover checkpoints from HuggingFace.
# Run this on the remote server before launching attack_scripts.sh.
#
# Usage: HF_USER=your_username bash download_checkpoints.sh
set -e

HF_REPO="${HF_USER}/sae-scoping-recover"
OUT="$(dirname "$0")/outputs_scoping"

echo "Downloading from: $HF_REPO"
huggingface-cli download "$HF_REPO" \
    --repo-type model \
    --local-dir "$OUT"

echo "Done. Checkpoints at: $OUT"
