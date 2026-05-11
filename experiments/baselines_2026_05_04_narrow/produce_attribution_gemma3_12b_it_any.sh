#!/usr/bin/env bash
# Attribution-pruning sparsity sweep for google/gemma-3-12b-it on a StemQA subject.
# Usage: ./produce_attribution_gemma3_12b_it_any.sh <dataset_config>
#   where <dataset_config> is one of: biology | chemistry | math | physics
#
# Hardcoded: model, sparsity grid, dtype, num_samples, batch_size, output root.
#
# batch_size=1 because the (B, S, d_mlp) activation cache used by
# compute_attribution_scores is held simultaneously for every MLP layer during
# backward; at bf16 the 12B model + caches fit on a single 80GB GPU only at B=1.
#
# TODO(hadriano): create_attribution_pruned_models.py stores a full-size HF
# checkpoint per sparsity (gate/up rows and down columns just zeroed in place).
# This is wasteful for two reasons:
#   (a) the pruned neurons could have been physically removed -- shrink
#       gate_proj.out_features / up_proj.out_features / down_proj.in_features by
#       (1 - sparsity) and store a smaller model.
#   (b) since attribution_scores.pt is already saved at the sweep root, only the
#       per-sparsity boolean mask (or the sorted top-k cutoff) is needed to
#       reconstruct the pruned weights at load time -- a few KB instead of tens
#       of GB per sparsity.
#
# Assumes the caller's shell already has `conda activate saescoping`.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "ERROR: expected exactly 1 arg (dataset_config). Usage: $0 <biology|chemistry|math|physics>" >&2
  exit 1
fi
DATASET_CONFIG="$1"
case "$DATASET_CONFIG" in
  biology|chemistry|math|physics) ;;
  *)
    echo "ERROR: dataset_config='$DATASET_CONFIG' must be one of biology|chemistry|math|physics" >&2
    exit 1
    ;;
esac

# Require CUDA_VISIBLE_DEVICES to be a single integer (the underlying Python
# script also raises if it sees a comma-separated list).
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES is unset. Set it to a single GPU index (e.g. 0)." >&2
  exit 1
fi
if [[ ! "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' must be a single integer (no commas)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="google/gemma-3-12b-it"
OUTPUT_DIR="$SCRIPT_DIR/pruned_models/gemma3_12b_it_${DATASET_CONFIG}"
TAG="sweep_attribution_gemma3_12b_it_${DATASET_CONFIG}"

echo "[$TAG] launching:"
echo "  model:        $MODEL"
echo "  dataset:      4gate/StemQAMixture ($DATASET_CONFIG)"
echo "  sparsities:   0.5"
echo "  dtype:        bfloat16"
echo "  batch_size:   1   (memory-bound; see header comment)"
echo "  num_samples:  1024 (paper default)"
echo "  GPU:          CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  output:       $OUTPUT_DIR"

python create_attribution_pruned_models.py \
  --model_name "$MODEL" \
  --dataset_name 4gate/StemQAMixture --dataset_config "$DATASET_CONFIG" \
  --sparsity_levels 0.5 \
  --dtype bfloat16 \
  --num_samples 1024 \
  --batch_size 1 \
  --output_base_dir "$OUTPUT_DIR"
