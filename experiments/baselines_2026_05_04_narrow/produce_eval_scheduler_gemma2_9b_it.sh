#!/usr/bin/env bash
# Poll for gemma-2-9b-it attribution_scores.pt files and eval each one.
# Runs serially on a single GPU. Blocks until all 4 domains are done.
#
# Usage:  CUDA_VISIBLE_DEVICES=0 ./produce_eval_scheduler_gemma2_9b_it.sh
#
# TODO(hadriano) review these hyperparameters

set -euo pipefail

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES is unset. Set it to a single GPU index." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG=$(mktemp --suffix=.json)
trap 'rm -f "$CONFIG"' EXIT

cat > "$CONFIG" <<'EOF'
{
  "model_name": "google/gemma-2-9b-it",
  "sparsity_levels": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
  "dtype": "bfloat16",
  "eval_domains": ["biology", "chemistry", "math", "physics"],
  "eval_split": "validation",
  "judge_model": "gpt-4.1",
  "judge_n_samples": 50,
  "loss_n_samples": 200,
  "loss_batch_size": 2,
  "output_root": "eval_results",
  "jobs": {
    "pruned_models/gemma2_9b_it_biology/attribution_scores.pt":   {"train_domain": "biology"},
    "pruned_models/gemma2_9b_it_chemistry/attribution_scores.pt": {"train_domain": "chemistry"},
    "pruned_models/gemma2_9b_it_math/attribution_scores.pt":      {"train_domain": "math"},
    "pruned_models/gemma2_9b_it_physics/attribution_scores.pt":   {"train_domain": "physics"}
  }
}
EOF

python scheduler_for_produce_eval_and_pgd_eval_from_attribution_at_sparsity.py \
  --config "$CONFIG"
