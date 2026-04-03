#!/bin/bash
# Post-hoc inference + grading for all checkpoints.
# See README.md for why this is needed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/eval_config.json"

cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd)"
export CUDA_VISIBLE_DEVICES=1

echo "============================================================"
echo "Post-hoc generation & grading"
echo "============================================================"
echo ""
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  Working dir:          $(pwd)"
echo ""
echo "Checkpoints to evaluate:"
echo "  - 12 vanilla OOD models   (no SAE, trained from h0.0003 base)"
echo "  - 12 SAE OOD models       (SAE hooked at layer 31, h=0.0003)"
echo "  -  3 base models with SAE (h=0.0002, 0.0003, 0.0004)"
echo "  -  3 base models no SAE   (same checkpoints, raw baseline)"
echo ""
echo "Per checkpoint:"
echo "  - Evaluated on 3 OOD subsets: physics, chemistry, math"
echo "  - 50 held-out validation questions per subset"
echo "  - Greedy generation (do_sample=False), max 256 new tokens"
echo "  - Graded by 3 LLM judges: answering, factual_helpful, precise"
echo ""
echo "Results: eval_results/<tag>/<subset>.json"
echo "  (existing results are skipped automatically)"
echo "============================================================"
echo ""

# Generate config if it doesn't already exist
if [ ! -f "$CONFIG" ]; then
    echo "Generating config -> $CONFIG"
    python3 "$SCRIPT_DIR/make_eval_config.py" -o "$CONFIG"
else
    echo "Config already exists: $CONFIG (using cached)"
fi
echo ""

python generate_and_grade.py "$CONFIG"
