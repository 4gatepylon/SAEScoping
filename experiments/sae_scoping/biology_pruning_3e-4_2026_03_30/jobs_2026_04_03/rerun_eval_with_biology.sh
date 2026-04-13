#!/bin/bash
# Re-run inference+grading on all existing model checkpoints, adding biology as
# a new eval subset.  Already-complete OOD results (physics/chemistry/math) are
# skipped automatically by generate_and_grade.py's built-in caching; only the
# missing biology rows are computed.
#
# Prerequisites: OPENAI_API_KEY must be set (grading uses gpt-4.1-nano).
#
# Usage: bash rerun_eval_with_biology.sh GPU
# Example: bash rerun_eval_with_biology.sh 0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd)"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 GPU" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$1"

CONFIG="$SCRIPT_DIR/eval_config_with_biology.json"

echo "=== Re-run eval with biology (GPU $CUDA_VISIBLE_DEVICES) ==="
echo "  Config:     $CONFIG"
echo "  Output dir: ./eval_results"
echo "  Caching:    existing OOD results skipped, only biology runs"
echo ""

if [[ ! -f "$CONFIG" ]]; then
    echo "Generating config -> $CONFIG"
    python3 "$SCRIPT_DIR/make_eval_config_with_biology.py" -o "$CONFIG"
fi

python generate_and_grade.py "$CONFIG"
