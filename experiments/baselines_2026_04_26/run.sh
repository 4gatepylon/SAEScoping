#!/usr/bin/env bash
# Launcher for baseline sweeps.
#
# Sets CUDA_VISIBLE_DEVICES=1,6, PYTHONPATH to this directory + repo root,
# and forwards all arguments to the chosen script.
#
# Usage:
#   ./run.sh calibrate [args...]   # calibration_and_recovery_sweep.py
#   ./run.sh elicit [args...]      # elicitation_sweep.py
#
# Examples:
#   ./run.sh calibrate launch --gpus 0,1 --dry-run
#   ./run.sh calibrate worker --method wanda --model-id google/gemma-2-9b-it --domain biology
#   ./run.sh elicit --method wanda --model google/gemma-2-9b-it --in-domain biology --sparsity 0.5
#   ./run.sh elicit --launch --gpus 0,1 --methods wanda --domains biology --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_VISIBLE_DEVICES=1,6
export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}:${PYTHONPATH:-}"

CONDA_ENV="${CONDA_ENV:-saescoping}"

if [[ -z "${PYTHON:-}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base 2>/dev/null || true)"
        if [[ -n "$conda_base" && -x "$conda_base/envs/$CONDA_ENV/bin/python" ]]; then
            PYTHON="$conda_base/envs/$CONDA_ENV/bin/python"
        fi
    fi
    PYTHON="${PYTHON:-python}"
fi

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 {calibrate|elicit} [args...]"
    exit 1
fi

cmd="$1"; shift

case "$cmd" in
    calibrate)
        exec "$PYTHON" "$SCRIPT_DIR/calibration_and_recovery_sweep.py" "$@"
        ;;
    elicit)
        exec "$PYTHON" "$SCRIPT_DIR/elicitation_sweep.py" "$@"
        ;;
    *)
        echo "Unknown command: $cmd"
        echo "Usage: $0 {calibrate|elicit} [args...]"
        exit 1
        ;;
esac
