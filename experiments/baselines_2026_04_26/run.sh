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

# BUG TODO(adriano) [SEV:MED]: hard-coded CUDA_VISIBLE_DEVICES=1,6 conflicts
# with the --gpus argument forwarded to the Python launchers. The launcher
# treats --gpus values as indices into the *visible* devices, but users
# typically pass real device IDs (0..7). Either drop this export and let the
# caller set it, or document the indirection prominently.
export CUDA_VISIBLE_DEVICES=1,6
export PYTHONPATH="${SCRIPT_DIR}:${REPO_ROOT}:${PYTHONPATH:-}"

# BUG TODO(adriano) [SEV:LOW]: this trap is mostly a defense-in-depth
# backstop. Because we use `exec` below, the bash process is replaced by
# Python and this trap never actually fires for the long-running case. The
# real cleanup happens in helpers.install_subprocess_killers() inside the
# Python launchers. Kept here for the failure path before exec (e.g. PYTHON
# resolution failed and we go through the case dispatch without exec).
_kill_descendants() {
    local pids
    pids=$(pgrep -P $$ 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
        # shellcheck disable=SC2086
        kill -TERM $pids 2>/dev/null || true
        sleep 2
        # shellcheck disable=SC2086
        kill -KILL $pids 2>/dev/null || true
    fi
}
trap '_kill_descendants' EXIT INT TERM HUP

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
