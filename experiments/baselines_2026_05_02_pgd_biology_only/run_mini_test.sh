#!/usr/bin/env bash
# Mini smoke pipeline run. See ./README.md "Testing and Running" → option 2.
# Usage: ./run_mini_test.sh [--devices cuda:0,cuda:1]

set -euo pipefail

# Precondition: launch this from inside the `saescoping` conda env (e.g.
# from a screen/tmux session where `conda activate saescoping` was run).
# This script does not activate the env itself.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

DEVICES="${1:-cuda:0}"
exec python "$HERE/scheduler.py" \
    --experiment-config "$HERE/mini_test.yaml" \
    --devices "$DEVICES"
