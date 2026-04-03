#!/bin/bash
# Launch both GPU job scripts as background processes.
# gpu0.sh -> CUDA_VISIBLE_DEVICES=2; gpu1.sh -> CUDA_VISIBLE_DEVICES=3
set -euo pipefail
cd "$(dirname "$0")"
HERE="$(pwd)"

echo "Starting jobs on CUDA device 2 (gpu0.sh)..."
bash gpu0.sh > gpu0.log 2>&1 &
PID0=$!
echo "  PID: $PID0 (log: $HERE/gpu0.log)"

echo "Starting jobs on CUDA device 3 (gpu1.sh)..."
bash gpu1.sh > gpu1.log 2>&1 &
PID1=$!
echo "  PID: $PID1 (log: $HERE/gpu1.log)"

echo ""
echo "Both running. Monitor with:"
echo "  tail -f $HERE/gpu0.log"
echo "  tail -f $HERE/gpu1.log"
echo ""
echo "Waiting for both to finish..."
wait $PID0
STATUS0=$?
wait $PID1
STATUS1=$?

echo ""
echo "gpu0.sh (cuda:2) exit status: $STATUS0"
echo "gpu1.sh (cuda:3) exit status: $STATUS1"
