#!/bin/bash
# Launch both GPU job scripts as background processes.
# Each runs 3 sequential training jobs on its assigned GPU.
set -euo pipefail
cd "$(dirname "$0")"

echo "Starting GPU 0 jobs (pid logged below)..."
bash gpu0.sh > gpu0.log 2>&1 &
PID0=$!
echo "  GPU 0 PID: $PID0 (log: jobs/gpu0.log)"

echo "Starting GPU 1 jobs (pid logged below)..."
bash gpu1.sh > gpu1.log 2>&1 &
PID1=$!
echo "  GPU 1 PID: $PID1 (log: jobs/gpu1.log)"

echo ""
echo "Both running. Monitor with:"
echo "  tail -f jobs/gpu0.log"
echo "  tail -f jobs/gpu1.log"
echo ""
echo "Waiting for both to finish..."
wait $PID0
STATUS0=$?
wait $PID1
STATUS1=$?

echo ""
echo "GPU 0 exit status: $STATUS0"
echo "GPU 1 exit status: $STATUS1"
