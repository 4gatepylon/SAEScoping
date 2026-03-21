#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for THRESHOLD in 5e-4 1e-3; do
    OUTPUT_DIR="${SCRIPT_DIR}/outputs_scoping_h${THRESHOLD}"
    echo "========================================================"
    echo "Running pipeline with firing-rate-threshold=${THRESHOLD}"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "========================================================"
    python "${SCRIPT_DIR}/script_scoping_pipeline_stemqa_biology.py" \
        --stage all \
        --firing-rate-threshold "${THRESHOLD}" \
        --output-dir "${OUTPUT_DIR}"
done

echo "Sweep complete."
