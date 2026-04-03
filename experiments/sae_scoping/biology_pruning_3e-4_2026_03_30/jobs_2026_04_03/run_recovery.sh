#!/bin/bash
# Recovery training: train the biology-scoped model on chemistry/physics using
# a DOMAIN-MATCHED SAE (chemistry or physics neuron distribution), so we can
# compare how easily the model can recover when given the "right" neurons vs.
# the biology-locked ones used in run_elicitation.sh.
#
# Prerequisites:
#   1. Run run_distributions.sh to produce distributions_cache/.
#   2. Run find_threshold.py to find thresholds that keep ~2000 neurons.
#   3. Pass those thresholds here.
#
# Usage:
#   bash run_recovery.sh GPU_LIST CHEM_THRESHOLDS PHYS_THRESHOLDS
#
# Arguments:
#   GPU_LIST        Comma-separated GPU IDs, e.g. "0,1,2,3"
#   CHEM_THRESHOLDS Comma-separated thresholds for chemistry, e.g. "3e-4,4e-4"
#   PHYS_THRESHOLDS Comma-separated thresholds for physics, e.g. "3e-4,4e-4"
#
# Example (2 GPUs, 2 thresholds per domain = 4 jobs, 2 per GPU sequentially):
#   bash run_recovery.sh 1,2 3e-4,4e-4 3e-4,4e-4
#
# Example (single threshold per domain):
#   bash run_recovery.sh 1,2 3e-4 3e-4
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 GPU_LIST CHEM_THRESHOLDS PHYS_THRESHOLDS" >&2
    echo "  e.g. $0 0,1,2,3 3e-4,4e-4 3e-4,4e-4" >&2
    exit 1
fi

IFS=',' read -ra GPUS         <<< "$1"
IFS=',' read -ra CHEM_HS      <<< "$2"
IFS=',' read -ra PHYS_HS      <<< "$3"

N=${#GPUS[@]}

CKPT="downloaded/model_layers_31_h0.0003/outputs/checkpoint-2000"
CHEM_DIST="distributions_cache/ignore_padding_True/chemistry/layer_31--width_16k--canonical/distribution.safetensors"
PHYS_DIST="distributions_cache/ignore_padding_True/physics/layer_31--width_16k--canonical/distribution.safetensors"
WANDB_PROJECT="sae-elicitation-ood-2026-04-03"

# Validate distributions exist
for DIST in "$CHEM_DIST" "$PHYS_DIST"; do
    if [[ ! -f "$DIST" ]]; then
        echo "ERROR: distribution not found: $DIST" >&2
        echo "Run jobs_2026_04_03/run_distributions.sh first." >&2
        exit 1
    fi
done

# --- Build job list: (subset, dist_path, threshold) tuples ---
# Encode as parallel arrays
JOB_SUBSETS=()
JOB_DISTS=()
JOB_THRESHOLDS=()

for H in "${CHEM_HS[@]}"; do
    JOB_SUBSETS+=(chemistry)
    JOB_DISTS+=("$CHEM_DIST")
    JOB_THRESHOLDS+=("$H")
done
for H in "${PHYS_HS[@]}"; do
    JOB_SUBSETS+=(physics)
    JOB_DISTS+=("$PHYS_DIST")
    JOB_THRESHOLDS+=("$H")
done

N_JOBS=${#JOB_SUBSETS[@]}
echo "Recovery jobs: $N_JOBS total across ${N} GPU(s)"
for i in "${!JOB_SUBSETS[@]}"; do
    echo "  Job $i: subset=${JOB_SUBSETS[$i]}  h=${JOB_THRESHOLDS[$i]}"
done
echo ""

# --- Assign jobs to GPUs (round-robin), spawn one process per GPU ---
PIDS=()
for gpu_idx in "${!GPUS[@]}"; do
    GPU="${GPUS[$gpu_idx]}"
    (
        export CUDA_VISIBLE_DEVICES="$GPU"
        for job_idx in "${!JOB_SUBSETS[@]}"; do
            if [[ $((job_idx % N)) -eq $gpu_idx ]]; then
                SUBSET="${JOB_SUBSETS[$job_idx]}"
                DIST="${JOB_DISTS[$job_idx]}"
                H="${JOB_THRESHOLDS[$job_idx]}"
                # Use a tag that distinguishes domain-matched dist from biology dist
                TAG="${SUBSET}_dist_h${H}"
                OUT_DIR="./outputs/recovery_${TAG}/after_31/$SUBSET/checkpoint-2000"
                echo "=== [CUDA:$GPU] Recovery: $SUBSET  h=$H ==="
                python script_train_gemma9b_sae.py \
                    --sae --subset "$SUBSET" \
                    -c "$CKPT" -p "$DIST" -h "$H" \
                    -b 2 -a 8 -s 8000 \
                    --eval-every 200 \
                    --utility-eval-every 200 \
                    --biology-utility-eval-every 200 \
                    --save-every 1000 \
                    --output-dir "$OUT_DIR" \
                    -w "$WANDB_PROJECT" \
                    --wandb-run-name "recovery/$TAG/ckpt-2000" \
                    || echo "[CUDA:$GPU] FAILED: recovery $SUBSET h=$H"
            fi
        done
    ) &
    PIDS+=($!)
done

wait "${PIDS[@]}"
echo "All recovery jobs complete."
