#!/usr/bin/env bash
# Throwaway: Wanda on gemma-2-9b-it across 4 StemQA subsets.
#   n_calibration = 10000, n_train = 4 (dummy — just exercises the eval path),
#   n_test = 200, n_judge_samples = 50, sparsity levels default (0.0..0.9).
# GPU 2 runs biology then chemistry; GPU 3 runs physics then math, in parallel.
#
# Run from the repo root:
#   cd /mnt/align4_drive2/adrianoh/git/SAEScoping
#   bash experiments/baselines/throwaway_wanda_9b_bigcalib.sh
# Or detached:
#   nohup bash experiments/baselines/throwaway_wanda_9b_bigcalib.sh \
#       > logs/throwaway_wanda_9b_bigcalib.out 2>&1 &

set -u  # no -e: we want all four subsets attempted even if one fails
# TODO(claude) priority:low: wait PID2 PID3 returns only the exit status of the
# last PID, so if GPU-2's subset(s) silently die, the script reports success.
# Capture individual rc's and surface a pass/fail summary at the end.
cd "$(dirname "$0")/../.."  # cd to repo root regardless of caller CWD

MODEL="google/gemma-2-9b-it"
N_CALIB=10000
N_TRAIN=4     # tiny but non-zero so compute_loss path exercises an actual batch
N_TEST=200
N_JUDGE=50

mkdir -p logs

# The cache filename does NOT encode n_calibration, so if a prior run at a
# different calib size exists, we'd silently reuse it. Nuke the Wanda cache for
# these four subsets so the 10k saliency is freshly computed (and cached
# afterwards for any future sweeps at 10k).
for S in biology chemistry physics math; do
    rm -f "saliency_cache/google--gemma-2-9b-it/${S}/wanda_saliency.safetensors"
done

run_one() {
    local gpu=$1
    local subset=$2
    local log="logs/wanda_9b_${subset}_calib${N_CALIB}.log"
    echo "[GPU ${gpu}] start: ${subset} -> ${log}"
    CUDA_VISIBLE_DEVICES="${gpu}" python experiments/baselines/sweep_sparsity.py \
        --method wanda \
        --model "${MODEL}" \
        --device cuda:0 \
        --dataset-subset "${subset}" \
        --n-calibration "${N_CALIB}" \
        --n-train "${N_TRAIN}" \
        --n-test "${N_TEST}" \
        --n-judge-samples "${N_JUDGE}" \
        > "${log}" 2>&1
    echo "[GPU ${gpu}] done : ${subset} (rc=$?)"
}

# Serial within a GPU, parallel across GPUs.
run_gpu2() { run_one 2 biology;  run_one 2 chemistry; }
run_gpu3() { run_one 3 physics;  run_one 3 math;      }

run_gpu2 &
PID2=$!
run_gpu3 &
PID3=$!

wait "${PID2}" "${PID3}"
echo "=== all subsets done ==="
