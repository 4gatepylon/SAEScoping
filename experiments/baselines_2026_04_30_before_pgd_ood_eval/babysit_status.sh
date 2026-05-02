#!/usr/bin/env bash
# One-shot health/progress reporter for the parallel baseline sweep.
# Reads babysit_logs/<latest>/STATUS_cuda*.tsv (or $1 if given), greps each
# per-cell log for failure + progress markers, prints a tight summary.
#
# Usage:
#   bash experiments/baselines_2026_04_30_before_pgd_ood_eval/babysit_status.sh
#   bash experiments/baselines_2026_04_30_before_pgd_ood_eval/babysit_status.sh \
#        experiments/baselines_2026_04_30_before_pgd_ood_eval/babysit_logs/20260501_001234

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

LOG_DIR="${1:-}"
if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="$(ls -1dt "${HERE}/babysit_logs"/*/ 2>/dev/null | head -1)"
  LOG_DIR="${LOG_DIR%/}"
fi

if [[ -z "${LOG_DIR}" || ! -d "${LOG_DIR}" ]]; then
  echo "[babysit-status] no log dir found under ${HERE}/babysit_logs"
  echo "[babysit-status] has the runner been started?"
  exit 2
fi

shopt -s nullglob
# Order: main passes first (cuda0, cuda1), retries last.
MAIN_FILES=( "${LOG_DIR}"/STATUS_cuda0.tsv "${LOG_DIR}"/STATUS_cuda1.tsv )
RETRY_FILES=( "${LOG_DIR}"/STATUS_cuda0_retry.tsv "${LOG_DIR}"/STATUS_cuda1_retry.tsv )
STATUS_FILES=()
for f in "${MAIN_FILES[@]}" "${RETRY_FILES[@]}"; do
  [[ -f "${f}" ]] && STATUS_FILES+=( "${f}" )
done
shopt -u nullglob
if (( ${#STATUS_FILES[@]} == 0 )); then
  echo "[babysit-status] no STATUS_cuda*.tsv under ${LOG_DIR}"
  exit 2
fi

# Failure / progress patterns (case-sensitive — sweep_wanda.py is consistent).
FAIL_PAT='Traceback|CUDA out of memory|out of memory|RuntimeError|AssertionError|^Killed|killed:9|Segmentation fault|Error: |Aborted|wandb: ERROR'
PROGRESS_PAT='=== Baseline|=== nn\.Linear sparsity|llm_judge:|\[recovery\]|\[wandb\] initialized run|Summary:'

echo "[babysit-status] log_dir: ${LOG_DIR}"
echo

# nvidia-smi for the two devices we use.
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[babysit-status] gpus cuda:0 + cuda:1:"
  nvidia-smi --id=0,1 --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader 2>/dev/null \
    | sed 's/^/  /'
  echo
fi

# --- Aggregate per-cell final state (retry overrides main) ---------------
declare -A CELL_STATE
# Walk main files first, then retry files; later writes win.
for sf in "${STATUS_FILES[@]}"; do
  while IFS=$'\t' read -r script gpu state rc start_iso end_iso elapsed; do
    [[ "${script}" == "script" ]] && continue
    CELL_STATE["${script}"]="${state}"
  done < "${sf}"
done

n_cells=${#CELL_STATE[@]}
n_done=0; n_failed=0; n_running=0
for st in "${CELL_STATE[@]}"; do
  case "${st}" in
    done)    n_done=$(( n_done + 1 )) ;;
    failed)  n_failed=$(( n_failed + 1 )) ;;
    running) n_running=$(( n_running + 1 )) ;;
  esac
done

echo "[babysit-status] cells: done=${n_done} failed=${n_failed} running=${n_running} / ${n_cells} seen (of 8 total)"
echo "[babysit-status] (cells still queued and not yet running don't appear as rows)"
echo

# --- Per-row report, one table per queue ---------------------------------
for sf in "${STATUS_FILES[@]}"; do
  qname="$(basename "${sf}" .tsv)"
  # Retry STATUS files reference <cell>.retry.log; main reference <cell>.log.
  log_suffix=""
  [[ "${qname}" == *"_retry" ]] && log_suffix=".retry"

  echo "=== ${qname} ==="
  printf "%-44s %-8s %-8s %-4s %-9s %s\n" "script" "gpu" "state" "rc" "elapsed" "signals"
  printf '%s\n' "----------------------------------------------------------------------------------------------------------------"

  while IFS=$'\t' read -r script gpu state rc start_iso end_iso elapsed; do
    [[ "${script}" == "script" ]] && continue

    log="${LOG_DIR}/${script}${log_suffix}.log"
    signals=""

    if [[ -f "${log}" ]]; then
      last_progress="$(grep -E "${PROGRESS_PAT}" "${log}" 2>/dev/null | tail -1 | tr -d '\r' | cut -c1-60)"
      fail_hit="$(grep -m1 -E "${FAIL_PAT}" "${log}" 2>/dev/null | tr -d '\r' | cut -c1-60)"
      if [[ -n "${fail_hit}" ]]; then
        signals="ERR: ${fail_hit}"
      elif [[ -n "${last_progress}" ]]; then
        signals="${last_progress}"
      else
        signals="(no progress markers yet)"
      fi
    else
      signals="(no log yet — queued)"
    fi

    if [[ "${state}" == "running" && -n "${start_iso}" ]]; then
      start_s=$(date -d "${start_iso}" +%s 2>/dev/null || echo 0)
      if [[ "${start_s}" != "0" ]]; then
        now_s=$(date +%s)
        elapsed=$(( now_s - start_s ))"+"
      fi
    fi

    printf "%-44s %-8s %-8s %-4s %-9s %s\n" \
      "${script}" "${gpu}" "${state}" "${rc:--}" "${elapsed:--}" "${signals}"
  done < "${sf}"
  echo
done

echo "[babysit-status] tail any log:  tail -n 40 -F ${LOG_DIR}/<script>.log"
