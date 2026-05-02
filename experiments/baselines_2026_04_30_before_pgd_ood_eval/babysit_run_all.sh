#!/usr/bin/env bash
# Runs all 8 OOD-eval baseline scripts in this directory across two GPUs in
# parallel: cuda:0 and cuda:1. Each GPU gets a sequential queue of 4 cells.
# Queues are split so each GPU does 2x9b + 2x12b — wall time should be
# similar across queues.
#
# Per cell:
#   - WANDA_DEVICE=cuda:N injected (the per-cell run_*.sh honors this).
#   - stdout+stderr -> babysit_logs/<ts>/<cell>.log  (retry: <cell>.retry.log)
#   - status row appended to babysit_logs/<ts>/STATUS_cuda<N>.tsv
#       (retry pass writes to STATUS_cuda<N>_retry.tsv)
#       columns: script  gpu  state  exit_code  start_iso  end_iso  elapsed_s
#       state in {running, done, failed}
#   - one failure does NOT abort the queue (set +e around each cell).
#
# After both main queues drain, any cell with state=failed is retried ONCE on
# whichever GPU is free (round-robin across cuda:0 + cuda:1, parallel). Single
# retry per cell — anything that still fails is surfaced and exits non-zero.
#
# Conda: activates the `saescoping` env before launching anything (the per-cell
# .sh scripts assume the caller's shell already has it active).
#
# Usage:
#   bash experiments/baselines_2026_04_30_before_pgd_ood_eval/babysit_run_all.sh
#   bash experiments/baselines_2026_04_30_before_pgd_ood_eval/babysit_run_all.sh --dry-run
#
# --dry-run substitutes each real cell with a ~1s synthetic payload that emits
# real-looking progress markers; one slot per queue deliberately fails so the
# failure path + babysit_status.sh ERR signal are also exercised. No GPU work.
#
# Babysit while it runs (separate shell):
#   bash experiments/baselines_2026_04_30_before_pgd_ood_eval/babysit_status.sh
#
# Latest run dir (resolved by babysit_status.sh too):
#   ls -1dt experiments/baselines_2026_04_30_before_pgd_ood_eval/babysit_logs/* | head -1

set -uo pipefail

DRY_RUN=0
for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[babysit] unknown arg: ${arg}" >&2; exit 2 ;;
  esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${HERE}/babysit_logs/${TS}"
mkdir -p "${LOG_DIR}"

echo "[babysit] log dir: ${LOG_DIR}"
echo "[babysit] dry_run=${DRY_RUN}"

# --- Conda env activation -------------------------------------------------
# Per project memory: everything runs in `saescoping`. Activate here so the
# per-cell .sh files inherit it (they don't activate it themselves).
if [[ "${DRY_RUN}" -eq 0 ]]; then
  if [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "saescoping" ]]; then
    # Try the conventional miniconda/anaconda hooks.
    for cand in "$HOME/miniconda3/etc/profile.d/conda.sh" \
                "$HOME/anaconda3/etc/profile.d/conda.sh" \
                "/opt/conda/etc/profile.d/conda.sh"; do
      if [[ -f "${cand}" ]]; then
        # shellcheck disable=SC1090
        source "${cand}"
        break
      fi
    done
    if ! command -v conda >/dev/null 2>&1; then
      echo "[babysit] FATAL: conda not on PATH and no conda.sh found" >&2
      exit 3
    fi
    conda activate saescoping || { echo "[babysit] FATAL: failed to activate saescoping" >&2; exit 3; }
  fi
  echo "[babysit] conda env: ${CONDA_DEFAULT_ENV:-?}  python: $(command -v python)"
fi

# --- Queue definitions ----------------------------------------------------
# Each queue: 2x 9b + 2x 12b. Order: alternate model sizes so the queue
# starts on cheaper work and confirms plumbing before the 12b cells.
QUEUE_CUDA0=(
  "run_gemma-2-9b-it_biology.sh"
  "run_gemma-3-12b-it_biology.sh"
  "run_gemma-2-9b-it_chemistry.sh"
  "run_gemma-3-12b-it_chemistry.sh"
)
QUEUE_CUDA1=(
  "run_gemma-2-9b-it_math.sh"
  "run_gemma-3-12b-it_math.sh"
  "run_gemma-2-9b-it_physics.sh"
  "run_gemma-3-12b-it_physics.sh"
)

# --- Per-queue runner -----------------------------------------------------
# $1 = device (cuda:0 / cuda:1)
# $2 = tag ("" for main pass, "retry" for retry pass)
# $3.. = script names (relative to HERE)
run_queue() {
  local device="$1"; shift
  local tag="$1"; shift
  local gpu_idx="${device##*:}"
  local status_suffix=""
  local log_suffix=""
  if [[ -n "${tag}" ]]; then
    status_suffix="_${tag}"
    log_suffix=".${tag}"
  fi
  local status="${LOG_DIR}/STATUS_cuda${gpu_idx}${status_suffix}.tsv"
  printf "script\tgpu\tstate\texit_code\tstart_iso\tend_iso\telapsed_s\n" > "${status}"

  local cell start_iso start_s end_iso end_s elapsed rc state log

  for cell in "$@"; do
    log="${LOG_DIR}/${cell}${log_suffix}.log"
    start_iso="$(date -Iseconds)"
    start_s=$(date +%s)

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${cell}" "${device}" "running" "" "${start_iso}" "" "" >> "${status}"

    echo "[${device}${tag:+/${tag}}] >>> ${cell}  (log: ${log})"

    set +e
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      # Synthetic payload in a real subshell — emits a couple real-looking
      # progress markers. One cell per queue deliberately fails to exercise
      # the failure path + status reporter. Subshell so `exit` doesn't kill
      # the surrounding queue.
      (
        echo "[wandb] initialized run: synthetic-${cell} (dry-run)"
        echo "=== Baseline (pre-pruning) ==="
        sleep 1
        echo "=== nn.Linear sparsity 40.0% (step 0) ==="
        if [[ "${cell}" == *"chemistry"* && "${cell}" == *"3-12b"* ]] || \
           [[ "${cell}" == *"physics"*   && "${cell}" == *"3-12b"* ]]; then
          echo "Traceback (most recent call last):"
          echo "RuntimeError: synthetic dry-run failure"
          exit 1
        else
          echo "Summary: dry-run synthetic"
          echo "Artifacts: ${LOG_DIR}/(fake)"
          exit 0
        fi
      ) > "${log}" 2>&1
      rc=$?
    else
      WANDA_DEVICE="${device}" bash "${HERE}/${cell}" > "${log}" 2>&1
      rc=$?
    fi
    set -e

    end_iso="$(date -Iseconds)"
    end_s=$(date +%s)
    elapsed=$(( end_s - start_s ))
    state="done"; [[ "${rc}" -ne 0 ]] && state="failed"

    awk -v s="${cell}" -v st="${state}" -v rc="${rc}" \
        -v ei="${end_iso}" -v el="${elapsed}" '
        BEGIN { FS=OFS="\t"; replaced=0 }
        {
          if (!replaced && $1==s && $3=="running") {
            $3=st; $4=rc; $6=ei; $7=el; replaced=1
          }
          print
        }' "${status}" > "${status}.tmp" && mv "${status}.tmp" "${status}"

    echo "[${device}${tag:+/${tag}}] <<< ${cell}  state=${state} rc=${rc} elapsed=${elapsed}s"
  done
}

# --- Launch the two main queues in parallel ------------------------------
OVERALL_START=$(date +%s)

run_queue "cuda:0" "" "${QUEUE_CUDA0[@]}" &
PID0=$!
run_queue "cuda:1" "" "${QUEUE_CUDA1[@]}" &
PID1=$!

echo "[babysit] main queue pids: cuda:0=${PID0}  cuda:1=${PID1}"

set +e
wait "${PID0}"; RC0=$?
wait "${PID1}"; RC1=$?
set -e

MAIN_END=$(date +%s)
MAIN_ELAPSED=$(( MAIN_END - OVERALL_START ))
echo "[babysit] main pass done in ${MAIN_ELAPSED}s  (queue rc: cuda:0=${RC0} cuda:1=${RC1})"

# --- Retry pass for failed cells -----------------------------------------
# Collect every cell that ended in 'failed' across both main-pass STATUS files.
mapfile -t FAILED < <(
  for f in "${LOG_DIR}"/STATUS_cuda0.tsv "${LOG_DIR}"/STATUS_cuda1.tsv; do
    [[ -f "${f}" ]] || continue
    awk -F'\t' 'NR>1 && $3=="failed" { print $1 }' "${f}"
  done
)

if (( ${#FAILED[@]} > 0 )); then
  echo "[babysit] retry pass: ${#FAILED[@]} failed cell(s) -> ${FAILED[*]}"

  RETRY_CUDA0=()
  RETRY_CUDA1=()
  i=0
  for cell in "${FAILED[@]}"; do
    if (( i % 2 == 0 )); then RETRY_CUDA0+=("${cell}"); else RETRY_CUDA1+=("${cell}"); fi
    i=$(( i + 1 ))
  done

  RPID0=""; RPID1=""
  if (( ${#RETRY_CUDA0[@]} > 0 )); then
    run_queue "cuda:0" "retry" "${RETRY_CUDA0[@]}" &
    RPID0=$!
  fi
  if (( ${#RETRY_CUDA1[@]} > 0 )); then
    run_queue "cuda:1" "retry" "${RETRY_CUDA1[@]}" &
    RPID1=$!
  fi
  echo "[babysit] retry queue pids: cuda:0=${RPID0:-none}  cuda:1=${RPID1:-none}"

  set +e
  [[ -n "${RPID0}" ]] && wait "${RPID0}"
  [[ -n "${RPID1}" ]] && wait "${RPID1}"
  set -e
else
  echo "[babysit] retry pass: nothing to retry"
fi

OVERALL_END=$(date +%s)
TOTAL=$(( OVERALL_END - OVERALL_START ))

# --- Final summary -------------------------------------------------------
SUMMARY="${LOG_DIR}/SUMMARY.txt"
{
  echo "[babysit] all passes complete"
  echo "[babysit] total elapsed: ${TOTAL}s  (main pass ${MAIN_ELAPSED}s)"
  echo
  for f in "${LOG_DIR}"/STATUS_cuda0.tsv "${LOG_DIR}"/STATUS_cuda1.tsv \
           "${LOG_DIR}"/STATUS_cuda0_retry.tsv "${LOG_DIR}"/STATUS_cuda1_retry.tsv; do
    [[ -f "${f}" ]] || continue
    echo "--- ${f##*/} ---"
    column -t -s $'\t' "${f}"
    echo
  done
} | tee "${SUMMARY}"

# --- Final pass/fail determination ---------------------------------------
# A cell is OK iff its LATEST attempt (retry if present, else main) ended in
# state=done. A retry's existence overrides the main-pass row for that cell.
declare -A FINAL_STATE
for f in "${LOG_DIR}"/STATUS_cuda0.tsv "${LOG_DIR}"/STATUS_cuda1.tsv; do
  [[ -f "${f}" ]] || continue
  while IFS=$'\t' read -r script gpu state rc start_iso end_iso elapsed; do
    [[ "${script}" == "script" ]] && continue
    FINAL_STATE["${script}"]="${state}"
  done < "${f}"
done
for f in "${LOG_DIR}"/STATUS_cuda0_retry.tsv "${LOG_DIR}"/STATUS_cuda1_retry.tsv; do
  [[ -f "${f}" ]] || continue
  while IFS=$'\t' read -r script gpu state rc start_iso end_iso elapsed; do
    [[ "${script}" == "script" ]] && continue
    FINAL_STATE["${script}"]="${state}"
  done < "${f}"
done

still_failed=0
for cell in "${!FINAL_STATE[@]}"; do
  if [[ "${FINAL_STATE[${cell}]}" != "done" ]]; then
    still_failed=$(( still_failed + 1 ))
    echo "[babysit] STILL FAILED after retry: ${cell}" | tee -a "${SUMMARY}"
  fi
done

if (( still_failed > 0 )); then
  echo "[babysit] ${still_failed} cell(s) still failed" | tee -a "${SUMMARY}"
  exit 1
fi
echo "[babysit] all 8 cells finished cleanly" | tee -a "${SUMMARY}"
exit 0
