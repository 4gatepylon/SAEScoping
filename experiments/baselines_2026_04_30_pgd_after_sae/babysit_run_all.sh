#!/usr/bin/env bash
# Sequentially runs the 16 wanda+PGD-after-SAE scripts in this directory on
# cuda:3, inside the `saescoping` conda env.
#
# Per run:
#   - stdout+stderr -> babysit_logs/<ts>/<script>.log
#   - status row appended to babysit_logs/<ts>/STATUS.tsv
#       columns: script  state  exit_code  start_iso  end_iso  elapsed_s
#       state in {running, done, failed}
#   - one failure does NOT abort the sweep (set +e around each run).
#
# Default order: minis first (cheap smoke), then fulls; gemma-2-9b-it before
# gemma-3-12b-it. Override via $SCRIPTS_FILTER (regex) if you want a subset.
#
# Usage:
#   bash experiments/baselines_2026_04_30_pgd_after_sae/babysit_run_all.sh
#   bash experiments/baselines_2026_04_30_pgd_after_sae/babysit_run_all.sh --dry-run
#   SCRIPTS_FILTER='gemma-2-9b-it.*mini' bash .../babysit_run_all.sh
#
# --dry-run substitutes each real script with a ~1s synthetic payload that
# emits real-looking progress markers; one slot deliberately fails so the
# failure path + babysit_status.sh ERR signal are also exercised. No GPU work.
#
# Babysit while it runs (separate shell):
#   bash experiments/baselines_2026_04_30_pgd_after_sae/babysit_status.sh
#
# Latest run dir (resolved by babysit_status.sh too):
#   ls -1dt experiments/baselines_2026_04_30_pgd_after_sae/babysit_logs/* | head -1

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

STATUS="${LOG_DIR}/STATUS.tsv"
SUMMARY="${LOG_DIR}/SUMMARY.txt"
ENV_LOG="${LOG_DIR}/ENV.txt"
printf "script\tstate\texit_code\tstart_iso\tend_iso\telapsed_s\n" > "${STATUS}"

# Pin physical GPU once for the whole sweep.
export CUDA_VISIBLE_DEVICES=3

# ----- saescoping conda env activation -----
# The per-run .sh scripts call `python` directly and assume an active env.
# Activate once here so every child run inherits it. set +u around sourcing
# because conda's init scripts touch unset vars.
if [[ "${CONDA_DEFAULT_ENV:-}" != "saescoping" ]]; then
  if [[ -z "${_CONDA_ROOT:-}" ]] && ! command -v conda >/dev/null 2>&1; then
    # Try common locations + the user's .bashrc (carries conda init block).
    if [[ -f "$HOME/.bashrc" ]]; then set +u; source "$HOME/.bashrc"; set -u; fi
    if ! command -v conda >/dev/null 2>&1; then
      for cand in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/conda" "/usr/local/miniconda3"; do
        if [[ -f "${cand}/etc/profile.d/conda.sh" ]]; then
          set +u; source "${cand}/etc/profile.d/conda.sh"; set -u
          break
        fi
      done
    fi
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "[babysit] ERROR: conda not on PATH; cannot activate saescoping" >&2
    exit 2
  fi
  set +u; conda activate saescoping; set -u
fi

# Sanity dump.
{
  echo "[babysit] log dir: ${LOG_DIR}"
  echo "[babysit] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "[babysit] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<unset>}"
  echo "[babysit] which python: $(command -v python || echo '<not found>')"
  python -c "import sys, torch; print(f'python={sys.version.split()[0]} torch={torch.__version__} cuda_avail={torch.cuda.is_available()} ndev={torch.cuda.device_count()}')" 2>&1 || echo "[babysit] WARN: torch import failed"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>&1 | sed 's/^/  /'
  fi
} | tee "${ENV_LOG}"

# Default order: minis first (smoke), fulls after; 9b before 12b.
ALL_SCRIPTS=(
  # minis -- fast smoke
  "wanda_with_pgd_v1_after_sae_gemma-2-9b-it_biology_mini.sh"
  "wanda_with_pgd_v1_after_sae_gemma-2-9b-it_chemistry_mini.sh"
  "wanda_with_pgd_v1_after_sae_gemma-2-9b-it_math_mini.sh"
  "wanda_with_pgd_v1_after_sae_gemma-2-9b-it_physics_mini.sh"
  "wanda_with_pgd_v1_after_sae_gemma-3-12b-it_biology_mini.sh"
  "wanda_with_pgd_v1_after_sae_gemma-3-12b-it_chemistry_mini.sh"
  "wanda_with_pgd_v1_after_sae_gemma-3-12b-it_math_mini.sh"
  "wanda_with_pgd_v1_after_sae_gemma-3-12b-it_physics_mini.sh"
  # fulls -- long
  "wanda_with_pgd_v1_after_sae_gemma-2-9b-it_biology_full.sh"
  "wanda_with_pgd_v1_after_sae_gemma-2-9b-it_chemistry_full.sh"
  "wanda_with_pgd_v1_after_sae_gemma-2-9b-it_math_full.sh"
  "wanda_with_pgd_v1_after_sae_gemma-2-9b-it_physics_full.sh"
  "wanda_with_pgd_v1_after_sae_gemma-3-12b-it_biology_full.sh"
  "wanda_with_pgd_v1_after_sae_gemma-3-12b-it_chemistry_full.sh"
  "wanda_with_pgd_v1_after_sae_gemma-3-12b-it_math_full.sh"
  "wanda_with_pgd_v1_after_sae_gemma-3-12b-it_physics_full.sh"
)

# Optional regex filter.
SCRIPTS=()
FILTER="${SCRIPTS_FILTER:-}"
for s in "${ALL_SCRIPTS[@]}"; do
  if [[ -z "${FILTER}" ]] || [[ "${s}" =~ ${FILTER} ]]; then
    SCRIPTS+=("${s}")
  fi
done
if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
  echo "[babysit] ERROR: SCRIPTS_FILTER='${FILTER}' matched 0 of ${#ALL_SCRIPTS[@]} scripts" >&2
  exit 2
fi

echo "[babysit] running ${#SCRIPTS[@]}/${#ALL_SCRIPTS[@]} scripts sequentially on cuda:3 (filter='${FILTER}')"

OVERALL_START=$(date +%s)

for s in "${SCRIPTS[@]}"; do
  log="${LOG_DIR}/${s}.log"
  start_iso="$(date -Iseconds)"
  start_s=$(date +%s)

  # Mark running so babysit_status.sh can distinguish queued from running.
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${s}" "running" "" "${start_iso}" "" "" >> "${STATUS}"

  echo "[babysit] >>> ${s}  (log: ${log})"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    set +e
    {
      echo "=== Baseline ${s} (DRY RUN) ==="
      sleep 1
      echo "=== nn.Linear sparsity 0.4 ==="
      sleep 0.2
      echo "[recovery] step 100/2005"
      # Make one slot fail to exercise the ERR path.
      if [[ "${s}" == *"chemistry_mini.sh" ]]; then
        echo "RuntimeError: synthetic dry-run failure"
        exit 1
      fi
      echo "Summary: ok"
      echo "Artifacts: /tmp/dry"
    } > "${log}" 2>&1
    rc=$?
    set -e
  else
    set +e
    bash "${HERE}/${s}" > "${log}" 2>&1
    rc=$?
    set -e
  fi

  end_iso="$(date -Iseconds)"
  end_s=$(date +%s)
  elapsed=$(( end_s - start_s ))
  state="done"; [[ "${rc}" -ne 0 ]] && state="failed"

  # Replace the trailing 'running' row for this script with the final row.
  awk -v s="${s}" -v st="${state}" -v rc="${rc}" \
      -v si="${start_iso}" -v ei="${end_iso}" -v el="${elapsed}" '
      BEGIN { FS=OFS="\t"; replaced=0 }
      {
        if (!replaced && $1==s && $2=="running") {
          $2=st; $3=rc; $5=ei; $6=el; replaced=1
        }
        print
      }' "${STATUS}" > "${STATUS}.tmp" && mv "${STATUS}.tmp" "${STATUS}"

  echo "[babysit] <<< ${s}  state=${state} rc=${rc} elapsed=${elapsed}s"
done

OVERALL_END=$(date +%s)
TOTAL=$(( OVERALL_END - OVERALL_START ))

{
  echo "[babysit] all scripts attempted"
  echo "[babysit] total elapsed: ${TOTAL}s"
  echo
  column -t -s $'\t' "${STATUS}"
} | tee "${SUMMARY}"

# Exit non-zero if any run failed, so a wrapping nohup/cron can pick it up.
if grep -P '^[^\t]+\tfailed\t' "${STATUS}" > /dev/null; then
  exit 1
fi
exit 0
