# Babysit runbook — OOD baseline sweep (no PGD)

Self-instructions for the Claude session driving this sweep. Read top to
bottom on resume; jump to the matching section when a poll surfaces a
specific signal.

## Output locations (where every artifact lands)

For the **current** run (started 2026-05-01 ~01:46):

- `RUNNER_BASH_PID = 3540842` — top-level bash that owns both queues. Never kill.
- `LOG_DIR = experiments/baselines_2026_04_30_before_pgd_ood_eval/babysit_logs/20260501_014604/`
  - `<cell>.log` — main-pass stdout+stderr per cell (8 of these).
  - `<cell>.retry.log` — retry-pass stdout+stderr (only if main failed).
  - `STATUS_cuda{0,1}.tsv` — per-queue main-pass status.
  - `STATUS_cuda{0,1}_retry.tsv` — per-queue retry status (only if retries ran).
  - `SUMMARY.txt` — written by runner at very end; its existence = "runner finished".
  - `.poll_state.txt` — my own bookkeeping for consecutive-failure tracking.
- `/tmp/babysit_launcher.log` — top-level runner stdout (the `[babysit]` /
  `[cuda:N]` lines). The string `all passes complete` here = end signal.
- Wandb runs (remote): project `saescoping--ood-baseline-no-pgd-2026-04-30`,
  run names `gemma-{2-9b,3-12b}-it_{biology,chemistry,math,physics}_no-pgd_ood_2026-04-30`.
  Cannot verify from this session — user must spot-check UI.
- Per-cell artifact dir: each cell's log prints `Artifacts: <path>` near the
  end. Path is set by `$SAESCOPING_ARTIFACTS_LOCATION` or `.` per
  `config_ood_eval.yaml: operational.artifacts_dir`. Naming follows
  `experiments/baselines_2026_04_29/NAMING.md`.

For **future** runs the same `LOG_DIR` pattern applies under
`babysit_logs/<TS>/`; resolve the latest with
`ls -1dt experiments/baselines_2026_04_30_before_pgd_ood_eval/babysit_logs/*/ | head -1`.

## Escalation policy (per user feedback memory)

- Runner is `set +e` per cell — single-cell failures are normal and the queue
  continues. Do NOT push-notify the user on isolated failures.
- **3 consecutive failures** = systemic breakage signal. Send a single
  `PushNotification` with: run dir, the 3 failing script names, first error
  marker (`Traceback` / `CUDA out of memory` / `Killed` / `wandb: ERROR`)
  from each log. Then KEEP POLLING — do not halt.
- Reset the consecutive-failure counter on the next cell that completes with
  state=done.
- Always re-fire the next cron tick until `SUMMARY.txt` exists (or the user
  explicitly says stop). Standing authorization for `dangerouslyDisableSandbox`
  on GPU-touching commands.

## 0. Operating context (do not violate)

- Allowed GPUs: **cuda:0 and cuda:1 only**. Other agents may use other GPUs;
  do not touch them.
- Python runs in **conda env `saescoping`**. The runner activates it; do not
  bypass.
- Stay inside `experiments/baselines_2026_04_30_before_pgd_ood_eval/`. Other
  agents may be editing other dirs; do not edit, refactor, or "tidy" files
  outside this dir.
- `dangerouslyDisableSandbox: true` ONLY for: `nvidia-smi`, launching the
  runner, killing/inspecting the runner's processes, and final wandb verify.
  Reads + edits stay sandboxed.

## 1. Launch

1. Verify clean GPU state:
   `nvidia-smi --id=0,1 --query-gpu=memory.free --format=csv,noheader`
   Expect both ≥ 70000 MiB free. If not, ping user and wait.
2. Launch in background:
   `bash experiments/baselines_2026_04_30_before_pgd_ood_eval/babysit_run_all.sh`
   via Bash with `run_in_background: true` AND `dangerouslyDisableSandbox: true`.
3. Capture the runner's bash PID (the Bash tool returns it). Save it in
   conversation context as `RUNNER_BASH_PID`.
4. Resolve `LOG_DIR`:
   `ls -1dt experiments/baselines_2026_04_30_before_pgd_ood_eval/babysit_logs/* | head -1`
5. First poll ~3 min later — confirm two queues are running, both GPUs
   showing python load, both first-cell logs growing.

## 2. Poll cadence

- Every ~7 min via ScheduleWakeup, target run_in_background tick.
- Each poll: run `babysit_status.sh`, then post a one-line ping in chat:
  `[poll HH:MM] main main main main / main main main main  (n_done/n_failed/n_running of 8)`
  Use one slot per cell, in queue order, with state glyph:
    `.` queued, `R` running, `D` done, `F` failed, `r` retry-running,
    `+` retry-done, `x` retry-failed-final, `?` suspect-stuck.
- On state change, also paste the `babysit_status.sh` per-row table for the
  affected queue.
- ~Every 30 min, post a one-line "still alive" check-in even if nothing
  changed, so silence is never ambiguous.

## 3. Triage decision tree (run this at every poll, before pinging)

For each cell where `state=running`:

a. **Log growing?** `stat -c %Y "${LOG_DIR}/${cell}.log"` vs prev poll's mtime.
   - Growing → fine, continue.
   - Unchanged ≥ 60 min → SUSPECT-STUCK. Go to §4.
   - Unchanged < 60 min → mark as "watch" but do nothing.

b. **Log contains failure marker?**
   `grep -E 'Traceback|CUDA out of memory|killed:9|^Killed|wandb: ERROR' log`
   - Hit → cell will exit non-zero soon; just note it. The runner records
     state=failed; retry pass picks it up automatically.

c. **GPU utilization sanity** (only spot-check 1×/30 min, not every poll):
   `nvidia-smi --id=0,1 --query-gpu=utilization.gpu,memory.used --format=csv`
   - Cell running but `utilization.gpu = 0` for ≥ 10 min while log grows →
     probably stuck on llm-judge OpenAI calls (CPU-bound), not real stuck.
     Don't kill.
   - Cell running, `utilization.gpu = 0`, log NOT growing → escalate to §4.

For cells where `state=failed` after main pass: nothing to do; retry pass
runs automatically once both main queues drain.

For cells where `state=failed` in a retry-pass STATUS file: this is a
permanent failure for this run. Surface to user with the failure marker
from the log. Do NOT retry again without explicit user OK.

## 4. Kill a suspected-stuck cell

Goal: cause the cell's python process to exit so the runner records it as
failed and the retry pass picks it up. Do NOT kill the runner itself.

1. Identify the python PID for that cell:
   `pgrep -af "sweep_wanda.py.*${cell_keyword}"` (e.g. keyword `biology`).
   Verify the device matches by checking `--device cuda:N` in the cmdline.
2. SIGTERM first, give it 30s, then SIGKILL:
   `kill -TERM <pid>; sleep 30; kill -KILL <pid> 2>/dev/null`
3. Watch the runner log — within ~1 min you should see
   `[cuda:N] <<< <cell>  state=failed rc=<nonzero>` and the queue advance.
4. Ping user: which cell, why killed, that retry pass will handle it.

NEVER kill `RUNNER_BASH_PID` or any of the queue subshell PIDs. Killing the
runner orphans both queues and you'd have to restart from scratch.

## 5. Detecting "done properly" (the user's definition)

A cell counts as truly done iff ALL of:

- `STATUS_cuda*.tsv` row says `state=done` (or its retry row does).
- Log contains a `Summary:` line.
- Log contains an `Artifacts:` line pointing to a real dir.
- Wandb run shows up under project
  `saescoping--ood-baseline-no-pgd-2026-04-30` with the expected
  `WANDB_NAME` (e.g. `gemma-2-9b-it_biology_no-pgd_ood_2026-04-30`) AND
  `state: finished` (not `crashed`/`running`).

After both passes complete, run a final manual verify:

```bash
LOG_DIR=$(ls -1dt experiments/baselines_2026_04_30_before_pgd_ood_eval/babysit_logs/*/ | head -1)
for log in "$LOG_DIR"/*.log; do
  cell=$(basename "$log" .log)
  has_summary=$(grep -c '^Summary:' "$log")
  has_artifacts=$(grep -c '^Artifacts:' "$log")
  has_err=$(grep -cE 'Traceback|CUDA out of memory|wandb: ERROR' "$log")
  echo "$cell  Summary=$has_summary  Artifacts=$has_artifacts  Errors=$has_err"
done
```

Any cell with `Summary=0` or `Artifacts=0` or `Errors>0` → not truly done.

For wandb verification I do NOT have wandb credentials in this session.
Ping user with the run names and ask them to spot-check the wandb UI; do
not claim wandb success on my own.

## 6. Recovery scenarios

### 6a. Runner process died
Symptom: `kill -0 ${RUNNER_BASH_PID}` returns nonzero.
- Check tail of `babysit_run_all.sh.log` (the runner's own stdout) for
  the cause.
- Do NOT auto-relaunch — partial state on disk. Ping user with what
  finished, what didn't, and ask whether to restart only the unfinished
  cells (via a small ad-hoc loop) or do a fresh full run.

### 6b. OOM in a 12b cell
Symptom: log has `CUDA out of memory`.
- Cell exits non-zero. Main pass continues. Retry pass will rerun it.
- A retry on the same GPU will OOM again. Before the retry pass starts,
  ping user: "12b cell OOMed; retry will likely OOM too. Want me to abort
  retry for this cell, lower batch_size to 1 (already at 1 for 12b — so
  this means lowering max_seq_len or enabling gradient_checkpointing
  through a config patch), or proceed and let it fail?"
- Default if no answer: let retry run; record permanent failure; surface.

### 6c. Wandb network blip mid-run
Symptom: `wandb: ERROR` in log but cell continued (wandb tolerates
transient network errors with `WANDB_MODE=online`).
- If cell exits 0 AND has Summary+Artifacts → log to user but treat
  as done; the wandb run may be incomplete server-side. Ask user to
  verify in UI.
- If cell exits non-zero because of wandb → retry pass handles.

### 6d. OpenAI rate limit in llm-judge
Symptom: log has `RateLimitError` or HTTP 429.
- The judge code may or may not retry internally. If cell exits non-zero,
  retry pass handles. If it exits 0 with `Summary:` and `Artifacts:`, it
  partially logged — check `llm_judge:` line count vs expected (4 domains
  × 5 sparsity points = 20 entries) and ping user if short.

### 6e. Both retry attempts also fail
Surface every still-failed cell with: cell name, original failure marker,
retry failure marker, both log paths. Do NOT retry a third time. Wait
for user direction.

## 7. Final report (post when everything ends)

When `babysit_run_all.sh` exits:

- Read its exit code from the Bash background-output stream.
- Run the §5 verify loop and paste the results table in chat.
- Summarize: how many of 8 finished cleanly, how many needed retry, how
  many still failed, total wall time, per-queue wall time.
- Ask user to spot-check wandb runs.

## 8. Quick command reference

```bash
# Status snapshot:
bash experiments/baselines_2026_04_30_before_pgd_ood_eval/babysit_status.sh

# Latest log dir:
LOG_DIR=$(ls -1dt experiments/baselines_2026_04_30_before_pgd_ood_eval/babysit_logs/*/ | head -1)

# Tail any cell:
tail -n 60 "${LOG_DIR}/<cell>.log"

# nvidia-smi for our two GPUs:
nvidia-smi --id=0,1 --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv

# Find python PIDs for our runs:
pgrep -af "sweep_wanda.py"

# Is the runner still alive?
kill -0 "${RUNNER_BASH_PID}" && echo alive || echo dead
```
