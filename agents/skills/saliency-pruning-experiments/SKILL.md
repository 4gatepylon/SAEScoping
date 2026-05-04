---
name: saliency-pruning-experiments
description: >-
  Run, design, and implement saliency-based weight-pruning experiments on
  language models. Covers all three experiment phases (gradient-map generation,
  pruning sweep, prune-and-recover sweep), writing dated shell-script run
  folders, discovering CLI options, and explaining the pruning algorithm to the
  user. Use when the user wants to run an experiment, understand what a script
  does, write a new run folder, interpret WandB results, or understand how
  saliency-based pruning works.
---

# Saliency Pruning Experiments

## First: read the documentation

Before doing anything, read these two files:

```
experiments/saliency_pruning/toy_sweep_2026_03_14/EXPERIMENT_README.md
experiments/saliency_pruning/toy_sweep_2026_03_14/CLAUDE.md
```

`EXPERIMENT_README.md` has the full CLI reference for every script.
`CLAUDE.md` has all coding standards and WandB naming conventions.

---

## Environment

```bash
conda activate saescoping
export PYTHONPATH=experiments/saliency_pruning/toy_sweep_2026_03_14
```

All shell scripts self-set `PYTHONPATH` so the user can run them directly from
any working directory.

---

## Algorithm overview

There are three experiment phases. Each phase produces files consumed by the next.

### Phase 1 — Gradient map (`python -m gradients_map run`)

Accumulates a per-parameter saliency score via EMA of gradients over a training
dataset. Outputs a `.safetensors` file under `biology/`.

Three variants:
- **`gradient_ema`** — `EMA(g_t)`, signed. Baseline.
- **`gradient_ema --abs-grad`** — `EMA(|g_t|)`. Prevents sign-cancellation.
  Recommended default.
- **`random`** — i.i.d. Uniform[0,1) noise. Control condition; no GPU needed.

### Phase 2 — Pruning sweep (`python sweep_eval_temp.py run | batch`)

For each sparsity level (default 0–100 % in 5 % steps): zero the
lowest-scoring weights, measure validation cross-entropy loss on 512 samples,
and grade 32 model generations with LLM judges. Logs everything to WandB.

Two pruning criteria:
- **`gradient`** — score = `|saliency|`
- **`taylor`** — score = `|saliency × weight|` (first-order Taylor; recommended)

Use `batch` to dispatch all `.safetensors` files × both criteria across
multiple GPUs automatically. `run` handles a single condition.

### Phase 3 — Prune-and-recover sweep (`python prune_and_maybe_recover_sweep.py`)

Binary search over sparsity to find the highest sparsity level at which the
model can recover to a quality threshold via SFT fine-tuning. Uses the same
saliency files as Phase 2.

---

## Discovering CLI options

Always run `--help` before writing a new experiment script to get the current
option list:

```bash
cd experiments/saliency_pruning/toy_sweep_2026_03_14
export PYTHONPATH=.

# Phase 1
conda run -n saescoping python -m gradients_map run --help
conda run -n saescoping python -m gradients_map batch --help

# Phase 2
conda run -n saescoping python sweep_eval_temp.py run --help
conda run -n saescoping python sweep_eval_temp.py batch --help

# Phase 3
conda run -n saescoping python prune_and_maybe_recover.py --help
conda run -n saescoping python prune_and_maybe_recover_sweep.py --help
```

Show the user the `--help` output so they can choose the right flags for their
experiment before you write any shell scripts.

---

## Writing a new experiment run folder

### Directory naming convention

```
experiments/saliency_pruning/toy_sweep_2026_03_14/
  gradient_map_runs_YYYY_MM_DD/     ← Phase 1 scripts
  sweep_runs_YYYY_MM_DD/            ← Phase 2 scripts
  prune_recover_runs_YYYY_MM_DD/    ← Phase 3 scripts (if needed)
```

Use today's date. One folder per experimental session. Each folder contains:
- One `.sh` file per run condition (or one `run_batch.sh` for batch mode)
- One `README.md` explaining the design, how to run, and expected outputs

### Shell script template

Every script must follow this exact pattern:

```bash
#!/usr/bin/env bash
# run_<description>.sh
#
# <One-line purpose statement>
# <Comparison purpose: what question this run answers>
#
# Output: <output file or dir>
# Date: YYYY-MM-DD

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$EXPERIMENT_DIR"
conda run --no-capture-output -n saescoping python -u <MODULE_OR_SCRIPT> \
    --<flag>  <value> \
    --<flag>  <value> \
    "$@"
```

Key rules:
- `set -euo pipefail` always
- Compute `SCRIPT_DIR` and `EXPERIMENT_DIR` so scripts run from any cwd
- Export `PYTHONPATH="$EXPERIMENT_DIR"` (the `toy_sweep_2026_03_14/` dir)
- Use `conda run --no-capture-output -n saescoping`
- End with `"$@"` to forward extra flags
- Use `python -u -m gradients_map` for Phase 1 (it is a package)
- Use `python -u "$EXPERIMENT_DIR/sweep_eval_temp.py"` for Phase 2

### WandB naming (from CLAUDE.md)

- **Project**: `saescoping--pruning--{script_name_without_py}`
  - Phase 1 → `saescoping--pruning--gradients_map`
  - Phase 2 → `saescoping--pruning--sweep_eval_temp`
  - Phase 3 → `saescoping--pruning--prune_and_maybe_recover_sweep`
- **Run name**: `YYYY-MM-DD_<short_description>`, e.g. `2026-03-20_ema_gradient_abs`

### README.md template for run folders

Every run folder needs a `README.md` with:
1. One-sentence summary of the experiment
2. Table: script → mode → key parameter → output file
3. What each comparison tells you (the scientific question)
4. How to run (copy-paste commands, including `chmod +x`)
5. Default settings table
6. Expected outputs (WandB run names, output files/dirs)

---

## Existing patterns to follow

### `gradient_map_runs_2026_03_20/` (Phase 1 — individual scripts)

Three scripts: `run_ema_gradient.sh`, `run_ema_gradient_abs.sh`, `run_random.sh`.
Each outputs one `.safetensors` to `biology/`. Pass `--wandb-run-name` explicitly.

### `sweep_runs_2026_03_19/` (Phase 2 — individual scripts per condition)

Four scripts for a 2×2 design (saliency × criterion). Use this pattern when:
- You need fine-grained per-run control
- You will run conditions on specific GPU assignments manually

### `sweep_runs_2026_03_20/` (Phase 2 — single batch script, preferred)

One `run_batch_sweep.sh` calling `sweep_eval_temp.py batch`. Discovers all
`.safetensors` in `biology/` automatically, distributes across GPUs, skips
completed runs. Prefer this pattern for new sweep experiments.

---

## Answering "how does X work" questions

When the user asks about algorithm details, explain using these concepts:

| Question | Answer |
|----------|--------|
| Why EMA? | Smooths noisy per-batch gradients. Higher `--beta` (e.g. 0.99) → smoother but slower to converge |
| Why `--abs-grad`? | Prevents gradients that flip sign across examples from cancelling out, giving near-zero saliency to genuinely important weights |
| Why Taylor criterion? | `|grad × weight|` approximates the first-order change in loss when the weight is zeroed — better proxy for output impact than `|grad|` alone |
| Why a random baseline? | Validates that the saliency ordering provides real signal; any method that cannot beat random pruning is not useful |
| What is `--precision 0.05`? | Sparsity grid step size; `0.05` → 21 evaluation points at 0%, 5%, …, 100% |

For deeper details, point the user to `EXPERIMENT_README.md` and run
`--help` on the relevant script to show them all available options.

---

## Workflow for a new experiment session

1. Read `EXPERIMENT_README.md` and `CLAUDE.md` (always do this first).
2. Run `--help` on the relevant scripts; show output to user.
3. Clarify with the user: which saliency maps? which criteria? which GPUs? which sparsity grid?
4. Create `<phase>_runs_YYYY_MM_DD/` with all `.sh` files and a `README.md`.
5. Verify scripts with `bash -n <script>.sh` (dry syntax check).
6. Remind the user to `chmod +x <dir>/*.sh` before running.
