# Pruning experiments — 2026-03-21

Three scripts covering the full pruning pipeline: raw pruning, single-sparsity
prune-and-recover, and binary-search prune-and-recover sweep. All use the
recommended absolute-EMA saliency map (`biology/ema_grads_abs.safetensors`)
from `gradient_map_runs_2026_03_20/` with the Taylor criterion.

## Prerequisite

`biology/ema_grads_abs.safetensors` must exist.  Run
`gradient_map_runs_2026_03_20/run_ema_gradient_abs.sh` first if it does not.

---

## The three scripts

| Script | Tool | Purpose | Output |
|--------|------|---------|--------|
| `run_prune_only.sh` | `prune.py` | Prune 70 % of weights, save model | `pruned_models_2026_03_21/taylor_70pct/` |
| `run_prune_and_recover.sh` | `prune_and_maybe_recover.py` | Prune 50 %, recover via SFT if loss > 2.5 | `results/prune_recover_50pct.json` |
| `run_prune_and_recover_sweep.sh` | `prune_and_maybe_recover_sweep.py` | Binary-search max sparsity recoverable to loss ≤ 2.5 | WandB + `sweep_output_2026_03_21/` |

### What each script tells you

| Script | Question answered |
|--------|------------------|
| `run_prune_only.sh` | What does the model look like at 70 % sparsity *without* any attempt at recovery? (Sets expectations.) |
| `run_prune_and_recover.sh` | Can a single chosen sparsity (50 %) be recovered to acceptable quality? How many SFT steps does it take? |
| `run_prune_and_recover_sweep.sh` | What is the *highest* sparsity level at which quality can be recovered to ≤ 2.5 loss? |

---

## Running

```bash
# Make executable (first time only)
chmod +x pruning_2026_03_21/*.sh

# 1. Inspect raw pruning effect (no GPU training — model is not trained)
./pruning_2026_03_21/run_prune_only.sh

# 2. Prune + recover at a fixed sparsity
./pruning_2026_03_21/run_prune_and_recover.sh

# 3. Binary-search sweep for maximum recoverable sparsity
./pruning_2026_03_21/run_prune_and_recover_sweep.sh
```

All scripts forward extra flags via `"$@"`:

```bash
# Adjust sparsity for a quick smoke-test
./pruning_2026_03_21/run_prune_only.sh --sparsity 0.3

# Change threshold for prune-and-recover
./pruning_2026_03_21/run_prune_and_recover.sh --threshold 2.8

# Change threshold for binary-search sweep
./pruning_2026_03_21/run_prune_and_recover_sweep.sh --threshold 2.8

# Skip model save in prune-only (no disk cost)
./pruning_2026_03_21/run_prune_only.sh --output-dir ""
```

---

## Default settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Saliency map | `biology/ema_grads_abs.safetensors` | Abs-EMA; no sign-cancellation |
| Saliency criterion | `taylor` | `\|grad × weight\|` — recommended |
| Metric | `loss` | Validation cross-entropy |
| Loss threshold | `2.5` | **Adjust** to ~10 % above the 0 %-sparsity baseline from `sweep_eval_temp.py` |
| Eval samples | 128 | Used for all three scripts |
| Recovery samples | 512 | Training data for recovery SFT |
| Recovery batch size | 4 | |
| `run_prune_only` sparsity | 0.70 | 70 % weights zeroed |
| `run_prune_and_recover` sparsity | 0.50 | 50 % weights zeroed |
| `run_prune_and_recover_sweep` binary-search steps | 8 | `k_min=0.0`, `k_max=1.0` |
| Max recovery steps per sweep step | 200 | |

### ⚠️ Calibrate `--threshold` before running

The threshold `2.5` is a placeholder. To set it correctly:

1. Run `sweep_eval_temp.py run` at 0 % sparsity (or read the 0 %-sparsity
   point from an existing WandB run in `sweep_runs_2026_03_20`).
2. Note the `val_loss` value.
3. Set `--threshold` to approximately `baseline_loss × 1.10` (10 % tolerance).

---

## Expected outputs

| Output | Location |
|--------|----------|
| Pruned model weights | `pruned_models_2026_03_21/taylor_70pct/` (~18 GB) |
| Prune-and-recover JSON | `pruning_2026_03_21/results/prune_recover_50pct.json` |
| Sweep checkpoints | `sweep_output_2026_03_21/` |
| Sweep WandB run | `saescoping--pruning--prune_and_maybe_recover_sweep` / `2026-03-21_abs_ema_taylor_loss_sweep` |

### JSON result schema (`prune_and_maybe_recover.py`)

```json
{
  "sparsity": 0.5,
  "n_weights_zeroed": 2500000000,
  "metric_type": "loss",
  "metric_before_recovery": 2.71,
  "metric_after_recovery": 2.43,
  "recovery_steps": 150,
  "recovery_stopped_early": true
}
```

### WandB metrics (`prune_and_maybe_recover_sweep.py`)

Each binary-search step logs:
- `metric_before_recovery` and `metric_after_recovery` vs. `sparsity`
- `recovery_steps` per step
- Final `best_sparsity` summary

---

## Expected runtimes (1× A100 80 GB)

| Script | Approximate time |
|--------|-----------------|
| `run_prune_only.sh` | ~5 min (model load + forward eval only) |
| `run_prune_and_recover.sh` | 30 min – 2 h (depends on how many recovery steps) |
| `run_prune_and_recover_sweep.sh` | 4 – 16 h (8 steps × up to 200 recovery steps each) |
