# Pruning sweep runs — 2026-03-20

Single batch script that sweeps all saliency maps in `biology/` against both
pruning criteria, distributed across two GPUs.

## Why this replaces sweep_runs_2026_03_19

| Issue | 2026-03-19 | 2026-03-20 |
|-------|-----------|-----------|
| `HFGenerator` caching | Same generator reused for all sparsity levels → stale responses | Fresh instance per sparsity level |
| WandB x-axis | Auto-incrementing step counter | Mapped to sparsity fraction |
| Generation persistence | Not saved | Written to `sweep_generations_2026_03_20/<run>/` |
| Redundant reruns | Always reruns | Skips completed runs (non-empty output dir) |
| Parallelism | One script per run, manual GPU assignment | `batch` dispatches all runs across GPUs automatically |

## Runs produced

| Saliency file | Criterion | WandB run name |
|---------------|-----------|----------------|
| `ema_grads_2026_03_15` | gradient | `ema_grads_2026_03_15_gradient` |
| `ema_grads_2026_03_15` | taylor | `ema_grads_2026_03_15_taylor` |
| `random_2026_03_15` | gradient | `random_2026_03_15_gradient` |
| `random_2026_03_15` | taylor | `random_2026_03_15_taylor` |
| `random` | gradient | `random_gradient` |
| `random` | taylor | `random_taylor` |

6 runs total, distributed round-robin across CUDA devices 2 and 3 (3 per GPU).

Any new `.safetensors` files added to `biology/` (e.g. `ema_grads.safetensors`
once the gradient-map screens finish) will be picked up automatically on the
next run — existing completed runs are skipped unless `--force` is passed.

## Running

```bash
chmod +x sweep_runs_2026_03_20/run_batch_sweep.sh

# Run from the experiment root (saescoping env must be active)
./sweep_runs_2026_03_20/run_batch_sweep.sh

# Force-rerun everything (e.g. after a bug fix)
./sweep_runs_2026_03_20/run_batch_sweep.sh --force

# Loss-only pass (skip LLM judge API calls)
./sweep_runs_2026_03_20/run_batch_sweep.sh --no-generation
```

Extra flags are forwarded to `sweep_eval_temp.py batch` unchanged.

## Default settings

| Parameter | Value |
|-----------|-------|
| Sparsity grid | 0%, 5%, …, 100% (21 levels, `--precision 0.05`) |
| Loss samples | 512 (128 batches of 4) |
| Generation samples | 32 per level |
| Max new tokens | 256 |
| GPUs | cuda:2, cuda:3 |
| WandB project | `sae-scoping-pruning` |
| Output dir | `sweep_generations_2026_03_20/<run_name>/` |

## Expected WandB outputs

Plot `val_loss` and `generation/overall_mean_score` vs `sparsity` for all runs
on the same axes.  The two `ema_grads_2026_03_15_*` runs should diverge from
their matching `random_*` controls as sparsity increases — if gradient-based
saliency is working, the EMA runs should degrade more slowly.
