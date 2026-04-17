# Pruning sweep runs — 2026-03-19

Four scripts that together form a controlled experiment for saliency-based weight pruning.
Run from any directory; each script resolves its own paths relative to its location.

## The 2×2 design

|                        | Gradient criterion `\|grad\|` | Taylor criterion `\|grad × weight\|` |
|------------------------|-------------------------------|--------------------------------------|
| **EMA saliency map**   | `run_ema_gradient.sh`         | `run_ema_taylor.sh`                  |
| **Random saliency map**| `run_random_gradient.sh`      | `run_random_taylor.sh`               |

**EMA saliency map** — scores computed by accumulating real gradients over 16 384 biology train examples (`biology/ema_grads.safetensors`).

**Random saliency map** — scores are i.i.d. Uniform[0, 1); no gradient signal (`biology/random.safetensors`).

**Gradient criterion** — final pruning score = `|score|` (no weight-magnitude term).

**Taylor criterion** — final pruning score = `|score × weight|`; approximates first-order change in loss when a weight is zeroed.

## What each comparison tells you

| Comparison | Question answered |
|------------|------------------|
| `ema_taylor` vs `random_gradient` | Does gradient saliency + Taylor beat pure chance? (main result) |
| `ema_taylor` vs `ema_gradient` | Does the Taylor `× weight` term improve on plain `\|grad\|`? |
| `ema_gradient` vs `random_gradient` | Does gradient ordering alone beat random, before accounting for weight size? |
| `random_taylor` vs `random_gradient` | Does weighting by `\|weight\|` alone (without real gradients) matter? |

## Running

```bash
# Make executable (first time only)
chmod +x sweep_runs_2026_03_19/*.sh

# Run each condition (ideally in parallel on separate GPUs)
CUDA_VISIBLE_DEVICES=0 ./sweep_runs_2026_03_19/run_ema_taylor.sh
CUDA_VISIBLE_DEVICES=1 ./sweep_runs_2026_03_19/run_ema_gradient.sh
CUDA_VISIBLE_DEVICES=2 ./sweep_runs_2026_03_19/run_random_taylor.sh
CUDA_VISIBLE_DEVICES=3 ./sweep_runs_2026_03_19/run_random_gradient.sh
```

Any extra `sweep_eval_temp.py` flags can be appended and will be forwarded:

```bash
# Quick loss-only pass (no LLM judge API calls)
./sweep_runs_2026_03_19/run_ema_taylor.sh --no-generation

# Finer grid around the interesting 50–90 % region
./sweep_runs_2026_03_19/run_ema_taylor.sh \
    --sparsity-levels 0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95
```

## Default settings per run

| Parameter | Value |
|-----------|-------|
| Sparsity grid | 0%, 5%, 10%, …, 100% (21 points, `--precision 0.05`) |
| Loss samples | 512 (128 batches of 4) |
| Generation samples | 32 |
| Max new tokens | 256 |
| WandB project | `sae-scoping-pruning` |

## Expected outputs in wandb (`sae-scoping-pruning` project)

Runs are named `2026-03-19_{condition}`:

- `2026-03-19_ema_taylor`
- `2026-03-19_ema_gradient`
- `2026-03-19_random_taylor`
- `2026-03-19_random_gradient`

Plot `val_loss` and `generation/overall_mean_score` against `sparsity` for all four runs on the same axes to see at what fraction the signal runs diverge from their random controls.
