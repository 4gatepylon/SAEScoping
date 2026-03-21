# Controlled Pruning Evaluation

This document explains how to run the saliency-based pruning sweep to understand how aggressively `google/gemma-2-9b-it` can be pruned on the biology dataset before quality degrades.

## Overview

The test has two steps:

1. **Compute a saliency map** (`gradients_map.py`) — run (or reuse) a cached `.safetensors` file that scores each weight by how much it matters for the target dataset.
2. **Sweep pruning levels** (`sweep_eval_temp.py`) — for each sparsity fraction, zero out the lowest-scoring weights, measure validation loss, and grade 32 model generations with LLM judges. All results go to wandb.

Two saliency criteria are supported, and ideally both are run so results can be compared:

| Criterion | Score per weight | When to use |
|-----------|-----------------|-------------|
| `gradient` | `\|grad\|` | Faster; treats all weights equally regardless of magnitude |
| `taylor` | `\|grad × weight\|` | Better proxy for output change; recommended |

A **random baseline** (weights zeroed at random) is also supported to calibrate how much the saliency ordering actually matters.

The gradient map script also supports `--abs-grad`, which accumulates `EMA(|g_t|)` instead of `EMA(g_t)`. This prevents sign-cancellation across training examples from artificially driving parameter saliency scores toward zero. Run both variants (`ema_grads.safetensors` and `ema_grads_abs.safetensors`) to see whether sign-cancellation matters in practice.

---

## Prerequisites

```bash
conda activate saescoping
export PYTHONPATH=experiments/saliency_pruning/toy_sweep_2026_03_14
```

Ensure you have an OpenAI-compatible API key set (used by `grade_chats.py`'s LLM judges).

---

## Step 1 — Compute saliency maps

Only needs to be done once per criterion. The output already exists for `gradient_ema`:

```
biology/ema_grads.safetensors   # EMA gradient map (already computed)
biology/random.safetensors      # random baseline (already computed)
```

To regenerate or produce a fresh map:

```bash
# EMA gradient map — signed  (~hours on 1 GPU, 16 384 biology train examples)
python gradients_map.py run \
    --mode gradient_ema \
    --output-path biology/ema_grads.safetensors \
    --dataset-size 16384 \
    --batch-size 2 \
    --num-epochs 2 \
    --beta 0.95

# EMA gradient map — absolute value (prevents sign-cancellation)
python gradients_map.py run \
    --mode gradient_ema \
    --abs-grad \
    --output-path biology/ema_grads_abs.safetensors \
    --dataset-size 16384 \
    --batch-size 2 \
    --num-epochs 2 \
    --beta 0.95

# Random baseline (seconds — no training required)
python gradients_map.py run \
    --mode random \
    --output-path biology/random.safetensors
```

Key options:

| Option | Default | Notes |
|--------|---------|-------|
| `--dataset-size` | 16 384 | Training examples to accumulate over |
| `--beta` | 0.95 | EMA decay; higher = smoother but slower to converge |
| `--batch-size` | 2 | Reduce if OOM |
| `--num-epochs` | 2 | More epochs → more gradient signal |
| `--abs-grad` | off | Accumulate `EMA(\|g_t\|)` to avoid sign-cancellation |

### Batch mode — run all variants in one command

```bash
# All three variants across two GPUs (skips any that already exist)
python gradients_map.py batch --devices 0,1

# Force recompute of gradient variants only
python gradients_map.py batch \
    --variants gradient_ema,gradient_ema_abs \
    --devices 0,1 \
    --force
```

`batch` accepts the same per-run options as `run` (`--dataset-size`, `--beta`, etc.) and applies them uniformly to every child process.  Available variants: `gradient_ema`, `gradient_ema_abs`, `random`.

---

## Step 2 — Run the pruning sweep

Default settings produce **21 evaluation points** at sparsity 0%, 5%, 10%, …, 100% (`--precision 0.05`). Each point: 512 validation loss samples + 32 graded generations.

### Batch mode — one command for all conditions

```bash
# Run all .safetensors files in biology/ × both criteria across 4 GPUs
python sweep_eval_temp.py batch --saliency-dir biology/ --devices 0,1,2,3

# Loss-only pass, force-rerun everything
python sweep_eval_temp.py batch --saliency-dir biology/ --devices 0 \
    --no-generation --force
```

`batch` skips runs whose output directory already contains results; use `--force` to rerun.

---

### Individual runs (run manually for a single condition)

```bash
# 1. Gradient saliency (signed EMA)
python sweep_eval_temp.py run \
    --saliency-path biology/ema_grads.safetensors \
    --saliency-type gradient \
    --wandb-run-name "ema_grads_gradient"

# 2. Taylor saliency (signed EMA; better proxy for output change)
python sweep_eval_temp.py run \
    --saliency-path biology/ema_grads.safetensors \
    --saliency-type taylor \
    --wandb-run-name "ema_grads_taylor"

# 3. Gradient saliency (absolute EMA — avoids sign-cancellation)
python sweep_eval_temp.py run \
    --saliency-path biology/ema_grads_abs.safetensors \
    --saliency-type gradient \
    --wandb-run-name "ema_grads_abs_gradient"

# 4. Taylor saliency (absolute EMA)
python sweep_eval_temp.py run \
    --saliency-path biology/ema_grads_abs.safetensors \
    --saliency-type taylor \
    --wandb-run-name "ema_grads_abs_taylor"

# 5. Random baseline — purely random pruning order
python sweep_eval_temp.py run \
    --saliency-path biology/random.safetensors \
    --saliency-type gradient \
    --wandb-run-name "random_gradient"

# 6. Random × weight magnitude baseline
python sweep_eval_temp.py run \
    --saliency-path biology/random.safetensors \
    --saliency-type taylor \
    --wandb-run-name "random_taylor"

# 7. Loss-only fast pass (no LLM judge API calls; cheap sanity check)
python sweep_eval_temp.py run \
    --saliency-path biology/ema_grads.safetensors \
    --saliency-type taylor \
    --no-generation \
    --wandb-run-name "taylor_loss_only"
```

### All CLI options

```
--saliency-path PATH        Required. .safetensors file from gradients_map.py
--saliency-type [gradient|taylor]
                            Default: gradient
--model-id TEXT             Default: google/gemma-2-9b-it
--dataset-name TEXT         Default: 4gate/StemQAMixture
--dataset-subset TEXT       Default: biology
--n-samples INT             Samples for loss (uses n//batch_size batches). Default: 512
--batch-size INT            Default: 4
--n-generation-samples INT  Samples for generate+grade. Default: 32
--max-seq-len INT           Default: 1024
--max-new-tokens INT        Default: 256
--precision FLOAT           Sparsity grid step size. Default: 0.05 (21 levels)
--sparsity-levels TEXT      Comma-separated explicit levels, e.g. 0.0,0.3,0.7,0.9
                            Overrides --precision when set
--seed INT                  Default: 42
--wandb-project TEXT        Default: sae-scoping-pruning
--wandb-run-name TEXT       Auto-generated if not set
--no-generation             Skip generation + grading (loss only)
--device TEXT               Default: cuda if available
```

### Custom sparsity grid

```bash
# Coarser 11-point sweep (faster)
python sweep_eval_temp.py \
    --saliency-path biology/ema_grads.safetensors \
    --saliency-type taylor \
    --precision 0.1

# Dense sweep in the interesting 50–90 % range
python sweep_eval_temp.py \
    --saliency-path biology/ema_grads.safetensors \
    --saliency-type taylor \
    --sparsity-levels 0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95
```

---

## What to look at in wandb

All runs land in the **`sae-scoping-pruning`** project. For each run you get:

| Metric | Meaning |
|--------|---------|
| `val_loss` vs `sparsity` | When does loss begin to spike? |
| `generation/overall_mean_score` vs `sparsity` | When does generation quality noticeably drop? |
| `generation/answering`, `generation/factual_helpful`, `generation/precise` | Per-judge breakdown |

**Key questions to answer from the plots:**

- At what sparsity does `val_loss` first exceed the unpruned baseline by > 10 %?
- At what sparsity does `generation/overall_mean_score` drop below, say, 0.6?
- Does the Taylor criterion tolerate higher sparsity than gradient magnitude before quality drops?
- How much worse is the random baseline at the same sparsity? (Validates that saliency ordering is doing real work.)

---

## Expected runtimes (1× A100 80 GB)

| Step | Time |
|------|------|
| `gradients_map.py` (16 384 examples, 2 epochs) | ~4–6 h |
| `sweep_eval_temp.py` (21 levels, 512 loss + 32 gen each) | ~2–3 h |
| `sweep_eval_temp.py --no-generation` | ~30–45 min |

---

## Memory notes

- The sweep script saves a full CPU copy of all model weights (~18 GB for 9B bf16) to restore between pruning levels. Ensure the node has ≥ 40 GB CPU RAM.
- If you hit OOM on GPU, lower `--batch-size` to 2 or 1.

---

## Testing

### Unit tests — fast, CPU-only, no model download

Tests all algorithmic components (pruning logic, saliency scoring, EMA hooks, CLI command builders, file I/O) using tiny `nn.Module` instances.

```bash
conda activate saescoping
export PYTHONPATH=experiments/saliency_pruning/toy_sweep_2026_03_14

pytest experiments/saliency_pruning/toy_sweep_2026_03_14/tests/unit/ -v
```

See `tests/unit/README.md` for the full list of what each file covers.

### Integration test — single-GPU / CPU, downloads one small model

Loads `Qwen/Qwen2.5-Math-1.5B-Instruct`, strips it to 1 transformer layer, then runs the full pruning pipeline (`save/restore weights`, `compute_saliency_scores`, `apply_pruning` at multiple sparsity levels).

```bash
conda activate saescoping
export PYTHONPATH=experiments/saliency_pruning/toy_sweep_2026_03_14

python experiments/saliency_pruning/toy_sweep_2026_03_14/tests/test_sweep_pruning.py
```
