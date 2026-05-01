---
name: run_baselines
description: Plan, configure, and launch sweep-style training and eval jobs in this repo (currently wanda + PGD recovery; whatever pruning/recovery methods we add later). Covers mini-before-full smoke order, memory shaving, context-length and wall-time floors, GPU pinning, and the safety hooks documented in TODOs.md. Use when the user wants to start, design, or troubleshoot a baseline run; skip for ad-hoc one-shot script invocations.
user-invocable: false
---

> ⚠️ **Stale-by-design warning.** Every concrete number, knob, and
> example below comes from one slice of our work — they are NOT laws.
> Before relying on this skill, **ask the user to read SKILL.md and
> update it** with whatever they know that contradicts or supersedes
> these defaults (different pruning method, different model size,
> different host, new memory tricks, better budgets). Treat what's
> here as priors, not verdicts.

## What this skill is for

The repo runs sweep-shaped jobs that follow some variant of
*calibrate → prune → recover → elicit*. Today the only wired-up
pipeline is **wanda + PGD recovery** in
`experiments/baselines_2026_04_29/sweep_wanda.py` (driven by the
matrix scripts under `experiments/baselines_2026_04_30/`), but the
guidance below applies to any future method that fits the same
pipeline shape — random / magnitude / Taylor / unlearning / OOD
elicitation, etc. Treat all specific values (e.g. sparsity grid
`[0.4, 0.5, 0.6, 0.7]`, the gemma layer cutoffs, the per-model batch
decompositions) as **examples we happen to use**, not invariants.

## Pre-flight, in order

Always smoke a **mini** variant first
(`experiments/baselines_2026_04_30/wanda_with_pgd_v1_<model>_<domain>_mini.sh`
for the current pipeline; whatever the analogous "tiny dataset, few
steps, `--no-cache`" wrapper is for a future one). It should finish
in roughly a minute on a single GPU and produce the full artifact
tree per `experiments/baselines_2026_04_29/NAMING.md`. If anything is
missing or it OOMs, fix that before paying for a full run. When the
memory envelope is genuinely unclear (new model size, new host, new
memory knob), invoke the **`oom_grid`** skill before the mini —
`.claude/skills/oom_grid/SKILL.md`. Always pin the physical GPU via
`CUDA_VISIBLE_DEVICES=N` in the shell wrapper because HF Trainer
calls `torch.cuda.set_device("cuda:0")` regardless of where you put
the model.

## Memory and time budgets (these are hard caps; values are examples)

When the mini OOMs or `oom_grid` predicts it will, the levers we
have used so far — *for the wanda + PGD pipeline as one example* —
are: an 8-bit optimizer (`pgd.optim: adamw_bnb_8bit` saves ~63 GB on
a 9B model vs fp32 Adam), `pgd.gradient_checkpointing: true` (~4-5×
activation memory), restricting PGD to late layers via
`pgd.min_layer_idx` (e.g. `31` for gemma-2-9b cuts memory + per-step
validate cost by training only layers 32-41), and rebalancing
`train_batch_size × gradient_accumulation_steps` to keep the
effective batch fixed. Different methods will need different levers
— always check what's actually peaked in `nvidia-smi` before
guessing. **Wall-time cap: a single sweep script should not be
expected to run more than 8 hours total.** If it might, change knobs
(early stopping, smaller `max_steps`, layer-restricted recovery, or
parallelise across GPUs — the `adriano/baselines` branch has a
`--devices` dispatcher; this branch is single-device).

To estimate full wall-time when it's not obvious: the mini already
times every section (calibration, saliency, sweep eval, judge, PGD
training, ...) — **linearly extrapolate each section from its own
mini timing and sum the extrapolations**. Training is almost always
the slowest part, so the dominant term is `mini_pgd_seconds_per_step
× full_max_steps × n_sparsities`, but don't only extrapolate that
one; calibration on `n_calibration=1000` vs mini's `8` is also a
linear blow-up of its own. If the summed estimate exceeds 8 h, kick
back to the knob list above.

## Context length is non-negotiable

`max_seq_len` MUST be `> 256` always, SHOULD be `>= 1024` for any
real run, and ideally `>= 2048` for anything you'd publish or
cite. The *only* legitimate reason to drop below 1024 is **measured
evidence** that the dataset's questions and the LLM-judge prompt
template both fit comfortably under your chosen value (e.g. a
token-length histogram you commit alongside the run's
`metadata.json`). Lowering the seq-len to dodge OOM is the wrong fix
— shave optimizer state or activation memory first. The first place
to look for context-length-relevant prior decisions and pipeline-shape
constraints (early stopping, OOD vs in-domain budget, layer-restricted
PGD, multi-GPU PGD gotchas) is `experiments/baselines_2026_04_29/TODOs.md`
— read it before designing anything new.
