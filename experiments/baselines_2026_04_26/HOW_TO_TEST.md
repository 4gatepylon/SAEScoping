# How to Test — Saliency Baselines

Quick reference. Full plan: `TESTING_PLAN.md`.

## Setup

```bash
export WANDB_MODE=offline
source /path/to/.env  # OPENAI_API_KEY for judge
```

## Phase 1 — Does it run? (small model, ~10 min each)

**1a. Unit tests (no GPU):**
```bash
pytest sae_scoping/infrastructure/test_scheduler.py -v
pytest sae_scoping/examples/test_scheduler_gpu.py -v
./run.sh calibrate --help
./run.sh elicit --help
```

**1b. Calibrate + prune + recover smoke test:**
```bash
./run.sh calibrate worker \
    --method wanda --model-id google/gemma-2-2b-it --domain biology \
    --sparsity-levels 0.5 --n-calibration 32 --n-recovery 64 \
    --max-seq-len 512
```
Check: saliency cache, masks, recovered checkpoint, `metrics.json` (recovered loss < pruned loss).

**1c. Elicitation smoke test (needs 1b artifacts):**
```bash
./run.sh elicit \
    --method wanda --model google/gemma-2-2b-it \
    --in-domain biology --ood-domain chemistry --sparsity 0.5 \
    --n-train 64 --n-eval 32 --n-judge-samples 10
```
Check: no sparsity violations, `elicitation_results.json` written, judge scores consistent with loss.

**1d. Manifest round-trip:** run `calibrate launch` then `elicit --launch --dry-run`; expect 6 jobs (2 sparsities × 3 OOD domains).

## Phase 2 — Target scale

- **2a. gemma-2-9b-it:** same commands with `--model google/gemma-2-9b-it`. If OOM: `--sft-overrides '{"per_device_train_batch_size": 1, "gradient_accumulation_steps": 2}'`. Record peak GPU.
- **2b. gemma-3-12b-it:** likely needs `batch_size=1`, `gradient_checkpointing: true`, possibly `flash_attention_2`.
- **2c. All four methods (wanda/random/taylor/gradient) on 9B at sparsity 0.5.** Run Taylor before Gradient (shared `ema_grads.safetensors`).

## Phase 3 — Qualitative match to Wanda paper

Expected ranking at 50% sparsity: **Wanda ≈ Taylor > Gradient > Random**. Recovery should close most of the gap. Random must be worst (sanity).

Missing scripts: `eval_wikitext_ppl.py`, `eval_zeroshot.py`, `collect_results.py`.

## Phase 4 — Elicitation validation

- OOD loss decreases after elicitation; in-domain loss may rise (forgetting).
- `validate_sparsity=True` is hard-fail on violation.
- Verify `count_zeros(model)` is identical pre/post elicitation.
- Judge scores should track loss directionally.

## Phase 5 — Full grid

Once 1–4 pass:
```bash
./run.sh calibrate launch --gpus 1,6 \
    --methods wanda,random,taylor,gradient \
    --models google/gemma-2-9b-it \
    --domains biology,chemistry,math,physics \
    --sparsity-levels 0.3,0.5,0.7
./run.sh elicit --launch --gpus 1,6
```

## Top risks

- OOM on 12B → tune `sft_defaults.yaml`.
- `sft_defaults.yaml` must be committed before remote testing.
- Recovery can silently null out — check `metrics.json` for `recovered_model_dir: null`.
- WikiText eval not implemented — blocks direct quantitative comparison.
