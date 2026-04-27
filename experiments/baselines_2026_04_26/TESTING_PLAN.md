# Testing Plan — Saliency Baselines

Temporary doc. Delete once baselines are validated.

## Goal

Reliable, reproducible saliency pruning baselines that:
1. Run end-to-end without errors (calibrate → prune → recover → elicit → evaluate)
2. Produce results qualitatively consistent with Wanda (Sun et al., 2023)
3. Work on our target models within our GPU constraints (≤13B, A100 80GB)

## Target models

| Model | ~GPU mem (bf16) | Comparable to |
|---|---|---|
| `google/gemma-2-9b-it` | ~18 GB | LLaMA-7B/13B scale |
| `google/gemma-3-12b-it` | ~24 GB | LLaMA-13B scale |

We do NOT go above 13B.

## Reference: Wanda paper evaluation setup

From Sun et al. 2023 (arxiv:2306.11695):
- **Models:** LLaMA 7B/13B/30B/65B, LLaMA-2 7B/13B/70B
- **Calibration:** 128 samples from C4, full context length
- **Sparsities:** 50%, 60%, 70%, 80% unstructured; 4:8 and 2:4 N:M
- **Eval — perplexity:** WikiText validation set
- **Eval — zero-shot:** 7 tasks from EleutherAI LM Harness
- **Baselines:** Magnitude pruning, SparseGPT
- **Recovery:** Tested LoRA (r=8, 12h on 1 GPU) and full fine-tuning
- **Key number:** LLaMA-7B at 50% → Wanda 7.26 ppl vs dense 5.68 ppl (WikiText)

We use different models (Gemma, not LLaMA), so we can't match numbers
exactly. But qualitative patterns must hold: Wanda ≈ Taylor > Gradient >
Random, and recovery should close most of the gap at moderate sparsity.

---

## Phase 1: Does the code run?

All integration tests include LLM judge with `--n-judge-samples 10` to
exercise the full pipeline. This derisks judge failures in production runs.

### 1a. Unit tests (no model needed)

- [ ] `pytest sae_scoping/infrastructure/test_scheduler.py -v`
- [ ] `pytest sae_scoping/examples/test_scheduler_gpu.py -v`
- [ ] `./run.sh calibrate --help` and `./run.sh elicit --help` both load

### 1b. Smoke test: calibrate + prune + recover (gemma-2-2b-it, ~10 min)

```bash
export WANDB_MODE=offline
source /path/to/.env  # for OPENAI_API_KEY (judge needs it)

./run.sh calibrate worker \
    --method wanda --model-id google/gemma-2-2b-it --domain biology \
    --sparsity-levels 0.5 --n-calibration 32 --n-recovery 64 \
    --max-seq-len 512
```

**Check:**
- [ ] Saliency cache written: `artifacts/google--gemma-2-2b-it/biology/wanda_saliency.safetensors`
- [ ] Masks written: `artifacts/.../wanda/sp_0.50/masks.safetensors`
- [ ] Recovered checkpoint written: `artifacts/.../wanda/sp_0.50/recovered/`
- [ ] `metrics.json` has both pruned and recovered losses
- [ ] Recovered test loss < pruned test loss

### 1c. Smoke test: elicitation with judge (~10 min)

Requires 1b artifacts.

```bash
./run.sh elicit \
    --method wanda --model google/gemma-2-2b-it \
    --in-domain biology --ood-domain chemistry --sparsity 0.5 \
    --n-train 64 --n-eval 32 --n-judge-samples 10
```

**Check:**
- [ ] PGD training runs without sparsity violations
- [ ] `elicitation_results.json` written with in-domain and OOD losses
- [ ] Judge scores present (not errors) for both in-domain and OOD
- [ ] Judge scores directionally consistent with loss (lower loss ≈ higher score)

### 1d. Manifest round-trip

```bash
./run.sh calibrate launch --gpus 0 --methods wanda --domains biology \
    --models google/gemma-2-2b-it --sparsity-levels 0.3,0.5 \
    --n-calibration 32 --n-recovery 64

./run.sh elicit --launch --gpus 0 --dry-run
# Should list 6 jobs: 2 sparsities × 3 OOD domains
```

---

## Phase 2: Works at target scale

### 2a. gemma-2-9b-it — memory fit and basic correctness

```bash
./run.sh calibrate worker \
    --method wanda --model-id google/gemma-2-9b-it --domain biology \
    --sparsity-levels 0.5

./run.sh elicit \
    --method wanda --model google/gemma-2-9b-it \
    --in-domain biology --ood-domain chemistry --sparsity 0.5 \
    --n-train 64 --n-eval 32 --n-judge-samples 10
```

If OOM during recovery/elicitation:
```bash
--sft-overrides '{"per_device_train_batch_size": 1, "gradient_accumulation_steps": 2}'
```

Record peak GPU memory. Update `sft_defaults.yaml` per-model section.

### 2b. gemma-3-12b-it — same, expect tighter memory

Likely needs batch_size=1 and possibly `gradient_checkpointing: true`.
May also need `attn_implementation: "flash_attention_2"` instead of "eager".

### 2c. All four saliency methods on 9B

One domain (biology), one sparsity (0.5):
- [ ] Wanda — single forward pass calibration
- [ ] Random — no calibration (control baseline)
- [ ] Taylor — needs EMA gradient map (runs GradCollectTrainer, ~10 min)
- [ ] Gradient — shares EMA gradient cache with Taylor

Taylor and gradient share `ema_grads.safetensors`, so run Taylor first.

---

## Phase 3: Results match prior work (qualitatively)

### 3a. WikiText perplexity evaluation

**NOT YET IMPLEMENTED.** Need a script that:
1. Loads a pruned/recovered model
2. Computes perplexity on WikiText-2 validation set (standard LM eval)
3. Reports in a format comparable to Wanda Table 1

This is different from our current `compute_loss` which uses StemQA with
chat templates. WikiText perplexity is raw next-token prediction on plain
text — the standard metric for comparing pruning methods.

**Script needed:** `eval_wikitext_ppl.py` that takes a model checkpoint
path and prints perplexity. Should work on both the base model (dense
baseline) and pruned/recovered checkpoints.

### 3b. Zero-shot task evaluation

**NOT YET IMPLEMENTED.** Wanda uses EleutherAI LM Harness for 7 zero-shot
tasks. We should run at least a subset:

```bash
# Using lm-evaluation-harness (pip install lm-eval)
lm_eval --model hf --model_args pretrained=./artifacts/.../recovered \
    --tasks winogrande,hellaswag,arc_easy,piqa \
    --batch_size 4
```

**Script needed:** wrapper that runs lm-eval on our checkpoints and
collects results into a comparison table.

### 3c. Expected qualitative patterns

At 50% unstructured sparsity on gemma-2-9b-it:

| Check | Expected | How to verify |
|---|---|---|
| Method ranking | Wanda ≈ Taylor > Gradient > Random | Compare WikiText ppl across methods |
| Pruning degrades performance | ppl increases ~1.3-2x at 50% | Compare pruned vs dense ppl |
| Recovery helps | Recovered ppl < pruned ppl | Compare recovered vs pruned |
| Higher sparsity = worse | ppl at 70% >> ppl at 50% | Sparsity sweep curve is convex |
| Random is worst | Random ppl > all others | Sanity check — if not, something is broken |

### 3d. Comparison table format

Target output (fill in as results come in):

```
Model: google/gemma-2-9b-it | Domain: biology | Metric: test loss (StemQA)

Sparsity  Dense  Wanda  Taylor  Gradient  Random
0.0       ___    -      -       -         -
0.3       -      ___    ___     ___       ___
0.5       -      ___    ___     ___       ___
0.7       -      ___    ___     ___       ___

After recovery (PGD SFT, 3 epochs):
0.3       -      ___    ___     ___       ___
0.5       -      ___    ___     ___       ___
0.7       -      ___    ___     ___       ___
```

Same table for WikiText perplexity once eval script exists.

---

## Phase 4: Elicitation validation

### 4a. Does elicitation recover OOD performance?

For each (in_domain, ood_domain) pair:
- OOD loss should decrease after elicitation (elicitation works)
- In-domain loss should increase somewhat (catastrophic forgetting)
- The ratio measures scoping robustness

### 4b. PGD integrity

- `validate_sparsity=True` checks every step — any violation is a hard fail
- Additionally verify: `count_zeros(model)` before and after elicitation
  should return identical zero counts

### 4c. Judge evaluation on elicited models

```bash
./run.sh elicit \
    --method wanda --model google/gemma-2-9b-it \
    --in-domain biology --ood-domain chemistry --sparsity 0.5 \
    --n-train 512 --n-eval 200 --n-judge-samples 10
```

Check judge scores are directionally consistent with loss.

---

## Phase 5: Full grid

Once phases 1-4 pass:

```bash
# Calibrate + recover: 4 methods × 4 domains × 3 sparsities = 48 jobs per model
./run.sh calibrate launch --gpus 1,6 \
    --methods wanda,random,taylor,gradient \
    --models google/gemma-2-9b-it \
    --domains biology,chemistry,math,physics \
    --sparsity-levels 0.3,0.5,0.7

# Elicit: each domain → 3 OOD targets × 3 sparsities × 4 methods = 144 jobs
./run.sh elicit --launch --gpus 1,6

# Repeat for gemma-3-12b-it once hyperparameters are tuned
```

---

## Scripts still needed

| Script | Purpose | Priority |
|---|---|---|
| `eval_wikitext_ppl.py` | WikiText-2 perplexity on any checkpoint | High — standard pruning comparison metric |
| `eval_zeroshot.py` | Wrapper around lm-evaluation-harness | Medium — Wanda uses this, adds credibility |
| `collect_results.py` | Walk artifact dir, build comparison tables | Medium — automates Phase 3d |

## Hyperparameter decisions still needed

| Parameter | Current | Notes |
|---|---|---|
| Recovery epochs / LR | 3 / 1e-5 | May need tuning per model. Wanda paper uses LoRA r=8 for 12h. |
| Elicitation epochs / LR | 3 / 1e-5 | Enough to recover OOD? Too much = overfitting. |
| max_seq_len | 1024 | Wanda uses full context. Check StemQA answer length distribution. |
| n_calibration | 128 | Matches Wanda paper. Should be fine. |
| n_recovery / n_train | 500 / 512 | May need more for harder domains or higher sparsity. |
| 12B batch size | 2 | Likely needs 1 + grad_accum. Profile first. |
| 12B attn impl | eager | May need flash_attention_2 for memory. |

## Known risks

| Risk | Severity | Mitigation |
|---|---|---|
| OOM on 12B recovery/elicitation | High | Tune batch size, grad accum, grad checkpointing in sft_defaults.yaml |
| WikiText eval not implemented | High | Blocks quantitative comparison to Wanda paper |
| `sft_defaults.yaml` not pushed to GPU machine | High | Must commit before testing |
| Recovery silently swallows exceptions | Medium | Check metrics.json for `recovered_model_dir: null` |
| `load_nonoverlapping_splits(n_calibration=0)` untested | Medium | Used in elicitation eval |
| Gemma vs LLaMA makes direct comparison impossible | Medium | Focus on qualitative patterns, not exact numbers |
| Taylor/gradient need enough data for GradCollectTrainer | Medium | Default 4096 samples — verify StemQA subsets are large enough |
