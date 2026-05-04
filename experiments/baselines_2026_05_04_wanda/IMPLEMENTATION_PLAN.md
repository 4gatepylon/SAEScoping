# Implementation plan: adapting Wanda for Gemma2/Gemma3

This document describes how to refactor the copied Wanda code to work with Gemma-2 and Gemma-3 models, integrate with the project's existing patterns, and improve readability. The original files are preserved unmodified; refactoring will happen in subsequent commits.

## Phase 1: Gemma compatibility (must-have)

### 1.1 Sequence length control

**Problem**: `model.seqlen = model.config.max_position_embeddings` gives 8192 for Gemma-2 and 131072 for Gemma-3. The code allocates `(nsamples, seqlen, hidden_size)` tensors — at 131072 this will OOM immediately.

**Fix**: Add a `--max_seq_len` CLI argument (default 2048). Set `model.seqlen` to this value instead of `max_position_embeddings`. The Wanda paper used 2048 for calibration; there's no reason to go longer for calibration data.

### 1.2 Hardcoded buffer size in `prepare_calibration_input`

**Problem**: `inps = torch.zeros((128, model.seqlen, ...))` ignores `args.nsamples`.

**Fix**: Replace `128` with `args.nsamples`. This is a one-line fix but critical for correctness.

### 1.3 Tokenizer

**Problem**: `use_fast=False` in `main.py`. Gemma models require the fast tokenizer.

**Fix**: Remove `use_fast=False` (let HuggingFace auto-detect), or set `use_fast=True` explicitly.

### 1.4 Precision

**Problem**: `torch_dtype=torch.float16` is hardcoded. Gemma-3 is trained in `bfloat16`.

**Fix**: Use `torch_dtype="auto"` to let the model config choose, or add a `--dtype` flag.

### 1.5 Layer path

**Problem**: `model.model.layers` is hardcoded throughout `prune.py` and `main.py`.

**Status**: This actually works for Gemma-2 and Gemma-3 (`GemmaForCausalLM` uses `model.model.layers`). No change needed here, but worth verifying at runtime.

### 1.6 Attention implementation

**Problem**: Gemma-2 has sliding window attention and requires `attn_implementation="eager"` for reliable single-GPU inference (same issue we hit in the narrow baselines).

**Fix**: Add `attn_implementation="eager"` to `AutoModelForCausalLM.from_pretrained()`, or use the `load_model_kwargs` pattern from `shared.py`.

### 1.7 Position IDs in Catcher

**Problem**: The `Catcher` class in `prepare_calibration_input` captures `position_ids` from `kwargs`. This works for LLaMA and should work for Gemma, but should be verified.

**Status**: Likely fine — Gemma models pass `position_ids` through the same kwargs path. Verify at smoke-test time.

## Phase 2: Integration with project patterns (should-have)

### 2.1 Model/dataset allowlist

Integrate `SUPPORTED_MODEL_PATTERNS` and `SUPPORTED_DATASETS` from `shared.py` (same pattern as the narrow baselines). Currently the code accepts any model string.

### 2.2 Configurable calibration dataset

**Problem**: `prune_wanda` hardcodes `get_loaders("c4", ...)`. For our experiments we need to calibrate on StemQA (domain-specific calibration).

**Fix**: Add `--calibration_dataset` CLI argument. Extend `get_loaders` in `data.py` to handle `4gate/StemQAMixture` (with `--dataset_config` for subject).

### 2.3 CLI migration to Click

Per project conventions (CLAUDE.md), replace `argparse` with Click. The narrow baselines haven't done this yet either, so this is lower priority — could be done as a shared refactor across both baselines.

### 2.4 Output directory pattern

Replace `--save` / `--save_model` with `--output_base_dir`, matching the narrow baselines pattern.

### 2.5 Shared helpers

Use `validate_args`, `safe_batch_size`, and `load_model_kwargs` from `shared.py` instead of reimplementing model loading.

## Phase 3: Code cleanup (nice-to-have)

### 3.1 Remove unused pruning methods

Delete `prune_magnitude`, `prune_sparsegpt`, `prune_ablate` from `prune.py`. Remove `lib/sparsegpt.py` and `lib/ablate.py`. Remove the corresponding `--prune_method` choices from the CLI. We only need `prune_wanda`.

### 3.2 Deduplicate the Catcher pattern

The `Catcher` class + try/except loop appears identically in `prune_wanda`, `prune_sparsegpt`, and `prune_ablate`. After removing the latter two, there's only one copy left, but it's also duplicated in `prepare_calibration_input`. Extract to a shared helper.

### 3.3 Modernize evaluation

`eval_zero_shot` depends on a custom fork of `lm-evaluation-harness` (commit `df3da98`, 2023). Options:
- Port to current `lm-evaluation-harness` API (preferred, but significant work)
- Use our own eval pipeline if we have one
- Drop zero-shot eval from this script and do it as a separate step

WikiText-2 perplexity eval (`eval_ppl`) is self-contained and should work as-is.

### 3.4 Remove `--use_variant`

The Wanda "variant" (appendix experiment with alpha-based cumulative thresholding) is not the published method. Remove to reduce code surface.

### 3.5 Remove `eval_ppl_wikitext_train`

Dead code in `eval.py` — not called from anywhere.

## Phase 4: Testing

### 4.1 Smoke test (immediate, pre-refactor)

Verify the unmodified copied code runs on a small LLaMA model:
```bash
python main.py \
  --model NousResearch/Llama-3.2-1B \
  --prune_method wanda \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --nsamples 10 \
  --save /tmp/smoke_wanda_llama
```

### 4.2 Gemma smoke test (post Phase 1)

```bash
python main.py \
  --model google/gemma-2-2b-it \
  --prune_method wanda \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --nsamples 10 \
  --max_seq_len 512 \
  --save /tmp/smoke_wanda_gemma2
```

### 4.3 StemQA calibration test (post Phase 2)

```bash
python main.py \
  --model google/gemma-2-2b-it \
  --prune_method wanda \
  --sparsity_ratio 0.5 \
  --sparsity_type unstructured \
  --calibration_dataset 4gate/StemQAMixture \
  --dataset_config biology \
  --nsamples 10 \
  --max_seq_len 512 \
  --save /tmp/smoke_wanda_gemma2_stemqa
```

## Priority ordering

1. Phase 1 (Gemma compat) — blocks all real experiments
2. Phase 2.2 (StemQA calibration) — blocks domain-specific experiments
3. Phase 3.1 (remove unused methods) — reduces confusion, makes code ~60% shorter
4. Everything else — can be done incrementally
