# Project Guide

Architecture, modules, branches, and gotchas for the SAEScoping repository.

---

## What This Project Does

SAE Scoping constrains an LLM to a specific knowledge domain (e.g. biology) using Sparse Autoencoders. The pipeline has four stages:

1. **RANK** — Score every weight by importance (saliency methods: WANDA, SparseLLM, Taylor, Gradient, Random)
2. **PRUNE** — Zero out unimportant weights (global or per-row thresholding)
3. **RECOVER** — Fine-tune surviving weights to restore in-domain performance (PGD SFT keeps pruned weights at zero)
4. **ATTACK** — Adversarial evaluation to check if the model leaks out-of-scope knowledge

The `adriano/baselines` branch adds unlearning baselines (Gradient Diff, NPO, RMU) as an alternative to pruning.

---

## Package Layout (`sae_scoping/`)

```
sae_scoping/
├── datasets/
│   ├── qa_datasets.py          # HuggingFace QA dataset loading (STEM-QA biology/math/chem/physics)
│   ├── text_datasets.py        # Older text dataset loading (many TODO(Adriano) annotations)
│   └── messages_datasets.py    # Chat-formatted message datasets
│
├── evaluation/
│   ├── scoping_eval.py         # LLM-judge evaluator: relevance, fluency, ground_truth_similarity
│   ├── spylab_1click_judgement.py  # Base judge framework (from spylab, trojan logic removed)
│   ├── grade_chats/            # Chat grading utilities
│   ├── grade_model.py          # Model-level grading
│   ├── trainer_callbacks.py    # Evaluation callbacks for SFTTrainer
│   └── inference/
│       └── client/             # API/model generators for judge inference
│           ├── api_generator.py           # OpenAI-compatible API client via litellm
│           ├── model_generator.py         # Local HF model inference
│           ├── hardcoded_cache_generator.py  # Cached responses
│           └── length_aware_tokenizer.py  # Token-length-aware truncation
│
├── hyperparameter_optimization/
│   └── binary_search.py        # Binary search for max feasible hyperparameter value
│
├── models/
│   └── sae_enhanced_gemma2.py  # Custom Gemma2Model with SAE hooks on residual stream
│
├── training/
│   ├── weight_pruning.py       # Apply saliency masks to zero out weights (in-place)
│   ├── pgd_trainer.py          # SFTTrainer subclass: re-zeros pruned weights after each step
│   ├── saliency/
│   │   ├── wanda.py            # WANDA: |W| * ||X||_2 (one-shot, calibration-based)
│   │   ├── sparse_llm.py       # SparseLLM: iterative alternating optimization
│   │   ├── taylor.py           # Taylor: |grad * weight|
│   │   ├── grad.py             # Gradient: EMA |grad|
│   │   ├── random.py           # Random baseline
│   │   └── utils.py            # Shared saliency utilities
│   ├── unlearning/
│   │   ├── gradient_diff.py    # GradientDiff: maximize forget loss + minimize retain loss
│   │   ├── npo.py              # NPO: negative preference optimization on forget set
│   │   └── rmu.py              # RMU: representation misdirection via random targets
│   └── sae_enhanced/
│       ├── hooks/
│       │   ├── pt_hooks.py     # PyTorch forward hooks for SAE intervention
│       │   ├── pt_hooks_stateful.py  # Stateful hook variant
│       │   └── sae.py          # SAE loading/application
│       ├── firing_rates.py     # SAE feature activation statistics
│       ├── pruning.py          # SAE-aware pruning (based on feature importance)
│       ├── sae_aware_sft.py    # SFT that accounts for SAE features
│       └── utils.py            # SAE utilities
│
└── utils/
    └── gemma2/
        └── prompting.py        # Gemma-2 chat template helpers
```

---

## Experiment Scripts (`experiments/`)

```
experiments/
├── baselines/
│   ├── sweep_sparsity.py       # Unified sweep: prune/unlearn at multiple levels, evaluate
│   ├── launch_sweeps.py        # Parallel launcher: fan out sweeps across GPUs
│   ├── run_wanda.py            # Single-shot WANDA pruning
│   ├── run_sparse_llm.py       # Single-shot SparseLLM pruning
│   ├── test_wanda.py           # GPU integration tests for WANDA
│   ├── test_sparse_llm.py      # GPU integration tests for SparseLLM
│   └── test_unlearning.py      # GPU integration tests for unlearning
├── plot_eval_accuracy.py       # Plotting evaluation results
├── inspect_firing_rates.py     # SAE firing rate analysis
└── script_*.py                 # Older experiment scripts (2025-era, mostly superseded)
```

---

## Branches

### Active

| Branch | Purpose | Base |
|--------|---------|------|
| `adriano/baselines` | **Current work.** Pruning + unlearning baselines, sweep infrastructure, CPU/GPU tests | Latest on main |
| `adriano/evals_and_datascience` | STEM-QA equivalence judge, top-k overlap analysis, data science notebooks | main |
| `adriano/sae_pruning` | Domain scoping + adversarial elicitation job scripts, multi-GPU runs | main (post-refactor) |

### Stale / Experimental

| Branch | Purpose | Notes |
|--------|---------|-------|
| `adriano/saliency-pruning` | Early saliency pruning experiments | Superseded by `baselines` |
| `adriano/saliency-pruning_cursor_sweep` | PGD recovery sweeps (Cursor-assisted) | Merged concepts into `baselines` |
| `adriano/saliency-pruning_cursor_sweep-backup` | Backup of above | — |
| `adriano/sandboxing` | Claude Code sandbox setup experiments | Infrastructure only |
| `adriano/skills` | Claude Code skill-creator setup | Infrastructure only |
| `cais` / `aruna` / `gemma3` | Collaborator branches (pre-refactor) | Share old package structure |
| `adriano/icml-draft-{1,2,3,4}` | ICML submission iterations | Deprecated, messy squash commits |
| `adriano/wip-first-commit*` | Early refactoring attempts | Deprecated |
| `claude/icml-draft-1/gemma2and3extension` | Claude-generated Gemma 2+3 extension | Stale |

### Package Structure Difference

Pre-refactor branches (`cais`, `aruna`, `gemma3`, `icml-draft-*`) use:
- `trainers/` instead of `training/`
- `utils/hooks/` instead of `training/sae_enhanced/hooks/`

Post-refactor branches (`baselines`, `sae_pruning`, `saliency-pruning_cursor_sweep`) use the current layout.

---

## Key Dependencies

| Package | Version | Role |
|---------|---------|------|
| `torch` | 2.7.1 | Core tensor operations |
| `transformers` | 4.56.1 | Model loading, tokenizers |
| `trl` | 0.22.2 | SFTTrainer (base for PGD trainer), DPOTrainer (NPO) |
| `sae-lens` | 6.22.3 | Sparse Autoencoder loading and hooks |
| `litellm` | 1.74.7 | LLM judge API calls (routes to OpenAI) |
| `peft` | 0.16.0 | LoRA/adapter support |
| `datasets` | 4.0.0 | HuggingFace dataset loading |
| `eai-sparsify` | 1.3.0 | Sparse model utilities |
| `wandb` | 0.21.0 | Experiment tracking |

Python 3.12+ required. Use `/opt/miniconda3/envs/saescoping/bin/python`.

---

## How Things Connect

### Pruning Pipeline

```
calibration data → saliency method (wanda/sparse_llm/taylor/grad/random)
                       ↓
              saliency scores (dict[str, Tensor])
                       ↓
              weight_pruning.compute_keep_masks() → bool masks
                       ↓
              weight_pruning.apply_keep_masks_streaming() → model with zeros
                       ↓
              PGDSFTTrainer(masks=...) → recovery SFT (zeros stay zero)
                       ↓
              scoping_eval.OneClickLLMJudgeScopingEval → judge evaluation
```

### Unlearning Pipeline

```
forget_dataset + retain_dataset → unlearning method (gradient_diff/npo/rmu)
                                       ↓
                              modified model weights (no masks needed)
                                       ↓
                              scoping_eval → judge evaluation
```

### Sweep Infrastructure

```
launch_sweeps.py → spawns N subprocesses (one per GPU)
    ↓
sweep_sparsity.py → iterates sparsity levels [0.1, 0.2, ..., 0.9]
    ↓ for each level:
    ├── compute saliency + prune (or unlearn)
    ├── evaluate loss on retain/forget sets
    ├── optionally run LLM judge
    └── log to wandb
```

---

## Hardcoded Values & Magic Numbers

| Location | Value | What It Does |
|----------|-------|-------------|
| `sweep_sparsity.py:202` | `seed=42` | RNG seed, not in cache key — different seeds collide in cache |
| `sweep_sparsity.py:67` | cache filename | Ignores `n_calibration`, `max_seq_len`, dataset — stale cache reuse |
| `launch_sweeps.py:96` | 2B model tuning | Unknown models silently get Gemma-2-2B hyperparameters |
| `rmu.py:154-197` | `_get_hidden_size` etc. | Calls undefined names (missing underscore prefix) |
| `test_unlearning_cpu.py` | `hidden_size=64, num_hidden_layers=2` | Tiny model config for CPU tests |
| `test_wanda_cpu.py` | `num_attention_heads=4, intermediate_size=128` | Same tiny model config |
| `scoping_eval.py` | `{0, 1, 2}` judge scores | Hard-assert on integer scores — fragile if template changes |
| `weight_pruning.py:90` | `save_original_weights` | Clones ALL parameters (~18GB at 9B) |

---

## Overlapping Functionality

### Saliency Methods Share Common Patterns

All five saliency methods (`wanda`, `sparse_llm`, `taylor`, `grad`, `random`) follow the same interface:
- Accept a model + calibration data
- Return `dict[str, Tensor]` (saliency scores keyed by parameter name)
- `utils.py` provides shared helpers for collecting activations

But each implements this independently — no base class or shared protocol.

### Two Pruning Paths

1. **`weight_pruning.py`** — Generic: loads saliency from safetensors, computes masks, applies in-place
2. **`sae_enhanced/pruning.py`** — SAE-aware: uses SAE feature importance instead of weight saliency

These don't share code despite similar structure.

### Evaluation Duplication

- `scoping_eval.py` — Full LLM judge pipeline with multi-judge aggregation
- `spylab_1click_judgement.py` — Base judge framework it inherits from
- `grade_chats/generic_judges.py` — Another judge implementation
- `grade_model.py` — Yet another evaluation entry point

The relationship between these is: `scoping_eval` inherits from `spylab_1click_judgement`, `grade_model` and `grade_chats` are older alternatives. Only `scoping_eval` is used by the sweep scripts.

---

## Gotchas

### TRL Moves Models to Device Silently

`trl.SFTTrainer` and `trl.DPOTrainer` may move your model from CPU to the available accelerator (MPS on Mac, CUDA on Linux) during `.train()`. Post-training code that creates CPU tensors and passes them to the model will crash. This affects NPO and GradientDiff on Mac (MPS device mismatch). See TESTING_GUIDE.md for details.

### SparseLLM Memory

SparseLLM's `precompute_shared_data()` materializes full activation matrices (`X`, `Xinv`, etc.) per layer. For real models this is very memory-intensive. The `n_calibration` parameter directly controls memory usage — keep it small (4-8) for testing.

### Global vs Per-Row Thresholding

`sweep_sparsity.py:291` uses global thresholding by concatenating all saliency scores into a single CPU tensor. At 9B parameters this is ~36GB of CPU RAM. Per-row thresholding (WANDA default) avoids this.

### Cache Staleness

The sweep cache key (`sweep_sparsity.py:67`) doesn't include `n_calibration`, `max_seq_len`, or dataset identity. Changing these between runs reuses stale cached results silently.

### Wandb Auth Blocks Headless

`sweep_sparsity.py:400` calls `wandb.init()` unconditionally — blocks waiting for interactive auth in headless/CI environments. No `WANDB_MODE=disabled` fallback.

---

## Testing

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for the full test inventory, known bugs, and how to run everything.

Quick summary: 41 collectable CPU tests (37 pass, 4 RMU fail due to `NameError`), plus GPU integration tests requiring CUDA.
