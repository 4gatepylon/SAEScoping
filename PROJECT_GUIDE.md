# Project Guide

Architecture, modules, branches, and gotchas for the SAEScoping repository.

---

## What This Project Does

SAE Scoping constrains an LLM to a specific knowledge domain (e.g. biology) using Sparse Autoencoders. The pipeline has three active stages:

1. **RANK** — Score every weight by importance (saliency methods: WANDA, Taylor, Gradient, Random)
2. **PRUNE** — Zero out unimportant weights (global or per-row thresholding)
3. **RECOVER** — Fine-tune surviving weights to restore in-domain performance (PGD SFT keeps pruned weights at zero)

Evaluation uses an LLM-judge pipeline (`scoping_eval.py`) measuring relevance, fluency, and ground-truth similarity.

---

## Package Layout (`sae_scoping/`)

```
sae_scoping/
├── datasets/
│   └── qa_datasets.py          # Dataset loading, validation, formatting, non-overlapping splits
│
├── evaluation/
│   ├── loss.py                 # Shared batched cross-entropy loss + count_zeros
│   ├── scoping_eval.py         # LLM-judge evaluator: relevance, fluency, ground_truth_similarity
│   ├── spylab_1click_judgement.py  # Base judge framework (from spylab)
│   ├── grade_chats/            # Chat grading utilities
│   └── inference/
│       └── client/
│           ├── api_generator.py          # LiteLLM batch API calls (used by judge pipeline)
│           └── length_aware_tokenizer.py # Tokenizer wrapper for length-aware batching
│
├── examples/
│   └── test_wanda_gpu.py       # GPU integration tests for WANDA (10 tests, requires CUDA)
│
├── training/
│   ├── weight_pruning.py       # Apply saliency masks to zero out weights (in-place)
│   ├── pgd_trainer.py          # SFTTrainer subclass: re-zeros pruned weights after each step
│   ├── saliency/
│   │   ├── dispatch.py         # Unified saliency computation + masking across all methods
│   │   ├── wanda.py            # WANDA: |W| * ||X||_2 (one-shot, calibration-based)
│   │   ├── taylor.py           # Taylor: |grad * weight|
│   │   ├── grad.py             # Gradient: EMA |grad|
│   │   ├── random.py           # Random baseline
│   │   ├── utils.py            # Shared saliency constants and helpers
│   │   └── tests/
│   │       └── test_wanda_cpu.py  # CPU unit tests for WANDA (12 tests, all pass)
│   └── utils/
│       └── hooks/
│           ├── pt_hooks.py     # PyTorch forward hooks for SAE intervention
│           └── test_pt_hooks.py  # Hook unit tests (2 tests, all pass)
│
└── utils/
    └── cache.py                # Generic safetensors cache-or-compute helper
```

---

## Experiment Scripts (`experiments/`)

```
experiments/
├── baselines/
│   ├── sweep_sparsity.py       # Thin CLI: sweep sparsity levels, log to wandb
│   ├── launch_sweeps.py        # Parallel launcher: fan out sweeps across GPUs
│   └── sweep_scripts/          # Shell scripts for specific model sweeps
└── README.md
```

---

## Branches

### Active

| Branch | Purpose | Base |
|--------|---------|------|
| `adriano/baselines` | **Current work.** Pruning baselines (WANDA focus), sweep infrastructure, CPU/GPU tests | Latest on main |
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

Post-refactor branches (`baselines`, `sae_pruning`, `saliency-pruning_cursor_sweep`) use the current layout. Note: the `adriano/baselines` branch has further simplified the layout by removing `sae_enhanced/`, `unlearning/`, `sparse_llm`, `hyperparameter_optimization/`, `models/`, and several other modules (see recent cleanup commits).

---

## Key Dependencies

| Package | Version | Role |
|---------|---------|------|
| `torch` | 2.7.1 | Core tensor operations |
| `transformers` | 4.56.1 | Model loading, tokenizers |
| `trl` | 0.22.2 | SFTTrainer (base for PGD trainer) |
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
calibration data → dispatch.compute_saliency(method, ...)   [cached via utils.cache]
                       ↓
              saliency scores (dict[str, Tensor])
                       ↓
              dispatch.masks_for_sparsity(method, ...)       [per-row or global threshold]
                       ↓
              wanda.apply_masks_to_model() → model with zeros
                       ↓
              PGDSFTTrainer(masks=...) → recovery SFT (zeros stay zero)
                       ↓
              scoping_eval.OneClickLLMJudgeScopingEval → judge evaluation
```

### Sweep Infrastructure

```
launch_sweeps.py → spawns N subprocesses (one per GPU)
    ↓
sweep_sparsity.py (thin CLI) → iterates sparsity levels [0.1, 0.2, ..., 0.9]
    ↓ for each level:
    ├── dispatch.compute_saliency + dispatch.masks_for_sparsity
    ├── evaluation.loss.compute_loss on train/test sets
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
| `test_wanda_cpu.py` | `num_attention_heads=4, intermediate_size=128` | Tiny model config for CPU tests |
| `scoping_eval.py` | `{0, 1, 2}` judge scores | Hard-assert on integer scores — fragile if template changes |
| `weight_pruning.py:90` | `save_original_weights` | Clones ALL parameters (~18GB at 9B) |

---

## Overlapping Functionality

### Saliency Methods Share Common Patterns

The four saliency methods (`wanda`, `taylor`, `grad`, `random`) follow the same interface:
- Accept a model + calibration data
- Return `dict[str, Tensor]` (saliency scores keyed by parameter name)
- `utils.py` provides shared helpers for collecting activations

But each implements this independently — no base class or shared protocol.

### Evaluation Duplication

- `scoping_eval.py` — Full LLM judge pipeline with multi-judge aggregation
- `spylab_1click_judgement.py` — Base judge framework it inherits from
- `grade_chats/generic_judges.py` — Another judge implementation
- `grade_model.py` — Yet another evaluation entry point

The relationship between these is: `scoping_eval` inherits from `spylab_1click_judgement`, `grade_model` and `grade_chats` are older alternatives. Only `scoping_eval` is used by the sweep scripts.

---

## Gotchas

### TRL Moves Models to Device Silently

`trl.SFTTrainer` may move your model from CPU to the available accelerator (MPS on Mac, CUDA on Linux) during `.train()`. Post-training code that creates CPU tensors and passes them to the model will crash.

### Global vs Per-Row Thresholding

`sweep_sparsity.py:291` uses global thresholding by concatenating all saliency scores into a single CPU tensor. At 9B parameters this is ~36GB of CPU RAM. Per-row thresholding (WANDA default) avoids this.

### Cache Staleness

The sweep cache key (`sweep_sparsity.py:67`) doesn't include `n_calibration`, `max_seq_len`, or dataset identity. Changing these between runs reuses stale cached results silently.

### Wandb Auth Blocks Headless

`sweep_sparsity.py:400` calls `wandb.init()` unconditionally — blocks waiting for interactive auth in headless/CI environments. No `WANDB_MODE=disabled` fallback.

---

## Deleted Modules (historical reference)

The following were removed from `adriano/baselines` in cleanup commits (April 2026) and still exist on other branches:

- `sae_scoping/hyperparameter_optimization/` — Binary search for hyperparameter tuning
- `sae_scoping/training/saliency/sparse_llm.py` — SparseLLM iterative alternating optimization
- `sae_scoping/training/unlearning/` — GradientDiff, NPO, RMU unlearning baselines
- `sae_scoping/training/sae_enhanced/` — SAE-aware pruning, firing rates, SFT, hooks
- `sae_scoping/models/` — SAE-enhanced Gemma2 wrapper
- `sae_scoping/datasets/text_datasets.py`, `messages_datasets.py` — Older dataset loaders
- `sae_scoping/evaluation/inference/` — API/model generator clients for judge inference
- `experiments/baselines/run_sparse_llm.py`, `test_sparse_llm.py`, `test_unlearning.py` — GPU integration tests for deleted modules
- Various older experiment scripts (`script_*.py`, `plot_*.py`, notebooks, slurm scripts)

---

## Testing

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for the full test inventory, known bugs, and how to run everything.

Quick summary: 12 WANDA CPU tests (all pass), 10 GPU integration tests in `sae_scoping/examples/` (require CUDA), 2 pt_hooks tests (broken import).
