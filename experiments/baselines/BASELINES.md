# Scoping Baselines

Methods for narrowing general-purpose LLMs to single-purpose models. Two families: **weight pruning** (zero out unimportant weights) and **unlearning** (train the model to forget unwanted capabilities).

## Code Structure

```
sae_scoping/training/
├── saliency/                          # Weight pruning methods
│   ├── wanda.py                       # |W| * ||X||_2, per-row
│   ├── sparse_llm.py                  # Iterative alternating optimization
│   ├── taylor.py                      # |grad * weight|
│   ├── grad.py                        # EMA gradient collection
│   ├── random.py                      # Random baseline
│   ├── utils.py                       # Shared constants
│   └── tests/
│       ├── test_wanda_cpu.py          # 12 tests
│       └── test_sparse_llm_cpu.py     # 10 tests
├── unlearning/                        # Unlearning methods
│   ├── gradient_diff.py               # -CE(forget) + CE(retain)
│   ├── npo.py                         # Negative Preference Optimization
│   ├── rmu.py                         # Representation Misdirection (WMDP)
│   └── tests/
│       └── test_unlearning_cpu.py     # 11 tests
├── weight_pruning.py                  # Shared mask infrastructure
└── pgd_trainer.py                     # PGD SFT for recovery

experiments/baselines/
├── sweep_sparsity.py                  # Unified pruning sweep (all methods)
├── launch_sweeps.py                   # Multi-GPU parallel launcher
├── test_wanda.py                      # GPU integration (pruning)
├── test_sparse_llm.py                 # GPU integration (pruning)
├── test_unlearning.py                 # GPU integration (unlearning)
└── sweep_scripts/                     # Per-model shell scripts
```

## Weight Pruning Methods

All methods produce a saliency map (`dict[str, Tensor]`) that can be cached and reused across sparsity levels (except SparseLLM which caches per-sparsity masks).

| Method | Criterion | Thresholding | Cacheable | Reference |
|--------|-----------|-------------|-----------|-----------|
| `wanda` | `\|W[i,j]\| * \|\|X_j\|\|_2` | Per-row | Saliency map | Sun et al., 2023 |
| `taylor` | `\|grad * weight\|` | Global | Saliency map | Molchanov et al., 2019 |
| `gradient` | `\|grad\|` | Global | Saliency map | — |
| `random` | `Uniform(0,1)` | Global | Saliency map | — |
| `sparse_llm` | Iterative alternating opt | Per-row | Per-sparsity masks | Bai et al., 2024 |

### API

```python
# All pruning uses the same two-phase interface:
from experiments.baselines.sweep_sparsity import compute_saliency, masks_for_sparsity
from sae_scoping.training.saliency.wanda import apply_masks_to_model

# Phase 1: compute saliency (cached to disk)
saliency = compute_saliency("wanda", model, tokenizer, calib_texts, ...)

# Phase 2: mask at desired sparsity
masks = masks_for_sparsity("wanda", saliency, model, sparsity=0.5, ...)
n_zeroed = apply_masks_to_model(model, masks)
```

Or use each method directly:

```python
from sae_scoping.training.saliency.wanda import prune_wanda
n_zeroed = prune_wanda(model, tokenizer, calib_texts, sparsity=0.5)

from sae_scoping.training.saliency.sparse_llm import prune_sparse_llm
n_zeroed = prune_sparse_llm(model, tokenizer, calib_texts, sparsity=0.5)
```

### Sweep CLI

```bash
# Single method:
python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it

# All methods, parallel across GPUs:
python launch_sweeps.py --gpus 0,2,3,7 --all

# Custom cache dir and no caching:
python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it \
    --cache-dir /network/drive/cache --no-cache
```

## Unlearning Methods

All methods modify the model in-place using the same interface pattern.

| Method | Loss | Retain mechanism | Reference |
|--------|------|-----------------|-----------|
| `gradient_diff` | `-alpha*CE(forget) + beta*CE(retain)` | Gradient descent on retain set | Baseline in all unlearning papers |
| `npo` | `-(2/beta)*log(sigmoid(-beta*log_ratio))` | Optional KL on retain set | Zhang et al., NeurIPS 2024 |
| `rmu` | `MSE(h(forget), random_vec) + alpha*MSE(h(retain), h_orig)` | Activation matching at target layer | Li et al., 2024 (WMDP) |

### API

```python
from sae_scoping.training.unlearning.gradient_diff import unlearn_gradient_diff
from sae_scoping.training.unlearning.npo import unlearn_npo
from sae_scoping.training.unlearning.rmu import unlearn_rmu

# All share: model, tokenizer, forget_dataset, retain_dataset
# Datasets must have a "text" column (chat-template formatted)

unlearn_gradient_diff(model, tokenizer, forget_ds, retain_ds,
    forget_weight=1.0, retain_weight=5.0, max_steps=200)

unlearn_npo(model, tokenizer, forget_ds, retain_dataset=retain_ds,
    npo_beta=0.1, retain_weight=5.0, max_steps=200)

unlearn_rmu(model, tokenizer, forget_ds, retain_ds,
    hook_layer_id=7, steering_coeff=20.0, alpha=100.0, max_steps=80)
```

### RMU Implementation Notes

Our RMU follows the official WMDP code (`github.com/centerforaisafety/wmdp`):
- Control vector: `torch.rand` (uniform [0,1]), not Gaussian
- Full-sequence activations (not mean-pooled)
- `hook_layer_id` (where loss is computed) can differ from `update_layer_ids` (where gradients flow)
- Default `alpha=100` (retain loss weighted 100x)
- `param_ids` selects which parameters per layer to update (default: all)

## Caching

All intermediate artifacts are cached under `--cache-dir` (default: `./saliency_cache`).

```
saliency_cache/{model_slug}/{subset}/
├── wanda_saliency.safetensors         # Wanda scores
├── random_saliency.safetensors        # Random scores
├── ema_grads.safetensors              # EMA gradients (shared by taylor/gradient)
├── taylor_saliency.safetensors        # |grad * weight|
├── gradient_saliency.safetensors      # |grad|
├── sparse_llm_masks_0p3000.safetensors  # SparseLLM masks at 30%
└── sparse_llm_masks_0p5000.safetensors  # SparseLLM masks at 50%
```

Flags: `--cache-dir PATH` (custom location), `--no-cache` (always recompute).

## Tests

### CPU Unit Tests (no GPU, ~30s)

```bash
python -m pytest sae_scoping/training/saliency/tests/ sae_scoping/training/unlearning/tests/ -v
```

**33 tests total:**

| Suite | Tests | What's verified |
|-------|-------|----------------|
| Wanda (12) | Saliency shapes, non-negativity, CPU placement, per-row sparsity correctness, zero-sparsity identity, end-to-end pruning, mask return, forward pass after pruning |
| SparseLLM (10) | Mask binarity, shapes, keys, sparsity, shared data reuse across sparsities, shared data structure, end-to-end, mask return, forward pass |
| GD (3) | Params modified, forget loss increases, forward pass OK |
| NPO (4) | Params modified, forget loss increases, works without retain set, forward pass OK |
| RMU (4) | Only update layers change, forget loss increases, forward pass OK, all params unfrozen after |

### GPU Integration Tests (~3 min on gemma-2-2b-it)

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/baselines/test_unlearning.py
```

**9 tests** verifying actual domain-specific behavior:
- Forget-domain (math) loss increases by >5%
- Retain-domain (biology) loss doesn't more than double
- Model generates coherent text after unlearning

## Models

All baselines support:
- `google/gemma-2-2b-it` — ~5GB VRAM, fast iteration
- `google/gemma-2-9b-it` — ~18GB VRAM
- `google/gemma-3-12b-it` — ~24GB VRAM (bf16)

All must be instruction-tuned (IT) variants with chat templates. Datasets must be StemQA-format (columns: `question`, `answer`).
