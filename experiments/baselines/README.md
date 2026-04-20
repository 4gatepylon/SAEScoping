# Pruning Baselines

Weight pruning baselines for SAE Scoping. All methods prune weights to zero, producing sparse models that can be evaluated for in-domain utility and out-of-domain degradation.

## Code Structure

```
sae_scoping/training/saliency/         # Method implementations
├── wanda.py                           # Wanda: |W| * ||X||_2, per-row
├── sparse_llm.py                      # SparseLLM: iterative alternating optimization
├── taylor.py                          # Taylor: |grad * weight|
├── grad.py                            # EMA gradient collection (used by taylor/gradient)
├── random.py                          # Random uniform baseline
├── utils.py                           # Shared constants and helpers
└── tests/
    ├── test_wanda_cpu.py              # 12 CPU unit tests
    └── test_sparse_llm_cpu.py         # 10 CPU unit tests

sae_scoping/training/
├── weight_pruning.py                  # Shared: mask computation, weight save/restore
└── pgd_trainer.py                     # Shared: PGD SFT trainer for recovery

experiments/baselines/                 # Experiment scripts
├── sweep_sparsity.py                  # Unified sweep (all methods, all models)
├── launch_sweeps.py                   # Parallel multi-GPU launcher
├── sweep_scripts/                     # Per-model shell scripts
│   ├── sweep_gemma2_2b.sh
│   ├── sweep_gemma2_9b.sh
│   └── sweep_gemma3_12b.sh
├── run_wanda.py                       # Standalone Wanda experiment
├── run_sparse_llm.py                  # Standalone SparseLLM experiment
├── test_wanda.py                      # GPU integration tests (gemma-2-2b-it)
└── test_sparse_llm.py                 # GPU integration tests (gemma-2-2b-it)
```

## Methods

### Wanda (`wanda`)
**Criterion:** `S[i,j] = |W[i,j]| * ||X_j||_2`  
**Thresholding:** Per-row (each output neuron loses the same fraction of inputs)  
**Calibration:** Single forward pass on calibration data (no gradients)  
**Cacheable:** Yes — saliency map computed once, reused across sparsity levels

### SparseLLM (`sparse_llm`)
**Criterion:** Iterative alternating optimization linking FFN layers  
**Thresholding:** Per-row magnitude on optimized weights  
**Calibration:** Forward pass + pseudo-inverse computation (shared across sparsities)  
**Cacheable:** Shared data (hidden states, pseudo-inverses) computed once; per-sparsity masks cached to disk

### Taylor (`taylor`)
**Criterion:** `S[i,j] = |grad[i,j] * W[i,j]|`  
**Thresholding:** Global quantile  
**Prerequisite:** EMA gradient map (auto-computed and cached if missing)  
**Cacheable:** Yes — both gradient map and Taylor saliency cached

### Gradient (`gradient`)
**Criterion:** `S[i,j] = |grad[i,j]|`  
**Thresholding:** Global quantile  
**Prerequisite:** EMA gradient map (same as Taylor)  
**Cacheable:** Yes

### Random (`random`)
**Criterion:** `S[i,j] = Uniform(0, 1)`  
**Thresholding:** Global quantile  
**Cacheable:** Yes (deterministic with seed=42)

## Unified Sweep Interface

All methods use the same two-phase pipeline:

```python
# Phase 1: Compute saliency (cached, done once per method/model/dataset)
saliency_data = compute_saliency(method, model, tokenizer, calib_texts, ...)

# Phase 2: For each sparsity level, derive masks and apply
for sparsity in [0.0, 0.1, 0.2, ..., 0.9]:
    masks = masks_for_sparsity(method, saliency_data, model, sparsity, ...)
    n_zeroed = apply_masks_to_model(model, masks)
    # evaluate...
```

### CLI

```bash
# Single method sweep:
python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it

# Custom sparsity levels:
python sweep_sparsity.py --method taylor --model google/gemma-2-9b-it \
    --sparsity-levels 0.1,0.3,0.5,0.7

# Custom cache directory (e.g. network drive):
python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it \
    --cache-dir /network/drive/saliency_cache

# Disable caching (always recompute):
python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it --no-cache

# Skip LLM judge (loss only, faster):
python sweep_sparsity.py --method sparse_llm --model google/gemma-2-2b-it --no-judge

# Parallel across GPUs:
python launch_sweeps.py --gpus 0,2,3,7 --all
python launch_sweeps.py --gpus 0,2 --methods wanda,random --model google/gemma-2-9b-it
```

### Cache Structure

```
saliency_cache/
└── google--gemma-2-2b-it/
    └── biology/
        ├── wanda_saliency.safetensors
        ├── random_saliency.safetensors
        ├── ema_grads.safetensors           # shared by taylor and gradient
        ├── taylor_saliency.safetensors
        ├── gradient_saliency.safetensors
        ├── sparse_llm_masks_0p3000.safetensors
        ├── sparse_llm_masks_0p5000.safetensors
        └── ...
```

## Metrics Logged

Per sparsity level, logged to W&B:
- `train_loss` — cross-entropy on train split
- `test_loss` — cross-entropy on test split  
- `actual_sparsity` — measured fraction of zero weights
- `n_zeroed` — absolute count of zeroed weights
- `prune_time_s` — wall time for pruning
- `llm_judge/...` — LLM judge scores (relevance, fluency, ground_truth_similarity)

## Tests

### CPU Unit Tests (no GPU, ~17s total)

Run: `python -m pytest sae_scoping/training/saliency/tests/ -v`

**Wanda tests (12):**
| Test | What it verifies |
|------|-----------------|
| `test_returns_dict` | `compute_wanda_saliency` returns a non-empty dict |
| `test_keys_are_weight_params` | All keys end with `.weight` |
| `test_shapes_match_weights` | Score tensor shapes match model weight shapes |
| `test_scores_nonnegative` | All saliency scores >= 0 |
| `test_scores_on_cpu` | Scores returned on CPU (not GPU) |
| `test_mask_shapes` | Mask shapes match saliency shapes |
| `test_masks_are_bool` | Masks have dtype=torch.bool |
| `test_per_row_sparsity` | Each row has exactly `int(n_cols * sparsity)` zeros |
| `test_zero_sparsity_keeps_all` | Sparsity=0 produces all-True masks |
| `test_end_to_end` | `prune_wanda` zeros weights and increases zero count |
| `test_return_masks` | `return_masks=True` returns (int, dict) with bool CPU masks |
| `test_model_still_runs` | Model forward pass works after 50% pruning, no NaN |

**SparseLLM tests (10):**
| Test | What it verifies |
|------|-----------------|
| `test_returns_dict_of_masks` | Returns non-empty dict |
| `test_masks_are_binary` | All mask values are exactly 0.0 or 1.0 |
| `test_masks_have_correct_keys` | Keys end with `.weight` and exist in model |
| `test_masks_have_correct_shapes` | Mask shapes match model parameter shapes |
| `test_nonzero_sparsity` | 50% sparsity produces >0 zeros, actual sparsity in (0.1, 0.9) |
| `test_shared_data_reuse_across_sparsities` | Same shared data produces different masks at 30% vs 50%, with 50% having more zeros |
| `test_shared_data_has_correct_structure` | Shared data tensors on CPU, shapes consistent (X, Xinv, z, s, p, Y, attn_weights) |
| `test_end_to_end` | `prune_sparse_llm` zeros weights |
| `test_return_masks` | `return_masks=True` returns masks on CPU |
| `test_model_still_runs` | Forward pass works after pruning, no NaN |

### GPU Integration Tests (~90s each on gemma-2-2b-it)

Run: `CUDA_VISIBLE_DEVICES=0 python test_wanda.py` or `test_sparse_llm.py`

These use the real `google/gemma-2-2b-it` model with StemQA biology calibration data. They verify end-to-end correctness including actual loss changes and text generation quality after pruning.
