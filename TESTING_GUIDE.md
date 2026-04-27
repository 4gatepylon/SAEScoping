# Testing Guide

How to validate the components built in the last ~100 commits on the `adriano/baselines` branch.

---

## Quick Start

```bash
# Run all CPU tests (no GPU needed, ~10 minutes)
/opt/miniconda3/envs/saescoping/bin/python -m pytest \
    sae_scoping/hyperparameter_optimization/test_binary_search.py \
    sae_scoping/training/saliency/tests/test_wanda_cpu.py \
    sae_scoping/training/saliency/tests/test_sparse_llm_cpu.py \
    sae_scoping/training/unlearning/tests/test_unlearning_cpu.py \
    -v

# Expected: 37 passed, 4 failed (RMU — known bug, see below)
```

---

## Test Inventory

| File | Tests | GPU? | Runtime | Status |
|------|-------|------|---------|--------|
| `sae_scoping/hyperparameter_optimization/test_binary_search.py` | 8 | No | <1s | **All pass** |
| `sae_scoping/training/saliency/tests/test_wanda_cpu.py` | 12 | No | <30s | **All pass** |
| `sae_scoping/training/saliency/tests/test_sparse_llm_cpu.py` | 10 | No | <30s | **All pass** |
| `sae_scoping/training/unlearning/tests/test_unlearning_cpu.py` | 12 | No | ~9 min | **4 RMU fail** |
| `sae_scoping/training/sae_enhanced/hooks/test_pt_hooks.py` | 2 | No | <1s | **BROKEN** (import) |
| `sae_scoping/models/test_sae_enhanced_gemma2.py` | 0 | — | — | **Empty stub** |
| `experiments/baselines/test_wanda.py` | 6 | Yes | ~5 min | GPU integration |
| `experiments/baselines/test_sparse_llm.py` | 5 | Yes | ~5 min | GPU integration |
| `experiments/baselines/test_unlearning.py` | 9 | Yes | ~30 min | GPU integration |

**Total: 41 collectable tests + 1 broken import + 1 empty stub**

---

## Known Bugs in Tests

### BUG: RMU tests crash with `NameError` (4 tests)

**File:** `sae_scoping/training/unlearning/rmu.py` lines 154-197

The `unlearn_rmu()` function calls 4 undefined names:
- `_get_hidden_size(model)` → should be `get_hidden_size(model)` (defined at line 56)
- `_get_num_layers(model)` → should be `get_num_layers(model)` (defined at line 48)
- `_get_layer_module(model, ...)` → should be `get_layer_module(model, ...)` (defined at line 39)

All 4 RMU CPU tests fail immediately with `NameError: name '_get_hidden_size' is not defined`.

**Failing tests:**
- `TestRMU::test_only_update_layers_change`
- `TestRMU::test_forget_loss_increases`
- `TestRMU::test_model_still_runs`
- `TestRMU::test_all_params_unfrozen_after`

### BUG: test_pt_hooks.py has stale imports (2 tests blocked)

**File:** `sae_scoping/training/sae_enhanced/hooks/test_pt_hooks.py` lines 9-21

After the package reorganization (`trainers/` → `training/`, `utils/hooks/` → `training/sae_enhanced/hooks/`), imports were not updated. Pytest collection fails with `ModuleNotFoundError: No module named 'utils'`.

**Fix:** Change line 19 to `from sae_scoping.training.sae_enhanced.hooks.pt_hooks import named_forward_hooks`

### BUG: MPS device mismatch on Mac (NPO, GradientDiff)

If running on a Mac with MPS, TRL silently moves the model from CPU to `mps:0` during `.train()`. Post-training code creates CPU tensors and passes them to the MPS-resident model, causing:
```
RuntimeError: Placeholder storage has not been allocated on MPS device!
```

This affects `unlearn_npo()` and `unlearn_gradient_diff()`. Only triggers on Mac with MPS — on Linux (CUDA or CPU-only), tests pass fine.

### STUB: test_sae_enhanced_gemma2.py is empty

The file contains only TODOs. No tests for the SAE-enhanced Gemma2 wrapper model exist yet.

---

## Detailed Test Descriptions

### 1. Binary Search (`test_binary_search.py`)

**What it validates:** The `binary_search_step()` function for hyperparameter tuning — finding the maximum feasible value in a range where a train-and-eval callback determines feasibility.

**How to run:**
```bash
/opt/miniconda3/envs/saescoping/bin/python -m pytest \
    sae_scoping/hyperparameter_optimization/test_binary_search.py -v
```

**Tests:**
- `test_finds_max_feasible_float` — converges to correct threshold (linear callback)
- `test_finds_max_feasible_int` — works for integer hyperparameters
- `test_strictly_narrows` — bounds strictly narrow each call
- `test_error_on_invalid_range` — raises `InvalidRangeError` when lo >= hi
- `test_error_on_zero_steps` — raises `InvalidRangeError` when max_steps <= 0
- `test_eval_too_low_narrows_hi` — when eval is below range, upper bound shrinks
- `test_eval_too_high_narrows_lo` — when eval is above range, lower bound grows
- `test_int_no_progress_raises` — raises `NoProgressError` when int range can't narrow

**Commits validated:** `8e55e4a Add binary search hyperparameter optimization module`

---

### 2. WANDA CPU Tests (`test_wanda_cpu.py`)

**What it validates:** The WANDA pruning criterion (`|W| * ||X||_2`) on a tiny Gemma-2 model (2 layers, hidden=64).

**How to run:**
```bash
/opt/miniconda3/envs/saescoping/bin/python -m pytest \
    sae_scoping/training/saliency/tests/test_wanda_cpu.py -v
```

**Model:** Tiny Gemma-2 from `google/gemma-2-2b-it` config with `num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=4`. Real vocab size (256k) for valid tokenizer IDs.

**Calibration data:** 4 synthetic chat-templated texts, max_seq_len=64.

**Test classes:**

`TestComputeWandaSaliency` (5 tests):
- Output is non-empty dict with keys ending in `.weight`
- Saliency tensor shapes match weight shapes
- All scores non-negative, on CPU

`TestComputeWandaMasks` (4 tests):
- Masks are bool, correct shapes
- Per-row sparsity exactly matches target (40%)
- Zero sparsity keeps everything

`TestPruneWanda` (3 tests):
- Zeros increase after pruning (deepcopy to preserve fixture)
- `return_masks=True` returns `(n_zeroed, masks)` tuple
- Model forward pass works post-pruning (no NaN)

**Commits validated:** `8acf6cc implement Wanda pruning`, `b80eece add CPU unit tests for Wanda and SparseLLM`

---

### 3. SparseLLM CPU Tests (`test_sparse_llm_cpu.py`)

**What it validates:** SparseLLM iterative alternating optimization on tiny Gemma-2.

**How to run:**
```bash
/opt/miniconda3/envs/saescoping/bin/python -m pytest \
    sae_scoping/training/saliency/tests/test_sparse_llm_cpu.py -v
```

**Model & data:** Same tiny Gemma-2 and synthetic calibration texts as WANDA.

**Test classes:**

`TestComputeSparseLLMMasks` (5 tests):
- Masks are binary (0.0/1.0), correct shapes, keys match weight params
- 50% sparsity produces zeros (actual sparsity varies, checked 10%-90% range)

`TestSharedDataPrecomputation` (2 tests):
- `precompute_shared_data()` runs once, reused for sparsity=0.3 and 0.5
- Higher sparsity produces more zeros
- Validates internal structure: X, Xinv, z_init, s_init, p_init tensors

`TestPruneSparseLLM` (3 tests):
- End-to-end pruning increases zeros (1 iteration, 30% sparsity)
- `return_masks=True` works, masks on CPU
- Forward pass post-pruning (no NaN)

**Commits validated:** `f51b03f implement SparseLLM pruning`, `555802b refactor SparseLLM to separate shared precomputation`

---

### 4. Unlearning CPU Tests (`test_unlearning_cpu.py`)

**What it validates:** Three unlearning methods (GradientDiff, NPO, RMU) on tiny Gemma-2.

**How to run:**
```bash
/opt/miniconda3/envs/saescoping/bin/python -m pytest \
    sae_scoping/training/unlearning/tests/test_unlearning_cpu.py -v
# Expected: 8 pass, 4 fail (RMU bug)
```

**Datasets:** 8 samples each, "math" (forget) and "biology" (retain), tokenized with chat template.

**TestGradientDiff** (3 tests — all pass):
- Parameters change after 5 steps
- Forget loss increases after 20 steps (lr=1e-3)
- Forward pass works post-unlearning

**TestNPO** (4 tests — all pass):
- Parameters change, forget loss increases
- Works without retain dataset (pure NPO, no KL penalty)
- Forward pass works

**TestRMU** (4 tests — **all fail**, see known bug above):
- Intended to validate: only update layers change, forget loss increases, forward pass works, all params unfrozen after

**Commits validated:** `42f6bb2 implement three unlearning baselines`, `ac0a984 fix unlearning trainers for CPU and TRL compat`, `add4244 rewrite RMU to match official WMDP implementation`

---

### 5. GPU Integration Tests

These require a CUDA GPU and real model downloads (~5GB for gemma-2-2b-it).

#### WANDA GPU (`experiments/baselines/test_wanda.py`)

```bash
python experiments/baselines/test_wanda.py --device cuda:0
```

Tests on real `google/gemma-2-2b-it` (bfloat16):
- Saliency computation shapes and non-negativity
- Mask validity at 30% sparsity (per-row sparsity within 5% of target)
- End-to-end pruning at 50% (zeros increase)
- Loss is finite post-pruning (allows 10% margin below baseline)
- Generation works ("What is photosynthesis?" → 30 tokens)
- `return_masks` for PGD compatibility

#### SparseLLM GPU (`experiments/baselines/test_sparse_llm.py`)

```bash
python experiments/baselines/test_sparse_llm.py --device cuda:0
```

Similar to WANDA GPU but with `n_calibration=4` (SparseLLM needs more memory) and `n_iterations=2`.

#### Unlearning GPU (`experiments/baselines/test_unlearning.py`)

```bash
python experiments/baselines/test_unlearning.py --device cuda:0
```

Tests all 3 methods (GradientDiff, NPO, RMU) on real model:
- 100 train samples per domain, 30 eval
- 50 training steps per method
- Validates forget loss increases by ≥5%, retain loss doesn't more than double
- Generation check post-unlearning

**Note:** RMU GPU test will also fail due to the same `NameError` bug.

---

## Validating Specific Commits

### Package Reorganization (`f0b1c21 refactor: reorganize package structure`)

The package was reorganized from `trainers/` → `training/`, `utils/hooks/` → `training/sae_enhanced/hooks/`. Validate imports work:

```bash
python -c "from sae_scoping.training.saliency.wanda import compute_wanda_saliency; print('ok')"
python -c "from sae_scoping.training.saliency.sparse_llm import compute_sparse_llm_masks; print('ok')"
python -c "from sae_scoping.training.unlearning.rmu import unlearn_rmu; print('ok')"
python -c "from sae_scoping.training.unlearning.npo import unlearn_npo; print('ok')"
python -c "from sae_scoping.training.weight_pruning import prune_model; print('ok')"
python -c "from sae_scoping.training.pgd_trainer import PGDSFTTrainer; print('ok')"
python -c "from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval; print('ok')"
```

### Sweep Infrastructure (`e439610`, `57fc720`, `1c1a6a5`)

The unified sweep script, caching system, and parallel launcher. No automated tests exist — validate manually:

```bash
# Dry-run: list what would run (no --no-judge to skip expensive LLM judge)
python experiments/baselines/sweep_sparsity.py --help
python experiments/baselines/launch_sweeps.py --help
```

### Evaluation System (`scoping_eval.py`)

No dedicated tests. The evaluation system is validated indirectly through the sweep scripts and GPU integration tests. See [TODO(claude) annotations](#todoclaude-annotations) for known bugs.

---

## TODO(claude) Annotations

The previous agent annotated 28 bugs across the codebase. Here is the full inventory by priority:

### HIGH Priority (6)

| File | Line | Issue |
|------|------|-------|
| `experiments/baselines/sweep_sparsity.py` | 67 | Cache filename ignores `n_calibration`, `max_seq_len`, dataset — stale cache reuse |
| `experiments/baselines/sweep_sparsity.py` | 140 | Loss is per-batch mean, not token-weighted — metric shifts with batch size |
| `experiments/baselines/sweep_sparsity.py` | 164 | Empty texts return 0.0 silently (looks like valid loss) |
| `experiments/baselines/sweep_sparsity.py` | 202 | Seed hardcoded to 42, not in cache key — future seeds collide |
| `experiments/baselines/sweep_sparsity.py` | 291 | Global thresholding concatenates all scores into single CPU tensor (~36GB at 9B) |
| `experiments/baselines/sweep_sparsity.py` | 377 | Judge pool not held out from calibration/train data |
| `experiments/baselines/launch_sweeps.py` | 96 | Unknown models silently get 2B tuning — will OOM on larger models |

### MEDIUM Priority (12)

| File | Line | Issue |
|------|------|-------|
| `sweep_sparsity.py` | 107 | No `revision=` pin on HF dataset — upstream changes silently affect results |
| `sweep_sparsity.py` | 345 | No `torch.manual_seed` — future RNG-using saliency will drift |
| `sweep_sparsity.py` | 400 | `wandb.init` unconditional — blocks on interactive auth in headless runs |
| `sweep_sparsity.py` | 432 | Sparsity denominator counts all mask params, not just 2D weights |
| `sweep_sparsity.py` | 450 | Judge evaluator re-instantiated per sparsity level (template reload, cost guard reset) |
| `sweep_sparsity.py` | 454 | Judge exceptions silently swallowed — sweep "succeeds" with missing metrics |
| `sweep_sparsity.py` | 472 | `empty_cache` only after both eval+judge, not between them |
| `launch_sweeps.py` | 126 | Subprocess output swallowed — crash root cause lost |
| `launch_sweeps.py` | 230 | Round-robin GPU assignment — skewed runtimes leave GPUs idle |
| `scoping_eval.py` | 208 | Hard-assert on tokenizer idempotence — crashes judge for entire level |
| `scoping_eval.py` | 319 | Assumes judge scores are integers in {0, 1, 2} — fragile if template changes |
| `scoping_eval.py` | 432 | Double-sampling (upstream shuffle + inner re-sample) — redundant |
| `scoping_eval.py` | 478 | Set deduplication is non-deterministic (hash-seeded) — batch composition varies |
| `weight_pruning.py` | 90 | `save_original_weights` clones ALL params (~18GB at 9B) — could optimize |

### LOW Priority (7)

| File | Line | Issue |
|------|------|-------|
| `sweep_sparsity.py` | 113 | Assert gives no actionable hint on failure |
| `sweep_sparsity.py` | 404 | wandb config omits critical parameters |
| `sweep_sparsity.py` | 445 | No subset/method tag in print output |
| `sweep_sparsity.py` | 485 | No gc.collect between methods |
| `launch_sweeps.py` | 152 | `--saliency-path` flag has dual meaning |
| `wanda.py` | 196 | Non-2D parameter branch is unreachable dead code |
| `scoping_eval.py` | 110 | Docstring references "cybersecurity" but code uses "physics" |
