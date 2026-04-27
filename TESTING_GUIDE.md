# Testing Guide

How to validate the components on the `adriano/baselines` branch.

---

## Quick Start

```bash
# Run all CPU tests (no GPU needed, ~3 min)
/opt/miniconda3/envs/saescoping/bin/python -m pytest \
    sae_scoping/training/saliency/tests/test_wanda_cpu.py \
    -v

# Expected: 12 passed

# Run GPU integration tests (requires CUDA, ~5 min)
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/saescoping/bin/python -m pytest \
    sae_scoping/examples/test_wanda_gpu.py -v
```

---

## Test Inventory

| File | Tests | GPU? | Runtime | Status |
|------|-------|------|---------|--------|
| `sae_scoping/training/saliency/tests/test_wanda_cpu.py` | 12 | No | ~3 min | **All pass** |
| `sae_scoping/examples/test_wanda_gpu.py` | 10 | Yes | ~5 min | GPU integration |
| `sae_scoping/training/utils/hooks/test_pt_hooks.py` | 2 | No | <1s | **All pass** |

**Total: 24 collectable tests (14 CPU pass, 10 GPU skip without CUDA)**

### Deleted Tests (historical reference)

The following test files were removed in recent cleanup commits and exist on other branches:

- `sae_scoping/hyperparameter_optimization/test_binary_search.py` (8 tests) — deleted with module
- `sae_scoping/training/saliency/tests/test_sparse_llm_cpu.py` (10 tests) — deleted with sparse_llm.py
- `sae_scoping/training/unlearning/tests/test_unlearning_cpu.py` (12 tests, 4 RMU failing) — deleted with unlearning/
- `experiments/baselines/test_sparse_llm.py` (5 GPU tests) — deleted with module
- `experiments/baselines/test_unlearning.py` (9 GPU tests) — deleted with module

---

## Detailed Test Descriptions

### 1. WANDA CPU Tests (`test_wanda_cpu.py`)

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

---

### 2. GPU Integration Tests (`sae_scoping/examples/test_wanda_gpu.py`)

Requires CUDA and downloads ~5GB (gemma-2-2b-it).

```bash
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/saescoping/bin/python -m pytest \
    sae_scoping/examples/test_wanda_gpu.py -v
```

Uses library functions (`qa_datasets`, `evaluation.loss`, `saliency.wanda`) — no local duplication.

`TestSaliencyShapes` (3 tests):
- Non-empty saliency map
- Shapes match model weights
- All scores non-negative

`TestMasks` (3 tests):
- Per-row sparsity within 5% of target at 30%
- Zero sparsity keeps all weights
- 90% sparsity removes most weights

`TestPruneEndToEnd` (4 tests):
- Zeros increase after 50% pruning
- Loss is finite post-pruning
- Generation works ("What is photosynthesis?" -> 30 tokens)
- `return_masks` returns CPU bool masks for PGD

---

## Validating Imports

```bash
python -c "from sae_scoping.datasets.qa_datasets import load_qa_dataset; print('ok')"
python -c "from sae_scoping.evaluation.loss import compute_loss; print('ok')"
python -c "from sae_scoping.training.saliency.wanda import compute_wanda_saliency; print('ok')"
python -c "from sae_scoping.training.weight_pruning import prune_model; print('ok')"
python -c "from sae_scoping.training.pgd_trainer import PGDSFTTrainer; print('ok')"
```

---

## Sweep Infrastructure

The unified sweep script, caching system, and parallel launcher. No automated tests exist — validate manually:

```bash
python experiments/baselines/sweep_sparsity.py --help
python experiments/baselines/launch_sweeps.py --help
```

---

## TODO(claude) Annotations

Remaining annotations in the codebase (many were resolved during the refactor that moved logic into library code):

### HIGH Priority (1)

| File | Line | Issue |
|------|------|-------|
| `launch_sweeps.py` | 90 | Unknown models silently get 2B tuning — will OOM on larger models |

### MEDIUM Priority (5)

| File | Line | Issue |
|------|------|-------|
| `launch_sweeps.py` | 114 | Subprocess output swallowed — crash root cause lost |
| `launch_sweeps.py` | 217 | Round-robin GPU assignment — skewed runtimes leave GPUs idle |
| `scoping_eval.py` | 208 | Hard-assert on tokenizer idempotence — crashes judge for entire level |
| `scoping_eval.py` | 319 | Assumes judge scores are integers in {0, 1, 2} — fragile if template changes |
| `scoping_eval.py` | 432 | Double-sampling (upstream shuffle + inner re-sample) — redundant |
| `scoping_eval.py` | 478 | Set deduplication is non-deterministic (hash-seeded) — batch composition varies |
| `weight_pruning.py` | 90 | `save_original_weights` clones ALL params (~18GB at 9B) — could optimize |

### LOW Priority (2)

| File | Line | Issue |
|------|------|-------|
| `launch_sweeps.py` | 139 | `--saliency-path` flag has dual meaning |
| `scoping_eval.py` | 110 | Docstring references "cybersecurity" but code uses "physics" |
