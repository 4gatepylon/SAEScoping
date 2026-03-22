# Tests: `sweep_eval_temp.py`

This document describes the tests for the pruning pipeline in `sweep_eval_temp.py`,
covering both the existing unit tests and the end-to-end pipeline tests.

---

## Existing tests (`test_sweep_eval_temp.py`)

Cover individual functions in isolation with tiny hand-crafted models.

| What is tested | Test function(s) |
|---|---|
| `build_sparsity_levels` | `test_build_sparsity_levels_*` |
| `compute_saliency_scores` (gradient + taylor) | `test_compute_saliency_scores_*` |
| Weight save / restore round-trip | `test_save_restore_weights_*` |
| `sample_pruning_probes` | `test_sample_pruning_probes_*` |
| `assert_kept_weights_unchanged` | `test_assert_kept_weights_unchanged_*` |
| `assert_pruned_weights_are_zero` | `test_assert_pruned_weights_are_zero_*` |
| `assert_zero_count_geq_target` | `test_assert_zero_count_geq_target_*` |
| `apply_pruning` end-to-end | `test_apply_pruning_*` |
| `save_generations` | `test_save_generations_*` |
| `_is_run_complete` with/without sentinel | `test_is_run_complete_*` |
| `_build_sweep_cmd` flag propagation | `test_build_sweep_cmd_*` |

---

## End-to-end pipeline tests (`test_sweep_pipeline_e2e.py`)

Use tiny models (from config or custom `nn.Module`) on CPU with analytically
known saliency values to verify *correctness* — not just structural properties.

### Key design: analytically known fixtures

**`make_known_saliency_fixture()`** — two-group model where pruning set is 100%
predictable:
- `important.weight` (16 params) → saliency = 100.0 → never pruned
- `unimportant.weight` (8 params) → saliency = 0.001 → always pruned first

At sparsity = 8/24 ≈ 0.333, *exactly* `unimportant.weight` must be zeroed.

**`make_taylor_vs_gradient_fixture()`** — 1×2 weight matrix where:
- Gradient scoring prunes `w[0]` (score 0.5)
- Taylor scoring prunes `w[1]` (score 0.2)
- Both are locally optimal; they just prune *different* weights

### The global optimality validator

`assert_pruning_is_lowest_saliency(model, saliency_scores)` is the core
correctness check: after pruning, `max(saliency of zero weights)` ≤
`min(saliency of non-zero weights)`.  This verifies the pruning algorithm
selects the globally lowest-saliency weights — a property that can fail
if per-tensor thresholding is used incorrectly.

### Tests

| Test | Model | What it verifies |
|---|---|---|
| `test_pruning_selects_globally_lowest_saliency` | `_TwoGroupModel` | unimportant group fully pruned, important group fully preserved; global optimality |
| `test_taylor_vs_gradient_produce_different_pruning` | `_SingleLinearModel` | gradient and Taylor prune different individual weights; both locally optimal |
| `test_full_pipeline_single_sparsity` [qwen2, llama] | tiny HF from config | random saliency map, pruning optimal, sparsity achieved |
| `test_multi_sparsity_sweep_with_restore` [qwen2, llama] | tiny HF from config | 7 sparsity levels; pruning optimal at each; restore exact |

---

## Integration tests (`tests/test_gradient_map_integration.py`)

Run on **GPU** with a real pretrained model (Qwen/Qwen2.5-0.5B-Instruct truncated
to 2 layers) and a tiny in-memory SFT dataset (8 examples, no download required).

| Test | What it verifies |
|---|---|
| `test_grad_collect_trainer_produces_gradient_map` | GradCollectTrainer: weights unchanged, grad map covers all params, abs mode non-negative |
| `test_prune_sweep_end_to_end_with_real_model` | Full pipeline: EMA map → compute\_saliency\_scores → apply\_pruning at 4 sparsity levels → restore |

---

## How to run

```bash
# Unit tests only (CPU, fast, no downloads)
conda activate saescoping
export PYTHONPATH=experiments/saliency_pruning/toy_sweep_2026_03_14

pytest experiments/saliency_pruning/toy_sweep_2026_03_14/tests/unit/ -v

# Integration tests (GPU, downloads Qwen2.5-0.5B on first run, ~2-5 min)
# Ask before running — these require a GPU.
python experiments/saliency_pruning/toy_sweep_2026_03_14/tests/test_gradient_map_integration.py
```
