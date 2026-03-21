# Test plan: `sweep_eval_temp.py`

This document specifies the **new tests to be written** for `sweep_eval_temp.py`.
These complement the existing unit tests in `test_sweep_eval_temp.py`, which already
cover `build_sparsity_levels`, `compute_saliency_scores`, weight save/restore,
pruning validators, `apply_pruning` end-to-end with random saliency, `save_generations`,
`_is_run_complete`, and `_build_sweep_cmd`.

The gap in the existing tests is that they use **random saliency tensors**, so they can
only assert structural properties (counts, shapes, round-trips).  They cannot assert
that the *correct* weights are being pruned — which is the entire point of the system.

---

## Guiding principles

- All tests must run on CPU with no downloads.
- Use the **known-gradient synthetic model** (same design as in `gradients_map_readme.md`)
  wherever we need to assert which weights are pruned.
- Use **tiny real transformer configs** (randomly-initialised, no pretrained weights) for
  end-to-end pipeline tests that must exercise the HuggingFace interface.
- Prefer **property validators** in a shared `tests/unit/validators.py`.  Validators are
  plain functions, not test functions; they raise `AssertionError` with descriptive
  messages and can be composed freely.
- Model configs to support (gracefully skip if package unavailable):
  `Qwen2Config`, `Qwen3Config`, `LlamaConfig`, `Gemma2Config`, `Gemma3Config`.
  See `gradients_map_readme.md §Real-transformer tiny config models` for suggested dims.

---

## §Known-gradient synthetic model (recap)

The same `make_known_gradient_model()` factory described in `gradients_map_readme.md`
is used here.  It returns `(model, saliency_map, expected_prune_set)` where:

- `model` has analytically-known weight values.
- `saliency_map` is constructed analytically (or from a few training steps) so that
  the ranking is known a priori.
- `expected_prune_set` is the set of `(param_name, flat_index)` pairs that should be
  zeroed at each target sparsity level.

The construction should be documented inside the factory, including the math that
determines which weights are least salient.

The key insight: the model has parameters with **exactly zero gradient** (dead
parameters — e.g. embeddings that are never activated) and parameters with **large
gradient** (the ones that directly drive the correct-token prediction).  We therefore
know *a priori* that dead parameters should always be pruned first at any non-zero
sparsity, and important parameters should be pruned last.

---

## Tests to implement

### T-SE-1 · `apply_pruning` selects the correct weights

**File:** `tests/unit/test_sweep_eval_temp_e2e.py`

**Model:** known-gradient synthetic model.

**Steps:**
1. Build the model; construct the analytic saliency scores (gradient mode).
2. Call `apply_pruning(model, scores, sparsity_fraction=S)` for a small sparsity S
   chosen so that we know exactly which weights fall below the threshold.
3. Assert that each analytically expected pruned position is zero.
4. Assert that each analytically expected kept position is non-zero.

**Property validators:**

```python
def assert_correct_weights_pruned(model, saliency_scores, sparsity_fraction, tol=0.0):
    """
    Flatten all saliency scores into a single sorted list.
    Identify the bottom n_prune entries by value.
    Assert every corresponding model weight is exactly 0.0.
    Assert every top-(total - n_prune) entry is non-zero.
    """

def assert_pruning_is_lowest_saliency(model, saliency_scores):
    """
    For every zero weight in the model, assert its saliency score is less than or
    equal to every non-zero weight's saliency score (across all parameters jointly).
    This is a global optimality check: no non-zero weight has lower saliency than
    any zero weight.
    """
```

`assert_pruning_is_lowest_saliency` is the most important validator in this file.
It makes the correctness claim precise: we do not just check counts, we check that the
*right* individual weights were selected.

---

### T-SE-2 · `apply_pruning` with Taylor scores selects the correct weights

**File:** `tests/unit/test_sweep_eval_temp_e2e.py`

**Model:** known-gradient synthetic model with explicitly set weights and gradient map.

**Steps:**
1. Build the model; set weights to known values.
2. Construct the gradient map analytically.
3. Derive Taylor scores: `scores[name] = |grad[name] * weight[name]|`.
4. Call `apply_pruning(model, taylor_scores, sparsity_fraction=S)`.
5. Assert the correct weights are pruned using `assert_correct_weights_pruned`.

**Additional validator:**

```python
def assert_taylor_pruning_differs_from_gradient_pruning(
    model, grad_scores, taylor_scores, sparsity_fraction
):
    """
    Verify that the set of pruned weights under Taylor scoring is different from
    the set under gradient-only scoring (when weights have non-unit magnitude).
    This guards against a bug where taylor scores silently degenerate to gradient scores.
    """
```

---

### T-SE-3 · Full gradient-map → prune pipeline on a tiny real transformer

**File:** `tests/unit/test_sweep_eval_temp_e2e.py`

**Model:** tiny Qwen2 from config (fall back through the list above).

**Steps:**
1. Instantiate the model from config (random weights, no download).
2. Create a synthetic saliency map via `make_random_map` (simulates a pre-computed
   `.safetensors` file loaded into memory).
3. Call `compute_saliency_scores(model, saliency_map, "gradient")`.
4. Call `apply_pruning(model, scores, sparsity_fraction=0.5)`.
5. Restore weights and repeat with `"taylor"` scoring.

**Property validators to call after each pruning:**

```python
def assert_pruning_is_lowest_saliency(model, saliency_scores): ...  # from T-SE-1
def assert_kept_weights_match_pre_prune_snapshot(model, snapshot, mask): ...
def assert_sparsity_fraction_achieved(model, saliency_scores, target, tol=0.01):
    """Actual fraction of zeroed scored weights is within tol of target."""
def assert_restore_is_exact(model, snapshot):
    """After restore_original_weights, every weight is torch.equal to snapshot."""
```

This test exercises the actual data flow used in production:
`safetensors file → compute_saliency_scores → apply_pruning → restore`.

---

### T-SE-4 · Multi-sparsity sweep with cumulative restoration

**File:** `tests/unit/test_sweep_eval_temp_e2e.py`

**Model:** tiny Qwen2 from config.

**Steps:**
1. Snapshot original weights.
2. For sparsity levels `[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]`:
   a. Restore from snapshot (so each level runs on a clean unpruned model).
   b. Call `apply_pruning`.
   c. Apply all property validators below.
   d. Log the actual sparsity achieved for inspection.

**Property validators:**

```python
def assert_sparsity_fraction_achieved(model, saliency_scores, target, tol=0.01): ...
def assert_pruning_is_lowest_saliency(model, saliency_scores): ...
def assert_restore_is_exact(model, snapshot): ...
```

`assert_restore_is_exact` is called *after* restore, not after prune.  The sweep
verifies that the restore is bit-perfect at every level — a regression test for the
bug fixed earlier where `save_original_weights` was called inside the loop.

**Also verify:** at `sparsity=0.0` no weight changes; at `sparsity=1.0` every scored
weight is exactly zero.

---

### T-SE-5 · Saliency ranking is preserved through `compute_saliency_scores`

**File:** `tests/unit/test_sweep_eval_temp_e2e.py`

**Model:** known-gradient synthetic model.

This tests that `compute_saliency_scores` does not silently reorder or transform
scores in a way that corrupts the pruning selection.

**Steps:**
1. Build the known model; construct the raw gradient map analytically.
2. Call `compute_saliency_scores(model, raw_map, "gradient")`.
3. Assert the output is `|raw_map[name]|` for each key — same values as the input,
   just absolute (no reordering, no missing keys, no extra keys).
4. Call `compute_saliency_scores(model, raw_map, "taylor")`.
5. Assert the output equals `(raw_map[name] * weight[name]).abs()` analytically.
6. Assert the gradient and Taylor rankings *differ* on at least one parameter pair
   (i.e., Taylor changes the relative ordering due to weight magnitude).

**Validators:**

```python
def assert_gradient_scores_equal_abs_map(scores, raw_map): ...
def assert_taylor_scores_equal_analytic(scores, raw_map, model): ...
def assert_score_rankings_differ(scores_a, scores_b):
    """At least one parameter ranks higher in scores_a than scores_b and vice versa."""
```

---

## Validator module

All validators should live in `tests/unit/validators.py` (shared with the gradients_map
tests).  Group them into two sections with a comment separator:

```
# ── gradients_map validators ────────────────────────────────────────────────
def assert_weights_unchanged(...)
def assert_gradient_map_covers_all_params(...)
...

# ── sweep_eval_temp validators ───────────────────────────────────────────────
def assert_correct_weights_pruned(...)
def assert_pruning_is_lowest_saliency(...)
def assert_sparsity_fraction_achieved(...)
def assert_restore_is_exact(...)
def assert_taylor_pruning_differs_from_gradient_pruning(...)
def assert_gradient_scores_equal_abs_map(...)
def assert_taylor_scores_equal_analytic(...)
def assert_score_rankings_differ(...)
```

---

## What the existing tests already cover (do not duplicate)

- `build_sparsity_levels`: grid generation, CSV override, sort ✅
- `compute_saliency_scores`: gradient=`|tensor|`, taylor=`|grad*weight|`, unknown type ✅
- `save_original_weights` / `restore_original_weights`: clone independence, round-trip ✅
- `sample_pruning_probes`: shape, value correctness, skips missing ✅
- `assert_kept_weights_unchanged`, `assert_pruned_weights_are_zero`,
  `assert_zero_count_geq_target`: validator self-tests with mock masks ✅
- `apply_pruning`: 0%/50%/100% counts, round-trip with random saliency ✅
- `save_generations`, `_is_run_complete`, `_build_sweep_cmd` ✅

**The critical gap filled by the tests above:** the existing tests only verify *how many*
weights are pruned, never *which* ones.  The new tests verify that `apply_pruning`
always prunes the globally lowest-saliency weights — the only thing that makes saliency-
based pruning different from random pruning.

---

## How to run

```bash
cd experiments/saliency_pruning/toy_sweep_2026_03_14
conda activate saescoping
export PYTHONPATH=.
pytest tests/unit/test_sweep_eval_temp_e2e.py -v
```
