# Test plan: `gradients_map` package

This document specifies the **new tests to be written** for the `gradients_map` package.
These are in addition to the existing unit tests in `test_gradients_map.py`, which cover
`make_random_map`, `_register_ema_hooks` (signed vs. abs), `_VARIANT_SPECS` consistency,
`_build_run_cmd`, `make_taylor_map`, and path helpers.

The existing tests verify individual functions in isolation using hand-crafted
`nn.Module` stubs.  The tests below verify **realistic end-to-end pipelines** with
properties that matter for the saliency computation to be scientifically correct.

---

## Guiding principles

- Every test must run on CPU with no HuggingFace model downloads.
- Prefer **property validators** (modular functions that assert invariants) over
  inline assertions.  Each validator should be importable and reusable across tests.
- Use the **known-gradient synthetic model** (see §Model design below) wherever we want
  to assert *which* parameters receive high or low gradient magnitude.  This makes
  correctness claims precise rather than statistical.
- Use **tiny real transformer configs** (no pretrained weights, initialised randomly)
  for tests that must exercise the HuggingFace / SFT trainer interface.

---

## §Model design: the known-gradient synthetic model

### Purpose

To test that `GradCollectTrainer` actually computes gradient saliency for the *right*
parameters, we need a model where we can predict *a priori* which weights will have high
vs. low gradient magnitude.

### Construction (`make_known_gradient_model`)

Write a factory function that returns `(model, tokenizer, dataset, expected_ranking)`
where `expected_ranking` maps parameter names to expected relative saliency (high / low).

**Architecture:**

```
Embedding(vocab_size=4, dim=1)   — one scalar per token
Linear(1, 4, bias=False)         — LM head, one scalar per output token
```

**Weight initialisation:**

```
embed.weight[TARGET_TOKEN]  =  +BIG   # large positive → large gradient
embed.weight[FILLER_TOKEN]  =  +TINY  # small positive → small gradient
embed.weight[DEAD_TOKEN]    =   0.0   # exactly zero embedding → ~zero gradient
lm_head.weight[TARGET_TOKEN] = +BIG
lm_head.weight[OTHER_TOKENS] = -BIG
```

**Dataset:** always input `[FILLER_TOKEN]`, target `TARGET_TOKEN`.

**Why this works:**  On every forward pass the hidden state is `embed[FILLER_TOKEN]`
(small positive scalar).  The loss gradient back-propagates through `lm_head` then
through the embedding lookup.  Because the embedding for `DEAD_TOKEN` is never looked
up, its gradient is exactly zero.  Because `FILLER_TOKEN` has a small embedding, its
gradient through the LM head is proportionally small.  The exact ordering is determined
by the architecture and can be computed analytically.

The function should return the analytically expected ranking so tests can assert against
it without running any approximation.

---

## §Real-transformer tiny config models

For tests that exercise the full `SFTTrainer` / HuggingFace interface use randomly-
initialised models from config — **no weights are downloaded**.

Supported configs to try (in order of preference; test should skip gracefully if a
package is not installed):

| Config class | Suggested tiny params |
|---|---|
| `Qwen2Config` | `num_hidden_layers=1, hidden_size=32, num_attention_heads=2, intermediate_size=64, vocab_size=256` |
| `Qwen3Config` | same dims as above |
| `LlamaConfig` | `num_hidden_layers=1, hidden_size=32, num_attention_heads=2, intermediate_size=64, vocab_size=256` |
| `Gemma2Config` | `num_hidden_layers=1, hidden_size=32, num_key_value_heads=2, head_dim=16, intermediate_size=64, vocab_size=256` |
| `Gemma3Config` | same dims as Gemma2 |

Write a helper `make_tiny_causal_lm(config_class, **overrides)` that instantiates the
model from config, verifies `.model` attribute exists, and returns it on CPU.

Write a helper `make_synthetic_sft_dataset(tokenizer, n_examples=8, seq_len=16)` that
produces a minimal HuggingFace `Dataset` with a `"text"` column of short tokenised
sequences.  No real text required — random token IDs formatted as a string are fine.

---

## Tests to implement

### T-GM-1 · `GradCollectTrainer` — end-to-end property test

**File:** `tests/unit/test_gradients_map_e2e.py`

**Model:** tiny Qwen2 from config (fall back through the list above if unavailable).

**Steps:**
1. Instantiate the model and a synthetic SFT dataset (≤8 rows, seq_len=16).
2. Run `GradCollectTrainer` for `num_epochs=1` on the synthetic dataset.
3. Collect the resulting EMA gradient map via `trainer.ema_grads()`.

**Property validators to call after step 3:**

```python
def assert_weights_unchanged(model_before, model_after):
    """All weight tensors are byte-for-byte identical before and after training."""

def assert_gradient_map_covers_all_params(model, grad_map):
    """grad_map has exactly one key per named parameter that requires_grad."""

def assert_gradient_map_non_negative(grad_map):
    """All values are >= 0 (abs_grad=True mode) or may be mixed (abs_grad=False)."""

def assert_hook_fires_match_steps(model, expected_n_steps):
    """model._hook_fires[name] == expected_n_steps for every hooked parameter."""

def assert_zero_grad_is_noop(model, grad_map):
    """Calling model.zero_grad() does NOT clear param.grad (no-op patch is active)."""
```

**Also test the abs vs. signed divergence on this model:**
- Run once with `abs_grad=False`, once with `abs_grad=True`, same random seed.
- Assert the maps are *not* element-wise equal (they must differ on some parameter
  because a randomly-initialised model will have mixed-sign gradients across the
  synthetic data).
- Assert the abs-mode map is element-wise non-negative everywhere.

---

### T-GM-2 · `GradCollectTrainer` — correctness of gradient ranking

**File:** `tests/unit/test_gradients_map_e2e.py`

**Model:** the known-gradient synthetic model from §Model design.

**Steps:**
1. Build the synthetic model with analytically known gradient ranking.
2. Run `GradCollectTrainer` for a small number of steps (≥3).
3. Compare the resulting gradient map against the analytic expected ranking.

**Property validators:**

```python
def assert_gradient_ranking_matches_expected(grad_map, expected_ranking):
    """
    For each pair (high_param, low_param) in expected_ranking, verify:
        grad_map[high_param].mean() > grad_map[low_param].mean()
    Dead parameters (expected zero gradient) must have near-zero map values.
    """

def assert_dead_params_near_zero(grad_map, dead_param_names, tol=1e-6):
    """Parameters that were never activated have gradient map values < tol."""
```

This test is the key one that verifies the saliency computation is computing the
*right* thing, not just *something*.

---

### T-GM-3 · Taylor map derived from a known gradient map

**File:** `tests/unit/test_gradients_map_e2e.py`

**Model:** the known-gradient synthetic model.

**Steps:**
1. Build the model; set weights to known values.
2. Manually construct a gradient map (no training needed — use the analytic values).
3. Call `make_taylor_map(grad_map, model)`.
4. For each parameter, compute the expected Taylor score analytically:
   `expected[name] = (grad_map[name] * model_weights[name]).abs()`

**Property validators:**

```python
def assert_taylor_equals_analytic(taylor_map, grad_map, model):
    """taylor_map[name] == |grad_map[name] * param.data| for every key."""

def assert_taylor_differs_from_gradient(taylor_map, gradient_map):
    """
    Taylor and gradient rankings are not identical when weights have non-unit
    magnitude (which is guaranteed by construction in the known model).
    Specifically: a parameter with small gradient but large weight must rank
    higher in Taylor than in gradient-only scoring.
    """
```

The second validator is the critical one — it asserts that Taylor scoring is
*meaningfully different* from gradient scoring and in the expected direction.

---

## Validator module

All validators above should live in `tests/unit/validators.py` (new file), importable
from any test file:

```python
# tests/unit/validators.py
def assert_weights_unchanged(model_before_snapshot, model) -> None: ...
def assert_gradient_map_covers_all_params(model, grad_map) -> None: ...
def assert_gradient_map_non_negative(grad_map) -> None: ...
def assert_hook_fires_match_steps(model, expected_n_steps: int) -> None: ...
def assert_zero_grad_is_noop(model, grad_map) -> None: ...
def assert_gradient_ranking_matches_expected(grad_map, expected_ranking) -> None: ...
def assert_dead_params_near_zero(grad_map, dead_param_names, tol=1e-6) -> None: ...
def assert_taylor_equals_analytic(taylor_map, grad_map, model) -> None: ...
def assert_taylor_differs_from_gradient(taylor_map, gradient_map) -> None: ...
```

---

## What the existing tests already cover (do not duplicate)

- `make_random_map`: shape, range, determinism, trainable-only filtering ✅
- `_register_ema_hooks`: signed accumulation, alternating-sign convergence, abs vs.
  signed on a single backward pass ✅
- `_VARIANT_SPECS` / `_ALL_VARIANTS` completeness ✅
- `_build_run_cmd`: flag presence per variant ✅
- `make_taylor_map`: math correctness, non-negativity, missing-key skipping ✅
- `taylor_output_path` / `validate_taylor_source_path` ✅

---

## How to run

```bash
cd experiments/saliency_pruning/toy_sweep_2026_03_14
conda activate saescoping
export PYTHONPATH=.
pytest tests/unit/test_gradients_map_e2e.py -v
```
