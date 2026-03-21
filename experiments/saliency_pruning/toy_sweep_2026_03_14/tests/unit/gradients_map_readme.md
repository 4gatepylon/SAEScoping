# Tests: `gradients_map` package

This document describes the tests for the `gradients_map` package, both the
existing ones and the newer end-to-end tests added in `test_gradient_map_e2e.py`.

---

## Existing tests (`test_gradients_map.py`)

Cover individual functions in isolation using tiny hand-crafted `nn.Module` stubs.

| What is tested | Test function(s) |
|---|---|
| `make_random_map` shape, range, determinism, trainable-only filtering | `test_make_random_map_*` |
| `_register_ema_hooks` signed accumulation | `test_register_ema_hooks_signed_*` |
| `_register_ema_hooks` abs-grad mode | `test_register_ema_hooks_abs_*` |
| `_VARIANT_SPECS` / `_ALL_VARIANTS` completeness | `test_variant_specs_*` |
| `_build_run_cmd` flag presence per variant | `test_build_run_cmd_*` |
| `make_taylor_map` math, non-negativity, missing-key skipping | `test_make_taylor_map_*` |
| `taylor_output_path` / `validate_taylor_source_path` | `test_taylor_output_path_*` |
| `assert_all_params_require_grad` | `test_assert_all_params_require_grad_*` |

---

## End-to-end property tests (`test_gradient_map_e2e.py`)

Verify realistic hook behaviour on real HuggingFace model architectures
instantiated *from config* (no weight downloads).

### Shared helpers

**`tests/unit/validators.py`** — property validators used across test files:
- `assert_gradient_map_covers_all_params` — every trainable param has a key
- `assert_gradient_map_nonneg` — all values ≥ 0 (abs mode)
- `assert_hook_fires_match_steps` — `_hook_fires[name] == n_steps`
- `assert_weights_unchanged_from_snapshot` — no weight mutation during hooks
- `assert_grad_maps_differ` — two maps are not element-wise identical

**`tests/unit/model_factories.py`** — tiny model factories:
- `make_tiny_qwen2()` — 1-layer Qwen2 from config, hidden\_size=64, CPU
- `make_tiny_llama()` — 1-layer Llama from config, hidden\_size=64, CPU
- `TINY_HF_FACTORIES` — parametrisation dict for `@pytest.fixture(params=...)`
- `make_known_saliency_fixture()` — analytically known pruning set (see Pruning tests)
- `make_taylor_vs_gradient_fixture()` — case where Taylor and gradient prune differently

### Tests

| Test | Architectures | What it verifies |
|---|---|---|
| `test_ema_hooks_weights_invariant_and_grads_populated` | qwen2, llama | weights unchanged; all trainable params in grad map; hook fires = n\_steps |
| `test_ema_hooks_abs_vs_signed` | qwen2, llama | abs mode ≥ 0 everywhere; abs and signed maps differ |
| `test_gradient_ranking_dead_params_get_zero_saliency` | custom nn.Module | params outside the forward graph have `grad=None`; not included in map |

---

## How to run

```bash
# From the experiment root
conda activate saescoping
export PYTHONPATH=experiments/saliency_pruning/toy_sweep_2026_03_14

# All unit tests
pytest experiments/saliency_pruning/toy_sweep_2026_03_14/tests/unit/ -v

# Just the new e2e gradient-map tests
pytest experiments/saliency_pruning/toy_sweep_2026_03_14/tests/unit/test_gradient_map_e2e.py -v
```
