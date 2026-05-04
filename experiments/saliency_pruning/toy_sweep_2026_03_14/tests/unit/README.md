# Unit tests

Pure pytest tests for algorithmic components.  All tests run on CPU — no HuggingFace
weight downloads required.  Tests use either tiny hand-crafted `nn.Module` stubs or
randomly initialised HuggingFace models created from config (e.g. Qwen2Config).

---

## Files

| File | What it tests |
|------|---------------|
| `test_gradients_map.py` | `make_random_map`, `_register_ema_hooks` (signed / abs), `_VARIANT_SPECS` / `_ALL_VARIANTS`, `_build_run_cmd`, `make_taylor_map`, path helpers, `assert_all_params_require_grad` |
| `test_sweep_eval_temp.py` | `build_sparsity_levels`, `compute_saliency_scores`, weight save/restore, all pruning validators, `apply_pruning` end-to-end, `save_generations`, `_is_run_complete`, `_build_sweep_cmd` |
| `test_gradient_map_e2e.py` | `_register_ema_hooks` on real HF architectures (Qwen2, Llama) from config; abs vs. signed modes; dead-branch parameter isolation |
| `test_sweep_pipeline_e2e.py` | `apply_pruning` correctness with analytically known saliency; Taylor vs. gradient divergence; full pipeline on tiny HF models; multi-sparsity sweep with restore |

## Shared helpers

| File | Purpose |
|------|---------|
| `validators.py` | Reusable property-validator functions (`assert_pruning_is_lowest_saliency`, `assert_gradient_map_covers_all_params`, etc.) |
| `model_factories.py` | Tiny HF model factories (`make_tiny_qwen2`, `make_tiny_llama`) and analytical test fixtures (`make_known_saliency_fixture`, `make_taylor_vs_gradient_fixture`) |

---

## How to run

```bash
# From the repo root
conda activate saescoping
export PYTHONPATH=experiments/saliency_pruning/toy_sweep_2026_03_14

pytest experiments/saliency_pruning/toy_sweep_2026_03_14/tests/unit/ -v
```

Or from the experiment directory:

```bash
cd experiments/saliency_pruning/toy_sweep_2026_03_14
conda activate saescoping
export PYTHONPATH=.

pytest tests/unit/ -v

# Run a single test file
pytest tests/unit/test_gradient_map_e2e.py -v
pytest tests/unit/test_sweep_pipeline_e2e.py -v
```

---

## Integration tests

Integration tests live in `tests/` (not `tests/unit/`) and require a GPU.
See `tests/unit/sweep_eval_temp_readme.md` for details on
`tests/test_gradient_map_integration.py`.
