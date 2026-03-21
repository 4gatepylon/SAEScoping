# Unit tests

Pure pytest tests for algorithmic components. No HuggingFace model download required — all tests use tiny `nn.Module` instances on CPU and run in seconds.

## Files

| File | What it tests |
|------|---------------|
| `test_sweep_eval_temp.py` | `build_sparsity_levels`, `compute_saliency_scores`, weight save/restore, all pruning validators (`sample_pruning_probes`, `assert_kept_weights_unchanged`, `assert_pruned_weights_are_zero`, `assert_zero_count_geq_target`), `apply_pruning` end-to-end, `save_generations`, `_is_run_complete`, `_build_sweep_cmd` |
| `test_gradients_map.py` | `make_random_map`, `_register_ema_hooks` (signed vs abs-grad), `_VARIANT_SPECS`/`_ALL_VARIANTS` completeness, `_build_run_cmd` per variant |

## How to run

```bash
# From the repo root
conda activate saescoping
export PYTHONPATH=experiments/saliency_pruning/toy_sweep_2026_03_14

pytest experiments/saliency_pruning/toy_sweep_2026_03_14/tests/unit/ -v
```

Or from the experiment directory directly:

```bash
cd experiments/saliency_pruning/toy_sweep_2026_03_14
conda activate saescoping
export PYTHONPATH=.

pytest tests/unit/ -v
```