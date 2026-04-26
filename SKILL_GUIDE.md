# Skill Guide

Practical advice for continuing work on this codebase — gotchas, autonomous operation, and useful patterns.

---

## Environment Setup

```bash
# Always use the conda environment
/opt/miniconda3/envs/saescoping/bin/python -m pytest ...

# For OpenAI API (used by LLM judge):
set -a; . /Users/4gate/git-old/SAEScoping/.env; set +a
```

---

## Autonomous Operation Checklist

Before starting autonomous work (user AFK), verify these won't block:

```bash
# 1. Bash (sandboxed, auto-allowed)
echo "bash ok"

# 2. Python execution
/opt/miniconda3/envs/saescoping/bin/python -c "print('python ok')"

# 3. Git reads
git log --oneline -1

# 4. Pytest collection (no execution)
/opt/miniconda3/envs/saescoping/bin/python -m pytest --collect-only -q \
    sae_scoping/hyperparameter_optimization/test_binary_search.py 2>&1 | head -3

# 5. Network (if needed — requires sandbox network allowlist)
/opt/miniconda3/envs/saescoping/bin/python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co').status"
```

If any command prompts for permission, fix settings before the user leaves.

### Permission Configuration

Sandbox mode (`autoAllowBashIfSandboxed: true`) auto-allows all Bash commands.
Read/Edit/Write/Agent/WebFetch/WebSearch are explicitly allowed.
Network is restricted to allowlisted domains (see `.claude/settings.local.json`).

**Known issue:** Write to `.claude/` directory may still prompt even when Write is in the allow list. Use `_skills/` for scratch files instead.

---

## Useful Skills & Scripts

### Safe Exploration (`_skills/` and `.claude/skills/safe-explore/`)

| Script | Purpose |
|--------|---------|
| `branch_summary.py <branch>` | Unique commits, stat diff, file tree vs main |
| `branch_diff_files.py <branch>` | Added/modified/deleted files between branches |
| `file_on_branch.py <branch> <filepath>` | Read a file from any branch without checking out |
| `module_map.py <package_path>` | AST-based Python module mapper (classes, functions, exports) |
| `pytest_list.py <test_file>` | List tests without executing |
| `sfind.py <args>` | Safe `find` wrapper (blocks `-exec`, `-delete`, etc.) |

### Running Tests

```bash
# All CPU tests (~10 min)
/opt/miniconda3/envs/saescoping/bin/python -m pytest \
    sae_scoping/hyperparameter_optimization/test_binary_search.py \
    sae_scoping/training/saliency/tests/test_wanda_cpu.py \
    sae_scoping/training/saliency/tests/test_sparse_llm_cpu.py \
    sae_scoping/training/unlearning/tests/test_unlearning_cpu.py \
    -v

# Single test file
/opt/miniconda3/envs/saescoping/bin/python -m pytest \
    sae_scoping/hyperparameter_optimization/test_binary_search.py -v

# GPU tests (requires CUDA)
python experiments/baselines/test_wanda.py --device cuda:0
python experiments/baselines/test_sparse_llm.py --device cuda:0
python experiments/baselines/test_unlearning.py --device cuda:0
```

### Import Validation

After any package reorganization, verify imports:
```bash
/opt/miniconda3/envs/saescoping/bin/python -c "
from sae_scoping.training.saliency.wanda import compute_wanda_saliency
from sae_scoping.training.saliency.sparse_llm import compute_sparse_llm_masks
from sae_scoping.training.unlearning.rmu import unlearn_rmu
from sae_scoping.training.unlearning.npo import unlearn_npo
from sae_scoping.training.weight_pruning import prune_model
from sae_scoping.training.pgd_trainer import PGDSFTTrainer
from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval
print('all imports ok')
"
```

---

## Known Bugs & Fixes

### Must Fix (blocks tests)

1. **RMU undefined functions** (`rmu.py:154-197`): `_get_hidden_size`, `_get_num_layers`, `_get_layer_module` should be `get_hidden_size`, `get_num_layers`, `get_layer_module` (no underscore prefix). Blocks 4 CPU tests + GPU RMU test.

2. **Stale imports in test_pt_hooks.py** (line 19): `from utils.hooks.pt_hooks import ...` should be `from sae_scoping.training.sae_enhanced.hooks.pt_hooks import named_forward_hooks`. Blocks 2 tests.

### Watch Out For

3. **MPS device mismatch** (Mac only): TRL moves models to `mps:0` during `.train()`, post-training CPU tensor ops crash. Affects NPO and GradientDiff. Linux (CUDA/CPU) is fine.

4. **Cache key incomplete** (`sweep_sparsity.py:67`): Cache filename ignores `n_calibration`, `max_seq_len`, dataset. Change these between runs → stale results.

5. **Global thresholding OOM** (`sweep_sparsity.py:291`): Concatenates all saliency scores into single CPU tensor. ~36GB at 9B parameters.

6. **Judge data not held out** (`sweep_sparsity.py:377`): Judge evaluation pool overlaps with calibration/training data. Results may be inflated.

7. **Unknown models get 2B tuning** (`launch_sweeps.py:96`): Models not in the known list silently receive Gemma-2-2B hyperparameters. Will OOM on larger models.

### Lower Priority

See TESTING_GUIDE.md for the full inventory of 28 TODO(claude) annotations across HIGH/MEDIUM/LOW priority.

---

## Surprises & Non-Obvious Behaviors

### PGD Trainer's Dual Validation

`PGDSFTTrainer` has two sparsity checks: a mandatory pre-training check in `create_optimizer()` and an optional per-step check. The per-step check is cheap (`torch.any` per parameter) but catches mask/model mismatches after optimizer updates. Disable only if profiling shows it's a bottleneck.

### Weight Pruning Is In-Place

`weight_pruning.py` modifies model weights in-place. There's a `save_original_weights` option that clones ALL parameters (~18GB at 9B). If you need to undo pruning, save/reload the model instead.

### SparseLLM Shared Precomputation

`sparse_llm.py` separates shared precomputation (`precompute_shared_data()`) from per-sparsity mask computation. This lets you sweep multiple sparsity levels without recomputing activations. The shared data (X, Xinv, etc.) is large — keep `n_calibration` small.

### Unlearning Methods Return Different Things

- `unlearn_gradient_diff()` and `unlearn_npo()` return a `TrainerOutput` (from TRL)
- `unlearn_rmu()` returns nothing (modifies model in-place with custom training loop)

All three modify the model in-place, but only RMU explicitly freezes/unfreezes parameters.

### Judge Scores Are Fragile

`scoping_eval.py` hard-asserts that judge scores are integers in `{0, 1, 2}`. If the LLM judge returns anything else (e.g. "1.5", "N/A"), the entire evaluation level crashes. The judge template must be kept in sync with this assumption.

### Dataset Loading Paths

- `qa_datasets.py` — Clean, uses HuggingFace `datasets` library. Preferred for new work.
- `text_datasets.py` — Older, many TODO(Adriano) annotations. Works but needs cleanup.
- `messages_datasets.py` — Chat-formatted variant.

For the sweep scripts, calibration data comes from `text_datasets.py` and eval data from `qa_datasets.py`. They use different tokenization strategies.

---

## Multi-GPU Gotchas

1. **launch_sweeps.py uses round-robin GPU assignment.** If methods have different runtimes (they do — RMU >> WANDA), some GPUs sit idle.

2. **Subprocess output is swallowed** (`launch_sweeps.py:126`). If a child process crashes, the root cause is lost. Add `stdout=None, stderr=None` to `subprocess.Popen` for debugging.

3. **No `CUDA_VISIBLE_DEVICES` isolation.** Each subprocess gets a `--device cuda:N` argument but can still see all GPUs. A misbehaving method could grab memory on the wrong GPU.

4. **`empty_cache` timing** (`sweep_sparsity.py:472`): Cache is only cleared after both eval and judge, not between them. Judge OOM is possible if eval leaves residual tensors.

---

## What's Missing (Potential Future Work)

1. **No tests for SAE-enhanced model** (`test_sae_enhanced_gemma2.py` is an empty stub)
2. **No tests for evaluation system** (`scoping_eval.py` is only validated indirectly through sweeps)
3. **No CI/CD** — all tests are run manually
4. **No integration between pruning and unlearning** — they're separate pipelines
5. **No automated comparison** between pruning vs unlearning at equivalent capability levels
6. **Taylor and Gradient saliency** have no dedicated CPU tests (only tested via GPU integration tests)
7. **No seed management** — `torch.manual_seed` is not called before saliency computation

---

## Common Operations

### Adding a New Saliency Method

1. Create `sae_scoping/training/saliency/your_method.py`
2. Implement: `compute_your_method_saliency(model, data, ...) -> dict[str, Tensor]`
3. Add to `sweep_sparsity.py`'s method dispatch
4. Add CPU test in `sae_scoping/training/saliency/tests/test_your_method_cpu.py`
5. Reuse the tiny Gemma-2 fixture from `test_wanda_cpu.py` (copy the `@pytest.fixture` block)

### Adding a New Unlearning Method

1. Create `sae_scoping/training/unlearning/your_method.py`
2. Implement: `unlearn_your_method(model, tokenizer, forget_dataset, retain_dataset, ...) -> ...`
3. Add to `sweep_sparsity.py`'s method dispatch and `test_unlearning.py` GPU tests
4. Add CPU test class in `test_unlearning_cpu.py` (follow TestGradientDiff pattern)

### Running a Sparsity Sweep

```bash
# Dry run — check what would execute
python experiments/baselines/sweep_sparsity.py --help

# Single method, single GPU
python experiments/baselines/sweep_sparsity.py \
    --model google/gemma-2-2b-it \
    --method wanda \
    --device cuda:0

# Parallel across GPUs
python experiments/baselines/launch_sweeps.py \
    --model google/gemma-2-2b-it \
    --methods wanda sparse_llm \
    --gpus 0 1
```
