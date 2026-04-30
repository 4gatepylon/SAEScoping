# Testing Guide

How to validate the components on the `adriano/baselines` branch.

> **TODO(hadriano):** Tests are redundant and slow. CPU wanda tests call `compute_wanda_saliency` once per test instead of caching it as a module fixture; consolidate the 5 saliency-property tests into 1-2. GPU tests load a fresh 5GB model 4 times in `TestPruneEndToEnd`; share a single pruned-model fixture. `test_pt_hooks.py` has a dead `_test_pt_hooks_modify_inputs` integration test only reachable from `__main__`.

---

## Quick Start

Target specific test files rather than `pytest .` — collection imports `transformers` from every test module, which takes ~10s each even when tests are skipped.

```bash
# Run all CPU tests (no GPU needed)
CUDA_VISIBLE_DEVICES= python -m pytest -v -s --log-cli-level=INFO \
    sae_scoping/training/saliency/tests/test_wanda_cpu.py \
    sae_scoping/training/utils/hooks/test_pt_hooks.py

# Run scoping eval tests (mock judge only; set OPENAI_API_KEY for real judge tests)
CUDA_VISIBLE_DEVICES= python -m pytest -v -s --log-cli-level=INFO \
    sae_scoping/evaluation/test_scoping_eval.py

# Run GPU integration tests (requires CUDA)
CUDA_VISIBLE_DEVICES=0 python -m pytest -v -s --log-cli-level=INFO \
    sae_scoping/examples/test_wanda_gpu.py

# Run everything (slow collection — imports transformers 4x)
CUDA_VISIBLE_DEVICES=0 python -m pytest -v -s --log-cli-level=INFO .
```

**Flags explained:**
- `-s` — disables output capture so download progress bars and prints appear in real time
- `--log-cli-level=INFO` — streams Python logging live instead of buffering to end of test
- `CUDA_VISIBLE_DEVICES=` (empty) — hides all GPUs so `@requires_cuda` tests are skipped
- `CUDA_VISIBLE_DEVICES=0` — exposes GPU 0; change the number to pick a different GPU

**Model caching:** HuggingFace caches downloads to `~/.cache/huggingface/hub/`. First run downloads configs/tokenizers/models; subsequent runs use cache. Set `HF_HUB_OFFLINE=1` to skip the update check after the first download.

---

## Test Inventory

- **Wanda CPU tests** (`sae_scoping/training/saliency/tests/test_wanda_cpu.py`) — Validates saliency computation, mask generation, and pruning on a tiny Gemma-2 model. No GPU required.
- **Wanda GPU tests** (`sae_scoping/examples/test_wanda_gpu.py`) — End-to-end integration tests on a real Gemma-2-2b-it model: saliency shapes, mask sparsity, loss after pruning, generation, and PGD mask compatibility.
- **Scoping eval tests** (`sae_scoping/evaluation/test_scoping_eval.py`) — LLM-judge evaluation with mock and real (gpt-4.1-nano) judges. Mock tests are free; real tests require `OPENAI_API_KEY` and cost <$0.02.
- **PyTorch hooks tests** (`sae_scoping/training/utils/hooks/test_pt_hooks.py`) — Validates forward hook context manager. No GPU required.

---

## Validating Imports

```bash
python -c "from sae_scoping.datasets.qa_datasets import load_qa_dataset; print('ok')"
python -c "from sae_scoping.evaluation.loss import compute_loss; print('ok')"
python -c "from sae_scoping.training.saliency.wanda import compute_wanda_saliency; print('ok')"
python -c "from sae_scoping.training.pgd_trainer import PGDSFTTrainer; print('ok')"
```
