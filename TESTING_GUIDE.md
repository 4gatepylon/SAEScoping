# Testing Guide

How to validate the components on the `adriano/baselines` branch.

---

## Quick Start

```bash
# Run all CPU tests (no GPU needed)
/opt/miniconda3/envs/saescoping/bin/python -m pytest \
    sae_scoping/training/saliency/tests/test_wanda_cpu.py \
    -v

# Run GPU integration tests (requires CUDA)
CUDA_VISIBLE_DEVICES=0 /opt/miniconda3/envs/saescoping/bin/python -m pytest \
    sae_scoping/examples/test_wanda_gpu.py -v
```

---

## Test Inventory

- **Wanda CPU tests** (`sae_scoping/training/saliency/tests/test_wanda_cpu.py`) — Validates saliency computation, mask generation, and pruning on a tiny Gemma-2 model. No GPU required.
- **Wanda GPU tests** (`sae_scoping/examples/test_wanda_gpu.py`) — End-to-end integration tests on a real Gemma-2-2b-it model: saliency shapes, mask sparsity, loss after pruning, generation, and PGD mask compatibility.
- **PyTorch hooks tests** (`sae_scoping/training/utils/hooks/test_pt_hooks.py`) — Validates forward hook context manager. No GPU required.

---

## Validating Imports

```bash
python -c "from sae_scoping.datasets.qa_datasets import load_qa_dataset; print('ok')"
python -c "from sae_scoping.evaluation.loss import compute_loss; print('ok')"
python -c "from sae_scoping.training.saliency.wanda import compute_wanda_saliency; print('ok')"
python -c "from sae_scoping.training.pgd_trainer import PGDSFTTrainer; print('ok')"
```
