# SAE Scoping
Minimal viable version of 2024/2025 SAE Scoping work. The desired use-case is that you do:
1. Setup your environment with `conda create -n saescoping python=3.12 -y`
2. Install this and its depndencies with `pip install -e .`.

The folder-structure (etc...) should be self-explanatory. The pip-installable is inside `sae_scoping/` whereas `experiments` and other such folders are meant for one-off scripts (etc...) that are NOT libraries but may be important for results from the paper.

## Running tests

See [TESTING_GUIDE.md](TESTING_GUIDE.md) for the full test inventory, recommended flags, and caching notes. Quick version:

```bash
# CPU-only (target specific files to avoid slow collection)
CUDA_VISIBLE_DEVICES= python -m pytest -v -s --log-cli-level=INFO \
    sae_scoping/training/saliency/tests/test_wanda_cpu.py \
    sae_scoping/training/utils/hooks/test_pt_hooks.py

# GPU tests
CUDA_VISIBLE_DEVICES=0 python -m pytest -v -s --log-cli-level=INFO \
    sae_scoping/examples/test_wanda_gpu.py
```
