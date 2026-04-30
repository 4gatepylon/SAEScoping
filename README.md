# SAE Scoping
Minimal viable version of 2024/2025 SAE Scoping work. The desired use-case is that you do:
1. Setup your environment with `conda create -n saescoping python=3.12 -y`
2. Install this and its depndencies with `pip install -e .`.

The folder-structure (etc...) should be self-explanatory. The pip-installable is inside `sae_scoping/` whereas `experiments` and other such folders are meant for one-off scripts (etc...) that are NOT libraries but may be important for results from the paper.

## Running tests

```bash
# CPU-only tests (skips GPU tests)
CUDA_VISIBLE_DEVICES= pytest sae_scoping/ -v

# GPU tests (requires CUDA)
CUDA_VISIBLE_DEVICES=0 pytest sae_scoping/ -v
```

Setting `CUDA_VISIBLE_DEVICES=` (empty) hides all GPUs from PyTorch, so tests decorated with `requires_cuda` are automatically skipped.
