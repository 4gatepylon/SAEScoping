# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAE Scoping is a toolkit for training language models with Sparse Autoencoders (SAEs). The core workflow involves:
1. Ranking SAE neurons by firing frequency on a target dataset
2. Pruning SAEs to retain only in-domain features
3. Training models with SAE hooks while freezing pre-SAE layer parameters
4. Evaluating in-domain vs. out-of-domain performance

## Build & Development Commands

```bash
# Setup environment
conda create -n saescoping python=3.12 -y
pip install -e .

# Linting
ruff check . --fix

# Running tests
pytest sae_scoping/utils/hooks/test_pt_hooks.py
pytest sae_scoping/xxx_models/test_sae_enhanced_gemma2.py
```

## Environment Variables

- `WANDB_PROJECT`, `WANDB_RUN_NAME`, `WANDB_API_KEY` - Required for training
- `HF_TOKEN` - HuggingFace access
- `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` - For API generators

## Architecture

### Directory Structure

- `sae_scoping/` - Pip-installable library
- `experiments/` - One-off research scripts (NOT libraries), use `deleteme*` prefix for temp files

### Core Training Pipeline (`sae_scoping/trainers/sae_enhanced/`)

- `rank.py`: `rank_neurons()` - Accumulates firing counts to identify important SAE features
- `prune.py`: `get_pruned_sae()` - Creates masked SAE wrapper retaining top-k neurons
- `train.py`: `train_sae_enhanced_model()` - SFT training with hooks and frozen pre-SAE parameters

### Hook System (`sae_scoping/utils/hooks/`)

PyTorch forward hooks for activation engineering:

- `pt_hooks.py`: `named_forward_hooks()` context manager, `filter_hook_fn()` for tensor transformations
- `pt_hooks_stateful.py`: `Context` class for passing state between hooks
- `sae.py`: `SAEWrapper` (flattening wrapper), `SAELensEncDecCallbackWrapper` (callback on SAE latents)

Usage pattern:
```python
from functools import partial
from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
from sae_scoping.utils.hooks.sae import SAEWrapper

sae_wrapper = SAEWrapper(sae)
hook_dict = {"model.layers.0": partial(filter_hook_fn, sae_wrapper)}
with named_forward_hooks(model, hook_dict):
    model(**batch)
```

### Naming Conventions

- `xxx_` prefix: Work-in-progress code
- `deleteme*`: Temporary files (git-ignored)

## Code Style

- Uses `beartype` for runtime type checking and `jaxtyping` for tensor shape annotations
- Maintain strict type annotations when modifying code