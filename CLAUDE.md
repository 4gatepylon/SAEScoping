# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAE Scoping is a toolkit for training language models with Sparse Autoencoders (SAEs). The core workflow involves:
1. Ranking SAE neurons by firing frequency on a target dataset
2. Pruning or training SAEs (from scratch or finetuned) to retain only in-domain features
3. Training models with SAE hooks (or additional layers) while freezing pre-SAE layer parameters
4. Evaluating in-domain vs. out-of-domain performance with the purpose of maximizing in-domain performance and minimizing out-of-domain performance. Also, it is critical to maximize safety by minimizing jailbreak attack success rate (ASR) and trojan (backdoor) attack success rate (ASR).

## Build & Development Commands

```bash
# Setup environment
conda create -n saescoping python=3.12 -y
pip install -e .

# Linting
ruff check . --fix

# Formatting
ruff format .

# Running tests
pytest .
```

## Environment Variables

- `WANDB_PROJECT`, `WANDB_RUN_NAME`, `WANDB_API_KEY` - Required for training and logging nicely
- `HF_TOKEN` - HuggingFace access
- `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY` - For API generators
- `CUDA_VISIBLE_DEVICES` - Used to share GPU cloud VM with others without conflict

## Architecture

### Directory Structure

- `sae_scoping/` - Pip-installable library. This supports the main methods for training and pruning SAEs and training models.
- `experiments/` - One-off research scripts (NOT libraries), use `deleteme*` prefix for temp files (files with `*deleteme*` will be gitignored).

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

- `xxx_` prefix: Work-in-progress code (this is commited, but once we PR into main it should be finished and the `xxx_` prefix should be removed)
- `deleteme*`: Temporary files (git-ignored)
- `XXX` in code denotes that this MUST be fixed before PRing into main. It is often useful to grep to find things still missing before we are done.
- `TODO` in code denotes things that we may want to change later on (but this MAY be PRed into main)
- `SQUASH` in git commits denotes that this commit should be squashed with adjacent SQUASH commits into a single commit before PRing into main

## Code Style

- Uses `beartype` for runtime type checking and `jaxtyping` for tensor shape annotations
- Maintain strict type annotations when modifying code
- Write succinct, minimal, simple code
- Create clear abstractions and document with docstrings as needed for complex components.
- Stateful code should be OOP (in classes). State is meant to be maintained in classes. Stateless code can be in its own functions. Stateless code with shared functionality should be grouped by module (file/folder) or static class, but module is preferable.