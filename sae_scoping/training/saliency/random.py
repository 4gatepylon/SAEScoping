"""Random saliency map baseline (i.i.d. Uniform[0, 1) per parameter)."""

import torch
from transformers import AutoModelForCausalLM

from sae_scoping.training.saliency.wanda import _SKIP_LAYER_NAMES


def _should_score(name: str) -> bool:
    """Return False for embedding/lm_head parameters."""
    parts = name.split(".")
    return not any(part in _SKIP_LAYER_NAMES for part in parts)


def make_random_map(model: AutoModelForCausalLM, seed: int = 42) -> dict[str, torch.Tensor]:
    """Return i.i.d. Uniform[0, 1) scores for every prunable parameter."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return {name: torch.rand(param.shape, generator=rng) for name, param in model.named_parameters() if param.requires_grad and _should_score(name)}
