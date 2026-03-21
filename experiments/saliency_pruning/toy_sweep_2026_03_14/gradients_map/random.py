"""Random saliency map baseline (i.i.d. Uniform[0, 1) per parameter)."""

import torch
from transformers import AutoModelForCausalLM


def make_random_map(model: AutoModelForCausalLM, seed: int = 42) -> dict[str, torch.Tensor]:
    """Return i.i.d. Uniform[0, 1) scores for every trainable parameter."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return {
        name: torch.rand(param.shape, generator=rng)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
