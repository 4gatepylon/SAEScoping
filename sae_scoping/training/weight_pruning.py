"""Save/restore model weights and the global quantile sample budget.

Earlier versions of this module also held a 3-phase pruning pipeline
(``compute_keep_masks`` / ``apply_keep_masks_streaming`` / ``prune_model``).
That code was superseded by the saliency-method-specific paths in
``sae_scoping.training.saliency`` and removed.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedModel


def save_original_weights(model: PreTrainedModel) -> dict[str, torch.Tensor]:
    """Clone all parameter data to CPU for later restoration."""
    # TODO(claude) priority:medium: clones every parameter, including embeddings,
    # lm_head, and layer norms — ~18 GB CPU RAM per 9B job, ~36 GB if two 9B
    # sweeps run concurrently on the same box. Only pruning-eligible params (the
    # keys that will appear in masks) need to be saved; skip the rest.
    return {name: param.data.cpu().clone() for name, param in model.named_parameters()}


def restore_original_weights(
    model: PreTrainedModel,
    original_weights: dict[str, torch.Tensor],
) -> None:
    """Restore model parameters in-place from a CPU copy."""
    for name, param in model.named_parameters():
        if name in original_weights:
            param.data.copy_(original_weights[name].to(param.device))


# Number of elements sampled from all scores to estimate the global quantile.
# 10 M samples over 9 B elements gives < 0.01 % quantile error — more than
# sufficient for weight pruning. Consumed by
# ``sae_scoping.training.saliency.dispatch.masks_for_sparsity``.
_THRESHOLD_SAMPLE_BUDGET = 10_000_000
