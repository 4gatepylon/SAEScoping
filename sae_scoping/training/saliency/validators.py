# Validators for pruning mask sweeps. Used by Wanda (and any future
# saliency method that sweeps sparsities on a fixed saliency map).

from __future__ import annotations

import warnings

import torch


class MaskSubsetValidator:
    """Validates that successive pruning masks are monotonically increasing in sparsity.

    When sweeping sparsities low-to-high on a fixed saliency map, each mask's
    zero-set must be a superset of the previous mask's zero-set (every position
    that was pruned stays pruned). Stores the previous mask as CPU bool tensors
    to minimize memory.

    Args:
        enabled: If False, skip validation and don't store masks (logs a warning).
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._prev_masks: dict[str, torch.Tensor] | None = None
        if not enabled:
            warnings.warn(
                "MaskSubsetValidator disabled (--low-memory): "
                "monotonicity of pruning masks will NOT be checked.",
                stacklevel=2,
            )

    def validate_and_update(self, masks: dict[str, torch.Tensor]) -> None:
        """Check that every position zeroed in the previous mask is also zeroed now, then store the new mask."""
        if not self.enabled:
            return
        bool_masks = {k: v.bool().cpu() for k, v in masks.items()}
        if self._prev_masks is not None:
            for name, prev in self._prev_masks.items():
                was_pruned = ~prev
                now_kept = bool_masks[name]
                violations = was_pruned & now_kept
                if violations.any():
                    n = int(violations.sum().item())
                    raise AssertionError(
                        f"Mask monotonicity violated for '{name}': "
                        f"{n} position(s) were pruned in the previous mask "
                        f"but kept in the current mask."
                    )
        self._prev_masks = bool_masks
