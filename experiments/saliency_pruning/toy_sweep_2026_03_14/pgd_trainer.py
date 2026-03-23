"""
pgd_trainer.py

SFTTrainer subclass that enforces sparsity throughout recovery fine-tuning
via projected gradient descent (PGD).

After each optimiser step the trainer re-zeros every weight whose position
was zeroed by pruning, projecting the parameters back onto the feasible set

    C = { W  :  W[~mask] = 0 }

This prevents recovery SFT from re-growing pruned weights while still
letting surviving weights adapt freely.  The update rule is:

    W_{t+1} = P_C( W_t - lr * grad_t )
            = clip_to_zero( Adam/SGD update )

Public API
----------
PGDSFTTrainer            — drop-in replacement for SFTTrainer; accepts a
                           ``masks`` dict and applies the projection after
                           every optimiser step.
build_pgd_masks_from_model — convenience helper: derive a masks dict from the
                             current (already-pruned) model weights.
"""

from __future__ import annotations

from typing import Optional

import torch
from trl import SFTTrainer


# ---------------------------------------------------------------------------
# Convenience mask builder
# ---------------------------------------------------------------------------


def build_pgd_masks_from_model(
    model: torch.nn.Module,
    param_names: Optional[list[str]] = None,
) -> dict[str, torch.Tensor]:
    """Build a PGD mask dict by inspecting the model's current weight values.

    A weight element is treated as *pruned* (mask = False) if its value is
    exactly zero.  All other elements are True (free to update).

    Only parameters that have at least one zero entry are included in the
    returned dict; fully-dense parameters are omitted because their
    projection would be a no-op during training.

    Masks are returned on CPU to minimise GPU memory pressure; they are moved
    to the appropriate device inside :class:`PGDSFTTrainer` at setup time.

    Args:
        model: A pruned (or partially pruned) model whose zero pattern
               defines the sparsity constraint.  Weights are read but not
               modified.
        param_names: Optional allowlist of parameter names to consider.
                     If ``None``, all named parameters are inspected.

    Returns:
        Dict mapping parameter name → CPU bool keep-mask (True = free to
        update, False = must stay at zero).
    """
    if param_names is not None:
        param_names_set = set(param_names)
        named_params = (
            (n, p) for n, p in model.named_parameters() if n in param_names_set
        )
    else:
        named_params = model.named_parameters()  # type: ignore[assignment]

    masks: dict[str, torch.Tensor] = {}
    for name, param in named_params:
        mask = (param.data != 0).cpu()
        if not mask.all():  # skip fully-dense parameters (no-op projection)
            masks[name] = mask
    return masks


# ---------------------------------------------------------------------------
# Projection callable — wraps an optimiser step with the PGD zero-projection
# ---------------------------------------------------------------------------


class _ProjectedStep:
    """Callable that wraps an optimiser step with the PGD projection.

    Replaces ``optimizer.step`` in-place.  Each call:
        1. Invokes the original ``step`` (gradient update).
        2. Iterates over all parameter groups in the optimiser.
        3. For each parameter that has a registered mask, zeros out the
           positions where the mask is ``False``.

    Implemented as a class (rather than a nested function) so that it can
    be tested independently and so all captured state is explicit.

    Args:
        original_step: The unpatched ``optimizer.step`` callable.
        optimizer: The optimiser whose ``param_groups`` we iterate over.
        masks_by_id: Mapping from ``id(param)`` to an on-device bool mask.
                     Populated once by
                     :meth:`PGDSFTTrainer._build_mask_id_map`.
    """

    def __init__(
        self,
        original_step,
        optimizer: torch.optim.Optimizer,
        masks_by_id: dict[int, torch.Tensor],
    ) -> None:
        self._original_step = original_step
        self._optimizer = optimizer
        self._masks_by_id = masks_by_id

    def __call__(self, closure=None):
        result = self._original_step(closure)
        with torch.no_grad():
            for group in self._optimizer.param_groups:
                for param in group["params"]:
                    mask = self._masks_by_id.get(id(param))
                    if mask is not None:
                        param.data[~mask] = 0.0
        return result


# ---------------------------------------------------------------------------
# Projected-gradient-descent SFT trainer
# ---------------------------------------------------------------------------


class PGDSFTTrainer(SFTTrainer):
    """SFTTrainer with projected gradient descent.

    After every optimiser step the trainer re-zeros the weights at positions
    that are ``False`` in the corresponding entry of ``masks``, projecting
    parameters back onto the sparse feasible set established by pruning.

    The effective update at each step is:

        W_{t+1} = P_C( optimiser_step(W_t, grad_t) )

    where the projection P_C simply sets  W[~mask] = 0.

    Parameters
    ----------
    masks:
        Dict mapping *parameter name* (as returned by
        ``model.named_parameters()``) to a boolean ``torch.Tensor`` of the
        same shape as that parameter.  ``True`` means the weight is free to
        change; ``False`` means it must stay at zero.  Tensors may live on
        any device — they are moved to each parameter's device the first time
        ``create_optimizer`` is called.
    **kwargs:
        Forwarded verbatim to :class:`trl.SFTTrainer`.

    Example
    -------
    >>> from pgd_trainer import PGDSFTTrainer, build_pgd_masks_from_model
    >>> masks = build_pgd_masks_from_model(pruned_model)
    >>> trainer = PGDSFTTrainer(
    ...     masks=masks,
    ...     model=pruned_model,
    ...     args=training_args,
    ...     train_dataset=sft_dataset,
    ... )
    >>> trainer.train()
    """

    def __init__(
        self,
        masks: dict[str, torch.Tensor],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._pgd_masks: dict[str, torch.Tensor] = masks
        # Populated lazily in create_optimizer once the model is on its final
        # device and the optimiser has been constructed.
        self._pgd_masks_by_param_id: dict[int, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Optimiser creation — install the projection hook
    # ------------------------------------------------------------------

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create the base optimiser, then patch its ``step`` to apply PGD."""
        optimizer = super().create_optimizer()
        self._build_mask_id_map()
        self._install_projection_hook(optimizer)
        return optimizer

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_mask_id_map(self) -> None:
        """Build a ``{id(param) → on-device mask}`` lookup table.

        Keying by ``id(param)`` (rather than name) eliminates string-hash
        overhead from the hot path inside the patched ``optimizer.step``.
        Masks are moved to each parameter's device here, once at setup time,
        rather than on every step.

        Raises:
            ValueError: If any mask's shape does not match its parameter.
        """
        self._pgd_masks_by_param_id = {}
        for name, param in self.model.named_parameters():
            mask = self._pgd_masks.get(name)
            if mask is None:
                continue
            if mask.shape != param.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match parameter "
                    f"'{name}' shape {param.shape}."
                )
            self._pgd_masks_by_param_id[id(param)] = mask.to(
                device=param.device, dtype=torch.bool
            )

    def _install_projection_hook(self, optimizer: torch.optim.Optimizer) -> None:
        """Replace ``optimizer.step`` with a :class:`_ProjectedStep` instance.

        After this call ``optimizer.step(...)`` will:
            1. Execute the original gradient update.
            2. Zero out all masked positions via :class:`_ProjectedStep`.

        Args:
            optimizer: The freshly constructed optimiser whose ``step``
                       attribute will be replaced in-place.
        """
        optimizer.step = _ProjectedStep(
            original_step=optimizer.step,
            optimizer=optimizer,
            masks_by_id=self._pgd_masks_by_param_id,
        )
