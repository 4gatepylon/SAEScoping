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

Sparsity validation
-------------------
PGDSFTTrainer includes two layers of validation to detect bugs early:

1.  Pre-training check (always runs regardless of ``validate_sparsity``):
    ``_validate_initial_sparsity`` is called once inside ``create_optimizer``
    and raises ``ValueError`` if any mask=False position in the model is
    currently non-zero.  This catches mismatched masks (e.g. masks built
    from a different checkpoint than the model) before any GPU time is wasted.

2.  Per-step check (runs when ``validate_sparsity=True``, the default):
    after every projected optimizer step ``_ProjectedStep`` asserts that
    all mask=False positions are still exactly 0.0.  The check is a single
    ``torch.any`` call per masked parameter — cheap enough to leave on
    during normal training.  Disable with ``validate_sparsity=False`` only
    if profiling shows it is a bottleneck.

Both checks assume that pruned positions carry the value *exactly* 0.0.
This is the natural invariant maintained by the pruning pipeline:
``prune_model`` multiplies weights by the keep-mask so pruned weights
become exactly 0.0, and ``_ProjectedStep`` restores that state after every
gradient update.

Note: clamping or bounding *non-zero* (kept) weights — e.g. projecting
onto a norm ball or clipping by value — is **not** currently supported.
It may be added in a future version if there is a need to enforce additional
constraints on surviving weights during recovery.

Public API
----------
PGDSFTTrainer            — drop-in replacement for SFTTrainer; accepts a
                           ``masks`` dict and applies the projection after
                           every optimiser step.
build_pgd_masks_from_model — convenience helper: derive a masks dict from the
                             current (already-pruned) model weights.
assert_masked_weights_are_zero — low-level assertion used by both validation
                                 layers; also importable for external checks.
"""

from __future__ import annotations

from typing import Optional

import torch
from trl import SFTTrainer


# ---------------------------------------------------------------------------
# Sparsity assertion — shared by pre-training and per-step validation
# ---------------------------------------------------------------------------


def assert_masked_weights_are_zero(
    param: torch.Tensor,
    mask: torch.Tensor,
    param_name: str = "?",
    error_type: type = ValueError,
) -> None:
    """Raise if any element of ``param`` at a mask=False position is non-zero.

    This is the core invariant check for PGD: every weight that was zeroed
    by pruning (mask=False) must stay exactly 0.0 throughout training.

    The check is intentionally *exact* (``!= 0``) rather than approximate:
    the pruning pipeline sets pruned weights to exactly 0.0 and
    ``_ProjectedStep`` maintains that invariant via exact assignment, so
    any non-zero value at a masked position is always a real bug.

    Args:
        param: The parameter tensor to inspect (may be on any device).
        mask: Boolean keep-mask of the same shape.  True = free weight,
              False = must be zero.
        param_name: Human-readable name used in the error message.
        error_type: Exception class to raise on violation.  Use
            ``ValueError`` for pre-training checks and ``RuntimeError``
            for mid-training checks so callers can distinguish them.

    Raises:
        error_type: If any ``param[~mask]`` element is non-zero, with a
            message reporting the parameter name, violation count, and the
            maximum absolute value found.
    """
    pruned_values = param.data[~mask]
    if pruned_values.any():
        n_violations = int((pruned_values != 0).sum().item())
        max_abs = float(pruned_values.abs().max().item())
        raise error_type(
            f"PGD sparsity violation in '{param_name}': "
            f"{n_violations} pruned position(s) are non-zero "
            f"(max |value| = {max_abs:.4g})."
        )


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
        4. If ``validate=True``, asserts that all mask=False positions are
           now exactly 0.0, raising ``RuntimeError`` on any violation.

    Implemented as a class (rather than a nested function) so that it can
    be tested independently and so all captured state is explicit.

    Args:
        original_step: The unpatched ``optimizer.step`` callable.
        optimizer: The optimiser whose ``param_groups`` we iterate over.
        masks_by_id: Mapping from ``id(param)`` to an on-device bool mask.
                     Populated once by
                     :meth:`PGDSFTTrainer._build_mask_id_map`.
        names_by_id: Mapping from ``id(param)`` to parameter name string,
                     used in validation error messages.
        validate: If ``True``, assert sparsity invariant after every step.
    """

    def __init__(
        self,
        original_step,
        optimizer: torch.optim.Optimizer,
        masks_by_id: dict[int, torch.Tensor],
        names_by_id: dict[int, str],
        validate: bool = True,
    ) -> None:
        self._original_step = original_step
        self._optimizer = optimizer
        self._masks_by_id = masks_by_id
        self._names_by_id = names_by_id
        self._validate = validate

    def __call__(self, closure=None):
        result = self._original_step(closure)
        with torch.no_grad():
            for group in self._optimizer.param_groups:
                for param in group["params"]:
                    mask = self._masks_by_id.get(id(param))
                    if mask is not None:
                        param.data[~mask] = 0.0
        if self._validate:
            for group in self._optimizer.param_groups:
                for param in group["params"]:
                    mask = self._masks_by_id.get(id(param))
                    if mask is not None:
                        assert_masked_weights_are_zero(
                            param, mask,
                            param_name=self._names_by_id.get(id(param), "?"),
                            error_type=RuntimeError,
                        )
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

    Two validation layers are built in (see module docstring for details):

    * **Pre-training** (always active): raises ``ValueError`` before the
      first step if any mask=False weight is currently non-zero.
    * **Per-step** (active when ``validate_sparsity=True``): raises
      ``RuntimeError`` after each projected step if any mask=False weight
      is non-zero.

    Parameters
    ----------
    masks:
        Dict mapping *parameter name* (as returned by
        ``model.named_parameters()``) to a boolean ``torch.Tensor`` of the
        same shape as that parameter.  ``True`` means the weight is free to
        change; ``False`` means it must stay at zero.  Tensors may live on
        any device — they are moved to each parameter's device the first time
        ``create_optimizer`` is called.
    validate_sparsity:
        If ``True`` (default), assert after every projected step that all
        mask=False positions are exactly 0.0.  Disable only if profiling
        shows this is a bottleneck (each check is a single ``torch.any``
        call per masked parameter, which is fast but non-zero).
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
        validate_sparsity: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._pgd_masks: dict[str, torch.Tensor] = masks
        self._validate_sparsity: bool = validate_sparsity
        # Populated lazily in create_optimizer once the model is on its final
        # device and the optimiser has been constructed.
        self._pgd_masks_by_param_id: dict[int, torch.Tensor] = {}
        self._pgd_names_by_param_id: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Optimiser creation — install the projection hook
    # ------------------------------------------------------------------

    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create the base optimiser and run pre-training validation.

        The projection hook is NOT installed here — it must be deferred
        until after the LR scheduler is created, because PyTorch's
        scheduler wraps ``optimizer.step`` and expects a bound method
        (i.e. one with ``__func__``).  See ``create_optimizer_and_scheduler``.
        """
        optimizer = super().create_optimizer()
        self._build_mask_id_map()
        self._validate_initial_sparsity()
        return optimizer

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        """Create optimizer + scheduler, then install the PGD projection hook.

        The projection hook must be installed *after* the LR scheduler is
        constructed because PyTorch's ``LambdaLR`` (and friends) wrap
        ``optimizer.step`` via ``__func__``, which fails on a plain callable.
        By deferring the hook to after ``super().create_optimizer_and_scheduler``
        we let the scheduler patch the real bound method first, and then our
        ``_ProjectedStep`` wraps the scheduler-patched version.
        """
        super().create_optimizer_and_scheduler(num_training_steps)
        self._install_projection_hook(self.optimizer)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_mask_id_map(self) -> None:
        """Build ``{id(param) → on-device mask}`` and ``{id(param) → name}`` tables.

        Keying by ``id(param)`` (rather than name) eliminates string-hash
        overhead from the hot path inside the patched ``optimizer.step``.
        Masks are moved to each parameter's device here, once at setup time.

        Raises:
            ValueError: If any mask's shape does not match its parameter.
        """
        self._pgd_masks_by_param_id = {}
        self._pgd_names_by_param_id = {}
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
            self._pgd_names_by_param_id[id(param)] = name

    def _validate_initial_sparsity(self) -> None:
        """Assert that every mask=False weight is currently exactly 0.0.

        Called once during ``create_optimizer``, before any training step.
        Raises ``ValueError`` immediately if the model weights do not match
        the supplied masks — for example when masks were built from a
        different checkpoint than the model that was passed in.

        This check always runs regardless of the ``validate_sparsity`` flag,
        because catching a misconfiguration before training starts is always
        worth the one-time cost of iterating over the masked parameters.

        Raises:
            ValueError: If any mask=False weight is non-zero.
        """
        for name, param in self.model.named_parameters():
            mask = self._pgd_masks.get(name)
            if mask is None:
                continue
            on_device_mask = mask.to(device=param.device, dtype=torch.bool)
            assert_masked_weights_are_zero(param, on_device_mask, name, ValueError)

    def _install_projection_hook(self, optimizer: torch.optim.Optimizer) -> None:
        """Replace ``optimizer.step`` with a :class:`_ProjectedStep` instance.

        After this call ``optimizer.step(...)`` will:
            1. Execute the original gradient update.
            2. Zero out all masked positions.
            3. (If ``validate_sparsity``) assert all masked positions are zero.

        Args:
            optimizer: The freshly constructed optimiser whose ``step``
                       attribute will be replaced in-place.
        """
        optimizer.step = _ProjectedStep(
            original_step=optimizer.step,
            optimizer=optimizer,
            masks_by_id=self._pgd_masks_by_param_id,
            names_by_id=self._pgd_names_by_param_id,
            validate=self._validate_sparsity,
        )
