"""
Unit tests for pgd_trainer.py

All tests run on CPU only (no GPU required) and make no network calls.

Coverage:
- assert_masked_weights_are_zero : core assertion, error type, message content
- build_pgd_masks_from_model     : correctness, CPU placement, param_names filter
- _ProjectedStep                 : projection correctness, ordering, no-op cases,
                                   per-step validation (validate=True/False)
- PGDSFTTrainer._build_mask_id_map      : device placement, dtype, shape mismatch,
                                          unknown names, id keying
- PGDSFTTrainer._validate_initial_sparsity : pre-training check raises on violation
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from sae_scoping.training.pgd_trainer import (
    PGDSFTTrainer,
    _ProjectedStep,
    assert_masked_weights_are_zero,
    build_pgd_masks_from_model,
)


# ---------------------------------------------------------------------------
# Shared tiny-model helpers
# ---------------------------------------------------------------------------


def _make_linear_model() -> nn.Sequential:
    """2-layer linear model (no bias) with random init, always on CPU."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(4, 4, bias=False),
        nn.Linear(4, 2, bias=False),
    )


def _zero_first_half(model: nn.Module) -> nn.Module:
    """Zero out the first half of every parameter's flat view."""
    with torch.no_grad():
        for param in model.parameters():
            flat = param.data.view(-1)
            flat[: flat.numel() // 2] = 0.0
    return model


# ---------------------------------------------------------------------------
# Duck-type trainer for testing _build_mask_id_map without full SFTTrainer
# ---------------------------------------------------------------------------


class _FakePGDTrainer:
    """Minimal object satisfying _build_mask_id_map's attribute requirements."""

    def __init__(self, model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
        self.model = model
        self._pgd_masks = masks
        self._pgd_masks_by_param_id: dict[int, torch.Tensor] = {}


# ---------------------------------------------------------------------------
# TestBuildPGDMasksFromModel
# ---------------------------------------------------------------------------


class TestBuildPGDMasksFromModel:
    def test_all_nonzero_model_returns_empty_dict(self):
        model = _make_linear_model()
        with torch.no_grad():
            for p in model.parameters():
                p.data.fill_(1.0)
        masks = build_pgd_masks_from_model(model)
        assert masks == {}, "❌ fully-dense model should yield empty mask dict"
        print("✅ all-nonzero model returns empty dict")

    def test_partially_zeroed_model_correct_masks(self):
        model = _zero_first_half(_make_linear_model())
        masks = build_pgd_masks_from_model(model)
        for name, param in model.named_parameters():
            if name in masks:
                expected = (param.data != 0).cpu()
                assert torch.equal(masks[name], expected), (
                    f"❌ {name}: mask != (param != 0)"
                )
        print("✅ masks equal (param != 0) for each parameter")

    def test_fully_zeroed_param_all_false_mask(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        with torch.no_grad():
            model[0].weight.data.zero_()
        masks = build_pgd_masks_from_model(model)
        assert "0.weight" in masks, "❌ fully-zeroed param should appear in masks"
        assert not masks["0.weight"].any(), (
            "❌ fully-zeroed param should produce an all-False mask"
        )
        print("✅ fully-zeroed param produces all-False mask")

    def test_masks_are_on_cpu(self):
        model = _zero_first_half(_make_linear_model())
        masks = build_pgd_masks_from_model(model)
        for name, mask in masks.items():
            assert mask.device == torch.device("cpu"), f"❌ {name}: mask not on CPU"
        print("✅ all masks live on CPU")

    def test_masks_dtype_is_bool(self):
        model = _zero_first_half(_make_linear_model())
        masks = build_pgd_masks_from_model(model)
        for name, mask in masks.items():
            assert mask.dtype == torch.bool, f"❌ {name}: mask dtype {mask.dtype}"
        print("✅ all masks have dtype=bool")

    def test_mask_shape_matches_param_shape(self):
        model = _zero_first_half(_make_linear_model())
        masks = build_pgd_masks_from_model(model)
        for name, param in model.named_parameters():
            if name in masks:
                assert masks[name].shape == param.shape, (
                    f"❌ {name}: mask {masks[name].shape} != param {param.shape}"
                )
        print("✅ each mask shape matches its parameter shape")

    def test_param_names_filter_excludes_unlisted(self):
        model = _zero_first_half(_make_linear_model())
        first_name = next(n for n, _ in model.named_parameters())
        masks = build_pgd_masks_from_model(model, param_names=[first_name])
        for name in masks:
            assert name == first_name, f"❌ {name} was not in the allowlist"
        print("✅ param_names filter restricts masks to listed params only")

    def test_param_names_empty_list_returns_empty_dict(self):
        model = _zero_first_half(_make_linear_model())
        masks = build_pgd_masks_from_model(model, param_names=[])
        assert masks == {}, "❌ empty param_names list should return empty dict"
        print("✅ empty param_names list returns empty dict")

    def test_dense_param_excluded_even_if_listed_in_param_names(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        with torch.no_grad():
            model[0].weight.data.fill_(1.0)
        masks = build_pgd_masks_from_model(model, param_names=["0.weight"])
        assert masks == {}, (
            "❌ dense parameter should be excluded even when listed in param_names"
        )
        print("✅ dense parameter excluded even when explicitly listed in param_names")

    def test_true_where_nonzero_false_where_zero(self):
        """Spot-check: first half False, second half True."""
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        with torch.no_grad():
            param.data.view(-1)[:8] = 0.0   # first 8 zeroed
            param.data.view(-1)[8:] = 1.0   # last 8 non-zero
        masks = build_pgd_masks_from_model(model)
        mask = masks["0.weight"].view(-1)
        assert not mask[:8].any(), "❌ zeroed positions should be False in mask"
        assert mask[8:].all(), "❌ non-zero positions should be True in mask"
        print("✅ mask is False at zero positions and True at non-zero positions")


# ---------------------------------------------------------------------------
# TestProjectedStep
# ---------------------------------------------------------------------------


class TestProjectedStep:
    """Tests for _ProjectedStep without going through SFTTrainer."""

    def _make_optimizer_and_projected(
        self,
        model: nn.Module,
        masks_by_id: dict[int, torch.Tensor],
        lr: float = 0.5,
        validate: bool = False,
    ) -> torch.optim.Optimizer:
        """Return an SGD optimizer whose step is wrapped with _ProjectedStep."""
        names_by_id = {pid: f"param_{pid}" for pid in masks_by_id}
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer.step = _ProjectedStep(
            original_step=optimizer.step,
            optimizer=optimizer,
            masks_by_id=masks_by_id,
            names_by_id=names_by_id,
            validate=validate,
        )
        return optimizer

    def _backward(self, model: nn.Module) -> None:
        """One forward+backward pass with a fixed random input."""
        torch.manual_seed(99)
        x = torch.randn(2, 4)
        model(x).sum().backward()

    def test_masked_zero_positions_remain_zero(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        with torch.no_grad():
            param.data[0, 0] = 0.0   # the one explicitly zeroed element
        mask = (param.data != 0).clone()
        optimizer = self._make_optimizer_and_projected(model, {id(param): mask})

        for step in range(5):
            optimizer.zero_grad()
            self._backward(model)
            optimizer.step()
            assert param.data[0, 0].item() == 0.0, (
                f"❌ masked-zero position [0,0] drifted non-zero at step {step + 1}"
            )
        print("✅ masked-zero positions stay zero across 5 optimizer steps")

    def test_unmasked_positions_are_updated_by_optimizer(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        initial = param.data.clone()
        # All-True mask → projection is a no-op; weights should still change
        mask = torch.ones_like(param.data, dtype=torch.bool)
        optimizer = self._make_optimizer_and_projected(model, {id(param): mask})
        optimizer.zero_grad()
        self._backward(model)
        optimizer.step()
        assert not torch.equal(param.data, initial), (
            "❌ unmasked weights should be updated by the optimizer"
        )
        print("✅ unmasked positions are updated by the optimizer step")

    def test_all_true_mask_does_not_zero_out_weights(self):
        """All-True mask → projection is identity; weights that were zero may move."""
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        with torch.no_grad():
            param.data.zero_()
        mask = torch.ones_like(param.data, dtype=torch.bool)
        optimizer = self._make_optimizer_and_projected(model, {id(param): mask})
        optimizer.zero_grad()
        self._backward(model)
        optimizer.step()
        assert not (param.data == 0).all(), (
            "❌ all-True mask must not force weights back to zero"
        )
        print("✅ all-True mask is a no-op — weights are not forced to zero")

    def test_empty_masks_dict_does_not_block_updates(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        initial = param.data.clone()
        optimizer = self._make_optimizer_and_projected(model, masks_by_id={})
        optimizer.zero_grad()
        self._backward(model)
        optimizer.step()
        assert not torch.equal(param.data, initial), (
            "❌ empty masks dict should not prevent optimizer updates"
        )
        print("✅ empty masks dict is a no-op on projection; weights update freely")

    def test_original_step_is_invoked(self):
        """Verify the original optimizer step is actually called (weights change)."""
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        initial = param.data.clone()
        optimizer = self._make_optimizer_and_projected(model, masks_by_id={})
        optimizer.zero_grad()
        self._backward(model)
        optimizer.step()
        assert not torch.equal(param.data, initial), (
            "❌ original optimizer step was not invoked"
        )
        print("✅ original optimizer step is invoked before projection")

    def test_multi_param_model_all_masked_zeros_enforced(self):
        model = _zero_first_half(_make_linear_model())
        params = list(model.parameters())
        masks_by_id = {id(p): (p.data != 0).clone() for p in params}
        optimizer = self._make_optimizer_and_projected(model, masks_by_id)

        for _ in range(5):
            optimizer.zero_grad()
            torch.manual_seed(7)
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()

        for i, param in enumerate(params):
            flat = param.data.view(-1)
            n = flat.numel()
            zeroed = flat[: n // 2]
            assert (zeroed == 0).all(), (
                f"❌ param[{i}]: first half (masked zeros) drifted non-zero"
            )
        print("✅ all masked-zero positions enforced across 5 steps in 2-layer model")

    def test_projection_happens_after_gradient_update(self):
        """
        Projection should fire AFTER the optimizer updates weights, not before.
        Concretely: if a masked position starts at zero, the optimizer may
        temporarily write a non-zero value, but the projection zeroes it back.
        We verify the net effect (weight = 0) rather than the intermediate state.
        """
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        with torch.no_grad():
            param.data.zero_()
        mask = torch.zeros_like(param.data, dtype=torch.bool)  # all False = all locked
        optimizer = self._make_optimizer_and_projected(model, {id(param): mask})
        optimizer.zero_grad()
        self._backward(model)
        optimizer.step()
        assert (param.data == 0).all(), (
            "❌ all-False mask should leave all weights at zero after the projected step"
        )
        print("✅ projection fires after gradient update, restoring all-zero weights")


# ---------------------------------------------------------------------------
# TestBuildMaskIdMap
# ---------------------------------------------------------------------------


class TestBuildMaskIdMap:
    """Tests for PGDSFTTrainer._build_mask_id_map via _FakePGDTrainer."""

    def test_shape_mismatch_raises_value_error(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        wrong_mask = torch.ones(2, 2, dtype=torch.bool)  # (2,2) vs (4,4)
        fake = _FakePGDTrainer(model, {"0.weight": wrong_mask})
        with pytest.raises(ValueError, match="shape"):
            PGDSFTTrainer._build_mask_id_map(fake)
        print("✅ shape mismatch raises ValueError mentioning 'shape'")

    def test_unknown_param_name_silently_ignored(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        fake = _FakePGDTrainer(model, {"does_not_exist.weight": torch.ones(4, 4, dtype=torch.bool)})
        PGDSFTTrainer._build_mask_id_map(fake)
        assert fake._pgd_masks_by_param_id == {}, (
            "❌ unknown param name should produce empty id map"
        )
        print("✅ unknown parameter name is silently ignored")

    def test_keyed_by_param_id(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        fake = _FakePGDTrainer(model, {"0.weight": torch.zeros(4, 4, dtype=torch.bool)})
        PGDSFTTrainer._build_mask_id_map(fake)
        assert id(param) in fake._pgd_masks_by_param_id, (
            "❌ _pgd_masks_by_param_id must be keyed by id(param)"
        )
        print("✅ lookup table is keyed by id(param)")

    def test_mask_moved_to_param_device(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        cpu_mask = torch.ones(4, 4, dtype=torch.bool)
        fake = _FakePGDTrainer(model, {"0.weight": cpu_mask})
        PGDSFTTrainer._build_mask_id_map(fake)
        stored = fake._pgd_masks_by_param_id[id(param)]
        assert stored.device == param.device, (
            f"❌ stored mask device {stored.device} != param device {param.device}"
        )
        print(f"✅ mask moved to param device ({param.device})")

    def test_stored_mask_dtype_is_bool(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        float_mask = torch.ones(4, 4, dtype=torch.float32)
        fake = _FakePGDTrainer(model, {"0.weight": float_mask})
        PGDSFTTrainer._build_mask_id_map(fake)
        stored = fake._pgd_masks_by_param_id[id(param)]
        assert stored.dtype == torch.bool, f"❌ stored mask dtype {stored.dtype}"
        print("✅ stored mask dtype is bool regardless of input dtype")

    def test_empty_masks_produces_empty_id_map(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        fake = _FakePGDTrainer(model, {})
        PGDSFTTrainer._build_mask_id_map(fake)
        assert fake._pgd_masks_by_param_id == {}
        print("✅ empty masks dict produces empty id map")

    def test_multiple_params_all_registered(self):
        model = _make_linear_model()
        masks = {
            name: torch.zeros(param.shape, dtype=torch.bool)
            for name, param in model.named_parameters()
        }
        fake = _FakePGDTrainer(model, masks)
        PGDSFTTrainer._build_mask_id_map(fake)
        param_ids = {id(p) for _, p in model.named_parameters()}
        assert set(fake._pgd_masks_by_param_id.keys()) == param_ids, (
            "❌ not all parameter ids were registered in the id map"
        )
        print("✅ all parameter ids registered when all params are in masks dict")


# ---------------------------------------------------------------------------
# TestAssertMaskedWeightsAreZero
# ---------------------------------------------------------------------------


class TestAssertMaskedWeightsAreZero:
    def test_all_zero_raises_nothing(self):
        param = torch.zeros(4, 4)
        mask = torch.zeros(4, 4, dtype=torch.bool)  # all pruned
        assert_masked_weights_are_zero(param, mask)
        print("✅ all-zero param with all-False mask raises nothing")

    def test_dense_mask_raises_nothing(self):
        param = torch.randn(4, 4)
        mask = torch.ones(4, 4, dtype=torch.bool)  # all free — nothing to check
        assert_masked_weights_are_zero(param, mask)
        print("✅ all-True mask (no pruned positions) raises nothing")

    def test_nonzero_at_masked_position_raises(self):
        param = torch.zeros(4, 4)
        param[0, 0] = 1.0  # violation
        mask = torch.zeros(4, 4, dtype=torch.bool)
        with pytest.raises(ValueError):
            assert_masked_weights_are_zero(param, mask, param_name="fc.weight")
        print("✅ non-zero value at mask=False position raises ValueError")

    def test_error_message_contains_param_name(self):
        param = torch.zeros(4)
        param[1] = 0.5
        mask = torch.zeros(4, dtype=torch.bool)
        with pytest.raises(ValueError, match="my_special_param"):
            assert_masked_weights_are_zero(param, mask, param_name="my_special_param")
        print("✅ error message contains the parameter name")

    def test_error_message_contains_violation_count(self):
        param = torch.tensor([1.0, 2.0, 0.0, 0.0])
        mask = torch.zeros(4, dtype=torch.bool)  # all pruned; 2 violations
        with pytest.raises(ValueError, match="2"):
            assert_masked_weights_are_zero(param, mask)
        print("✅ error message contains the number of violations")

    def test_custom_error_type_is_raised(self):
        param = torch.tensor([1.0])
        mask = torch.zeros(1, dtype=torch.bool)
        with pytest.raises(RuntimeError):
            assert_masked_weights_are_zero(param, mask, error_type=RuntimeError)
        print("✅ custom error_type is used when provided")

    def test_exactly_zero_is_not_a_violation(self):
        param = torch.tensor([0.0, 1.0, 0.0])
        # Only position 1 is free; positions 0 and 2 are pruned and are zero
        mask = torch.tensor([False, True, False])
        assert_masked_weights_are_zero(param, mask)
        print("✅ exactly-zero values at mask=False positions pass cleanly")


# ---------------------------------------------------------------------------
# TestProjectedStepValidation
# ---------------------------------------------------------------------------


class TestProjectedStepValidation:
    """Tests for the per-step validation path in _ProjectedStep."""

    def _make_optimizer_and_projected(
        self,
        model: nn.Module,
        masks_by_id: dict,
        names_by_id: dict,
        validate: bool,
        lr: float = 0.5,
    ) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer.step = _ProjectedStep(
            original_step=optimizer.step,
            optimizer=optimizer,
            masks_by_id=masks_by_id,
            names_by_id=names_by_id,
            validate=validate,
        )
        return optimizer

    def test_validate_true_calls_assert_with_runtime_error_type(self):
        """
        Verify that validate=True calls assert_masked_weights_are_zero with
        error_type=RuntimeError (not ValueError, which is the pre-training type).
        We use mock.patch so the assertion fires regardless of the actual weight
        values, and we inspect the call arguments.
        """
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        with torch.no_grad():
            param.data.zero_()

        mask = torch.zeros(4, 4, dtype=torch.bool)
        masks_by_id = {id(param): mask}
        names_by_id = {id(param): "0.weight"}

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step = _ProjectedStep(
            original_step=optimizer.step,
            optimizer=optimizer,
            masks_by_id=masks_by_id,
            names_by_id=names_by_id,
            validate=True,
        )
        optimizer.zero_grad()
        torch.manual_seed(42)
        model(torch.randn(2, 4)).sum().backward()

        with patch("pgd_trainer.assert_masked_weights_are_zero") as mock_assert:
            optimizer.step()
            assert mock_assert.called, "❌ assert_masked_weights_are_zero was not called"
            call_kwargs = mock_assert.call_args
            used_error_type = call_kwargs.kwargs.get(
                "error_type", call_kwargs.args[3] if len(call_kwargs.args) > 3 else None
            )
            assert used_error_type is RuntimeError, (
                f"❌ per-step validation should use RuntimeError, got {used_error_type}"
            )
        print("✅ validate=True calls assert_masked_weights_are_zero with error_type=RuntimeError")

    def test_validate_false_does_not_raise_even_with_violation(self):
        """validate=False skips the check; no error even if a weight is bad."""
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        with torch.no_grad():
            param.data.zero_()
        mask = torch.zeros(4, 4, dtype=torch.bool)
        masks_by_id = {id(param): mask}
        names_by_id = {id(param): "0.weight"}

        optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

        def bad_original_step(closure=None):
            optimizer.__class__.step(optimizer, closure)
            with torch.no_grad():
                param.data[0, 0] = 99.0
            return None

        optimizer.step = _ProjectedStep(
            original_step=bad_original_step,
            optimizer=optimizer,
            masks_by_id=masks_by_id,
            names_by_id=names_by_id,
            validate=False,
        )
        optimizer.zero_grad()
        torch.manual_seed(42)
        model(torch.randn(2, 4)).sum().backward()
        # Should not raise even though position [0,0] ends up non-zero (bug)
        optimizer.step()
        print("✅ validate=False skips the check — no error even with a violation")

    def test_validate_true_passes_cleanly_after_correct_projection(self):
        """Normal operation: projection zeroes weights → validation passes."""
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        with torch.no_grad():
            param.data[0, :] = 0.0  # first row pruned
        mask = (param.data != 0).clone()
        masks_by_id = {id(param): mask}
        names_by_id = {id(param): "0.weight"}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step = _ProjectedStep(
            original_step=optimizer.step,
            optimizer=optimizer,
            masks_by_id=masks_by_id,
            names_by_id=names_by_id,
            validate=True,
        )
        for _ in range(3):
            optimizer.zero_grad()
            torch.manual_seed(7)
            model(torch.randn(2, 4)).sum().backward()
            optimizer.step()  # must not raise
        print("✅ validate=True passes cleanly across 3 steps of correct PGD")


# ---------------------------------------------------------------------------
# TestValidateInitialSparsity
# ---------------------------------------------------------------------------


class TestValidateInitialSparsity:
    """Tests for PGDSFTTrainer._validate_initial_sparsity via duck-type."""

    def test_raises_value_error_if_masked_weight_is_nonzero(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        with torch.no_grad():
            param.data.fill_(1.0)  # all non-zero
        # Mask says all positions are pruned → every weight violates invariant
        mask = torch.zeros(4, 4, dtype=torch.bool)
        fake = _FakePGDTrainer(model, {"0.weight": mask})
        with pytest.raises(ValueError, match="sparsity violation"):
            PGDSFTTrainer._validate_initial_sparsity(fake)
        print("✅ _validate_initial_sparsity raises ValueError on non-zero masked weight")

    def test_passes_when_all_masked_weights_are_zero(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        with torch.no_grad():
            param.data.zero_()
        mask = torch.zeros(4, 4, dtype=torch.bool)  # all pruned, all zero
        fake = _FakePGDTrainer(model, {"0.weight": mask})
        PGDSFTTrainer._validate_initial_sparsity(fake)  # must not raise
        print("✅ _validate_initial_sparsity passes when all masked weights are zero")

    def test_passes_with_mixed_mask_and_correct_weights(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        param = model[0].weight
        with torch.no_grad():
            param.data.fill_(1.0)
            param.data[0, :] = 0.0  # first row pruned and zero
        mask = torch.ones(4, 4, dtype=torch.bool)
        mask[0, :] = False  # first row is pruned
        fake = _FakePGDTrainer(model, {"0.weight": mask})
        PGDSFTTrainer._validate_initial_sparsity(fake)  # must not raise
        print("✅ _validate_initial_sparsity passes when only mask=False rows are zero")

    def test_error_message_includes_param_name(self):
        model = nn.Sequential(nn.Linear(4, 4, bias=False))
        with torch.no_grad():
            model[0].weight.data.fill_(1.0)
        mask = torch.zeros(4, 4, dtype=torch.bool)
        fake = _FakePGDTrainer(model, {"0.weight": mask})
        with pytest.raises(ValueError, match="0.weight"):
            PGDSFTTrainer._validate_initial_sparsity(fake)
        print("✅ _validate_initial_sparsity error message includes the parameter name")

    def test_skips_params_not_in_masks(self):
        """Parameters absent from the masks dict are not checked."""
        model = _make_linear_model()
        # Only mask the first parameter; second has non-zero weights but is unchecked
        first_name = next(n for n, _ in model.named_parameters())
        first_param = dict(model.named_parameters())[first_name]
        with torch.no_grad():
            first_param.data.zero_()
        mask = torch.zeros_like(first_param.data, dtype=torch.bool)
        fake = _FakePGDTrainer(model, {first_name: mask})
        PGDSFTTrainer._validate_initial_sparsity(fake)  # must not raise
        print("✅ _validate_initial_sparsity only checks parameters present in masks dict")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
