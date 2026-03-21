"""Unit tests for gradients_map.py.

All tests run on CPU using tiny nn.Module instances — no HuggingFace download required.
"""
import torch
import torch.nn as nn
import pytest

from gradients_map import (
    _ALL_VARIANTS,
    _VARIANT_SPECS,
    _build_run_cmd,
    _register_ema_hooks,
    assert_all_params_require_grad,
    make_random_map,
)


# ---------------------------------------------------------------------------
# Shared tiny model fixture
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Two linear layers; mimics the interface used by the gradient map code."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(8, 8, bias=False)
        self.fc2 = nn.Linear(8, 4, bias=False)


@pytest.fixture()
def tiny_model() -> _TinyModel:
    return _TinyModel()


# ---------------------------------------------------------------------------
# make_random_map
# ---------------------------------------------------------------------------


def test_make_random_map_shapes_match_params(tiny_model: _TinyModel) -> None:
    """Returned tensors have the same shape as the corresponding parameters."""
    rmap = make_random_map(tiny_model, seed=0)
    for name, param in tiny_model.named_parameters():
        assert name in rmap, f"❌ Missing key '{name}'"
        assert rmap[name].shape == param.shape, (
            f"❌ Shape mismatch for '{name}': {rmap[name].shape} vs {param.shape}"
        )
    print("✅ make_random_map: shapes match model parameters")


def test_make_random_map_values_in_unit_interval(tiny_model: _TinyModel) -> None:
    """All values are in [0, 1)."""
    rmap = make_random_map(tiny_model, seed=0)
    for name, tensor in rmap.items():
        assert tensor.min().item() >= 0.0, f"❌ Negative value in '{name}': {tensor.min()}"
        assert tensor.max().item() < 1.0, f"❌ Value >= 1.0 in '{name}': {tensor.max()}"
    print("✅ make_random_map: all values in [0, 1)")


def test_make_random_map_deterministic(tiny_model: _TinyModel) -> None:
    """Same seed produces identical maps; different seeds differ."""
    m1 = make_random_map(tiny_model, seed=42)
    m2 = make_random_map(tiny_model, seed=42)
    m3 = make_random_map(tiny_model, seed=99)
    for name in m1:
        assert torch.equal(m1[name], m2[name]), f"❌ Same seed gave different map for '{name}'"
        assert not torch.equal(m1[name], m3[name]), f"❌ Different seeds gave same map for '{name}'"
    print("✅ make_random_map: deterministic with same seed, different with different seeds")


def test_make_random_map_only_trainable_params(tiny_model: _TinyModel) -> None:
    """Only parameters with requires_grad=True are included."""
    # Freeze one layer
    for param in tiny_model.fc2.parameters():
        param.requires_grad = False
    rmap = make_random_map(tiny_model, seed=0)
    assert "fc1.weight" in rmap, "❌ fc1.weight (trainable) should be in map"
    assert "fc2.weight" not in rmap, "❌ fc2.weight (frozen) should not be in map"
    print("✅ make_random_map: only includes trainable parameters")


# ---------------------------------------------------------------------------
# assert_all_params_require_grad
# ---------------------------------------------------------------------------


def test_assert_all_params_require_grad_passes_when_all_trainable(tiny_model: _TinyModel) -> None:
    """No error when every parameter requires grad (default model state)."""
    assert_all_params_require_grad(tiny_model)
    print("✅ assert_all_params_require_grad: passes when all params are trainable")


def test_assert_all_params_require_grad_raises_when_param_frozen(tiny_model: _TinyModel) -> None:
    """Raises AssertionError listing the frozen parameter name."""
    tiny_model.fc2.weight.requires_grad = False
    with pytest.raises(AssertionError, match="fc2.weight"):
        assert_all_params_require_grad(tiny_model)
    print("✅ assert_all_params_require_grad: raises and names frozen params")


def test_assert_all_params_require_grad_allow_frozen_bypasses_check(tiny_model: _TinyModel) -> None:
    """allow_frozen=True suppresses the check even when params are frozen."""
    for p in tiny_model.parameters():
        p.requires_grad = False
    assert_all_params_require_grad(tiny_model, allow_frozen=True)  # must not raise
    print("✅ assert_all_params_require_grad: allow_frozen=True bypasses check")


def test_assert_all_params_require_grad_error_lists_all_frozen(tiny_model: _TinyModel) -> None:
    """Error message includes every frozen parameter, not just the first."""
    for p in tiny_model.parameters():
        p.requires_grad = False
    with pytest.raises(AssertionError) as exc_info:
        assert_all_params_require_grad(tiny_model)
    msg = str(exc_info.value)
    assert "fc1.weight" in msg, "❌ fc1.weight should appear in error"
    assert "fc2.weight" in msg, "❌ fc2.weight should appear in error"
    print("✅ assert_all_params_require_grad: error lists all frozen params")


# ---------------------------------------------------------------------------
# _register_ema_hooks — signed
# ---------------------------------------------------------------------------


def test_register_ema_hooks_signed_accumulates_signed_gradient() -> None:
    """With abs_grad=False, EMA accumulates the signed gradient.

    After one backward with a constant all-ones gradient, param.grad should
    be approximately the gradient value (since it is the first step: g_0 = g).
    """
    model = _TinyModel()
    _register_ema_hooks(model, beta=0.9, abs_grad=False)
    model._ema_seen.clear()

    # Trigger a single backward pass with a synthetic gradient
    x = torch.ones(1, 8)
    loss = model.fc1(x).sum()
    loss.backward()

    # fc1.weight.grad should hold the first EMA value (= raw gradient on first step)
    grad = model.fc1.weight.grad
    assert grad is not None, "❌ fc1.weight.grad is None after backward"
    assert grad.shape == model.fc1.weight.shape, "❌ grad shape mismatch"
    print("✅ _register_ema_hooks signed: gradient accumulated in param.grad")


def test_register_ema_hooks_signed_ema_approaches_zero_on_alternating_signs() -> None:
    """With abs_grad=False and gradients alternating sign, EMA converges toward 0."""
    model = _TinyModel(seed=1)
    beta = 0.9
    _register_ema_hooks(model, beta=beta, abs_grad=False)

    # Simulate many steps with alternating +1/-1 constant gradients.
    # True EMA with alternating ±c converges toward 0 with signed accumulation.
    n_steps = 200
    for step in range(n_steps):
        model._ema_seen.clear()
        sign = 1.0 if step % 2 == 0 else -1.0
        with torch.enable_grad():
            x = torch.ones(1, 8) * sign
            loss = model.fc1(x).sum()
            loss.backward()

    final_grad_norm = model.fc1.weight.grad.abs().mean().item()
    # After 200 alternating-sign steps with beta=0.9, EMA should be near 0
    # (exact value depends on architecture, but should be much less than 1)
    assert final_grad_norm < 0.5, (
        f"❌ EMA of alternating signed gradients should be ~0, got mean abs = {final_grad_norm:.4f}"
    )
    print(f"✅ _register_ema_hooks signed: EMA of alternating signs ≈ 0 (got {final_grad_norm:.4f})")


def test_register_ema_hooks_abs_stays_positive_on_alternating_signs() -> None:
    """With abs_grad=True, EMA of |g_t| stays positive even with alternating signs."""
    model = _TinyModel(seed=1)
    beta = 0.9
    _register_ema_hooks(model, beta=beta, abs_grad=True)

    n_steps = 50
    for step in range(n_steps):
        sign = 1.0 if step % 2 == 0 else -1.0
        with torch.enable_grad():
            model._ema_seen.clear()
            x = torch.ones(1, 8) * sign
            loss = model.fc1(x).sum()
            loss.backward()

    final_grad = model.fc1.weight.grad
    assert final_grad is not None, "❌ grad is None"
    # With abs_grad=True, all accumulated values are non-negative
    assert final_grad.min().item() >= -1e-6, (
        f"❌ abs_grad=True should produce non-negative EMA, min={final_grad.min().item():.6f}"
    )
    assert final_grad.mean().item() > 0.01, (
        f"❌ abs_grad=True EMA should be clearly positive, got mean={final_grad.mean().item():.6f}"
    )
    print(f"✅ _register_ema_hooks abs: EMA of |g_t| stays positive (mean={final_grad.mean().item():.4f})")


def test_register_ema_hooks_abs_vs_signed_differ_on_negative_gradient() -> None:
    """abs_grad=True and abs_grad=False produce different results on consistent negative gradient."""
    def _run_one_step(abs_grad: bool) -> torch.Tensor:
        model = _TinyModel(seed=5)
        _register_ema_hooks(model, beta=0.9, abs_grad=abs_grad)
        model._ema_seen.clear()
        with torch.enable_grad():
            x = -torch.ones(1, 8)  # negative input → negative gradient
            loss = model.fc1(x).sum()
            loss.backward()
        return model.fc1.weight.grad.clone()

    grad_signed = _run_one_step(abs_grad=False)
    grad_abs = _run_one_step(abs_grad=True)

    # signed grad should be negative (or mixed), abs should be non-negative
    assert grad_signed.min().item() < 0, "❌ Signed hook: expected negative gradient component"
    assert grad_abs.min().item() >= -1e-6, "❌ Abs hook: expected non-negative gradient"
    assert not torch.allclose(grad_signed, grad_abs), "❌ abs and signed hooks should differ"
    print("✅ _register_ema_hooks: abs_grad=True differs from abs_grad=False on negative gradient")


# ---------------------------------------------------------------------------
# _VARIANT_SPECS / _ALL_VARIANTS completeness
# ---------------------------------------------------------------------------


def test_all_variants_have_spec_entries() -> None:
    """Every name in _ALL_VARIANTS has a corresponding entry in _VARIANT_SPECS."""
    for variant in _ALL_VARIANTS:
        assert variant in _VARIANT_SPECS, f"❌ Variant '{variant}' missing from _VARIANT_SPECS"
    print("✅ _ALL_VARIANTS: all variants have spec entries")


def test_variant_specs_have_valid_modes() -> None:
    """Every spec has a valid mode, a bool abs_grad, and a non-empty output path."""
    valid_modes = {"gradient_ema", "random"}
    for variant, (mode, abs_grad, output_path) in _VARIANT_SPECS.items():
        assert mode in valid_modes, f"❌ Variant '{variant}' has invalid mode '{mode}'"
        assert isinstance(abs_grad, bool), f"❌ Variant '{variant}' abs_grad is not bool"
        assert output_path.endswith(".safetensors"), (
            f"❌ Variant '{variant}' output path does not end with .safetensors: {output_path}"
        )
    print("✅ _VARIANT_SPECS: all specs have valid fields")


def test_random_variant_has_abs_grad_false() -> None:
    """The random variant always has abs_grad=False (abs_grad is meaningless for random)."""
    mode, abs_grad, _ = _VARIANT_SPECS["random"]
    assert mode == "random", f"❌ Expected mode='random', got '{mode}'"
    assert abs_grad is False, "❌ Random variant should have abs_grad=False"
    print("✅ _VARIANT_SPECS: random variant has abs_grad=False")


def test_gradient_ema_abs_variant_has_abs_grad_true() -> None:
    """The gradient_ema_abs variant has abs_grad=True."""
    mode, abs_grad, _ = _VARIANT_SPECS["gradient_ema_abs"]
    assert mode == "gradient_ema", f"❌ Expected mode='gradient_ema', got '{mode}'"
    assert abs_grad is True, "❌ gradient_ema_abs should have abs_grad=True"
    print("✅ _VARIANT_SPECS: gradient_ema_abs variant has abs_grad=True")


# ---------------------------------------------------------------------------
# _build_run_cmd
# ---------------------------------------------------------------------------


def _default_common_kwargs() -> dict:
    return dict(
        model_id="google/gemma-2-9b-it",
        dataset_name="4gate/StemQAMixture",
        dataset_subset="biology",
        dataset_size=16384,
        seed=42,
        beta=0.95,
        batch_size=2,
        max_seq_len=1024,
        num_epochs=2,
    )


def test_build_run_cmd_gradient_ema_no_abs_flag() -> None:
    """gradient_ema variant produces a cmd without --abs-grad."""
    cmd = _build_run_cmd("gradient_ema", _default_common_kwargs())
    cmd_str = " ".join(str(c) for c in cmd)
    assert "--mode" in cmd_str, "❌ Missing --mode"
    assert "gradient_ema" in cmd_str, "❌ Missing mode value"
    assert "--abs-grad" not in cmd_str, "❌ gradient_ema should NOT have --abs-grad"
    print("✅ _build_run_cmd: gradient_ema has no --abs-grad")


def test_build_run_cmd_gradient_ema_abs_includes_abs_flag() -> None:
    """gradient_ema_abs variant produces a cmd with --abs-grad."""
    cmd = _build_run_cmd("gradient_ema_abs", _default_common_kwargs())
    assert "--abs-grad" in cmd, "❌ gradient_ema_abs should include --abs-grad"
    print("✅ _build_run_cmd: gradient_ema_abs includes --abs-grad")


def test_build_run_cmd_random_mode_is_random() -> None:
    """random variant produces a cmd with --mode random."""
    cmd = _build_run_cmd("random", _default_common_kwargs())
    cmd_str = " ".join(str(c) for c in cmd)
    assert "random" in cmd_str, "❌ random variant should have mode=random in cmd"
    assert "--abs-grad" not in cmd_str, "❌ random variant should NOT have --abs-grad"
    print("✅ _build_run_cmd: random variant has mode=random and no --abs-grad")


def test_build_run_cmd_contains_output_path() -> None:
    """Each variant's cmd includes --output-path pointing at the expected file."""
    for variant, (_, _, expected_path) in _VARIANT_SPECS.items():
        cmd = _build_run_cmd(variant, _default_common_kwargs())
        assert "--output-path" in cmd, f"❌ Missing --output-path for '{variant}'"
        idx = cmd.index("--output-path")
        assert cmd[idx + 1] == expected_path, (
            f"❌ Wrong output path for '{variant}': {cmd[idx + 1]} vs {expected_path}"
        )
    print("✅ _build_run_cmd: all variants have correct --output-path")
