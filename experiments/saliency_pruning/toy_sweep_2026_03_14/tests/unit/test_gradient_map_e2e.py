"""End-to-end unit tests for gradient-map computation (_register_ema_hooks).

All tests run on CPU with randomly initialised HuggingFace models created from
config (no weight downloads required).  Each test verifies a distinct property
of the EMA gradient accumulation algorithm.

Tests
-----
test_ema_hooks_weights_invariant_and_grads_populated  [qwen2, llama]
    EMA hooks must not mutate model weights; after N backward steps every
    trainable parameter must carry a non-None grad tensor.

test_ema_hooks_abs_vs_signed  [qwen2, llama]
    Running abs-mode EMA (abs_grad=True) must produce non-negative values for
    every parameter and differ from the signed EMA on the same data.

test_gradient_ranking_dead_params_get_zero_saliency
    A parameter that is never part of any forward computation graph accumulates
    zero gradient signal.  The saliency map must reflect this.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from gradients_map.grad import _register_ema_hooks
from tests.unit.model_factories import TINY_HF_FACTORIES, run_single_synthetic_step
from tests.unit.validators import (
    assert_grad_maps_differ,
    assert_gradient_map_covers_all_params,
    assert_gradient_map_nonneg,
    assert_hook_fires_match_steps,
    assert_weights_unchanged_from_snapshot,
)

# Number of synthetic backward steps used in hook tests.
_N_STEPS = 5
# EMA decay factor used across tests.
_BETA = 0.9


# ---------------------------------------------------------------------------
# Fixture: parametrised over architectures
# ---------------------------------------------------------------------------


@pytest.fixture(params=list(TINY_HF_FACTORIES.keys()))
def tiny_causal_lm(request: pytest.FixtureRequest) -> nn.Module:
    """Fresh tiny causal-LM created from config for each architecture."""
    return TINY_HF_FACTORIES[request.param]()


# ---------------------------------------------------------------------------
# Helper: collect EMA gradient map after N steps
# ---------------------------------------------------------------------------


def _collect_ema_grads(
    model: nn.Module,
    n_steps: int,
    beta: float,
    abs_grad: bool,
) -> dict[str, torch.Tensor]:
    """Register hooks, run n_steps synthetic backward passes, return grad map."""
    _register_ema_hooks(model, beta=beta, abs_grad=abs_grad)
    for _ in range(n_steps):
        run_single_synthetic_step(model)
    return {
        n: p.grad.float().cpu().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


# ---------------------------------------------------------------------------
# Test 1: weights unchanged and all params have grads after N steps
# ---------------------------------------------------------------------------


def test_ema_hooks_weights_invariant_and_grads_populated(
    tiny_causal_lm: nn.Module,
) -> None:
    """EMA hooks leave model weights untouched and populate every trainable param's grad.

    After _N_STEPS backward passes:
    - model.named_parameters() values must be byte-for-byte equal to the pre-hook snapshot
    - every trainable param must have a non-None .grad
    - _hook_fires[name] == _N_STEPS for each param
    - grad map contains every trainable param key
    """
    model = tiny_causal_lm
    snapshot = {n: p.data.cpu().clone() for n, p in model.named_parameters()}

    grad_map = _collect_ema_grads(model, n_steps=_N_STEPS, beta=_BETA, abs_grad=False)

    assert_weights_unchanged_from_snapshot(model, snapshot)
    assert_hook_fires_match_steps(model, _N_STEPS)
    assert_gradient_map_covers_all_params(model, grad_map)

    print(
        f"✅ test_ema_hooks_weights_invariant_and_grads_populated: "
        f"{len(grad_map)} params covered, weights unchanged after {_N_STEPS} steps"
    )


# ---------------------------------------------------------------------------
# Test 2: abs mode gives non-negative map and differs from signed mode
# ---------------------------------------------------------------------------


def test_ema_hooks_abs_vs_signed(tiny_causal_lm: nn.Module) -> None:
    """Abs-mode EMA must be non-negative everywhere and differ from signed EMA.

    We run two separate models (fresh copies) through identical synthetic data:
    one with abs_grad=False, one with abs_grad=True.  The abs map must be >= 0
    everywhere, and the two maps must differ (gradients are mixed-sign, so the
    abs transform changes the values).
    """
    model_signed = tiny_causal_lm
    model_abs = copy.deepcopy(tiny_causal_lm)

    # Use a fixed manual seed so both models see the same synthetic inputs.
    torch.manual_seed(99)
    grad_map_signed = _collect_ema_grads(
        model_signed, n_steps=_N_STEPS, beta=_BETA, abs_grad=False
    )

    torch.manual_seed(99)
    grad_map_abs = _collect_ema_grads(
        model_abs, n_steps=_N_STEPS, beta=_BETA, abs_grad=True
    )

    assert_gradient_map_nonneg(grad_map_abs)
    assert_grad_maps_differ(grad_map_signed, grad_map_abs, "signed", "abs")

    print(
        "✅ test_ema_hooks_abs_vs_signed: abs map is non-negative and differs from signed map"
    )


# ---------------------------------------------------------------------------
# Test 3: parameters not in the computation graph accumulate zero gradient
# ---------------------------------------------------------------------------


class _ModelWithDeadBranch(nn.Module):
    """Simple model where one branch is deliberately excluded from forward."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.active = nn.Linear(4, 4, bias=False)
        # Never used in forward — should never receive a gradient.
        self.dead = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.active(x)


def test_gradient_ranking_dead_params_get_zero_saliency() -> None:
    """A dead parameter (absent from forward graph) must yield zero EMA gradient.

    After _N_STEPS backward passes, dead.weight.grad must be None (the EMA hook
    is registered but never fires because dead.weight has no gradient flow).
    Only active.weight should appear in the resulting gradient map.
    """
    model = _ModelWithDeadBranch()
    _register_ema_hooks(model, beta=0.0, abs_grad=True)  # beta=0: ema = latest grad

    for _ in range(_N_STEPS):
        if hasattr(model, "_ema_seen"):
            model._ema_seen.clear()
        x = torch.randn(2, 4)
        loss = model(x).pow(2).mean()
        loss.backward()

    active_grad = model.active.weight.grad
    dead_grad = model.dead.weight.grad

    assert active_grad is not None, (
        "❌ test_gradient_ranking_dead_params_get_zero_saliency: "
        "active.weight.grad is None — the hook should have populated it"
    )
    assert dead_grad is None, (
        "❌ test_gradient_ranking_dead_params_get_zero_saliency: "
        "dead.weight.grad is not None — a dead parameter should receive no gradient"
    )

    # Grad map (as GradCollectTrainer.ema_grads would build it) must not include dead.
    grad_map = {
        n: p.grad.float().cpu()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    assert "active.weight" in grad_map, (
        "❌ test_gradient_ranking_dead_params_get_zero_saliency: "
        "active.weight missing from grad_map"
    )
    assert "dead.weight" not in grad_map, (
        "❌ test_gradient_ranking_dead_params_get_zero_saliency: "
        "dead.weight should not be in grad_map"
    )

    print(
        "✅ test_gradient_ranking_dead_params_get_zero_saliency: "
        "active param in map, dead param excluded (grad is None)"
    )
