"""End-to-end unit tests for the pruning pipeline (apply_pruning and friends).

All tests run on CPU with either tiny synthetic nn.Module models or randomly
initialised HuggingFace models created from config (no weight downloads).

Tests
-----
test_pruning_selects_globally_lowest_saliency
    Uses a fixture with analytically known saliency values and verifies that
    apply_pruning zeroes exactly the expected low-saliency group.

test_taylor_vs_gradient_produce_different_pruning
    Constructs a case where gradient-based and Taylor-based saliency lead to
    different pruning decisions and verifies both are locally optimal.

test_full_pipeline_single_sparsity  [qwen2, llama]
    Runs apply_pruning on a tiny real HF model with a random saliency map at a
    moderate sparsity level and validates all correctness properties.

test_multi_sparsity_sweep_with_restore  [qwen2, llama]
    Runs the full sweep loop: save weights → prune at each of 7 sparsity levels
    → validate → restore → verify restore is exact.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from sweep_eval_temp import (
    apply_pruning,
    restore_original_weights,
    save_original_weights,
)
from gradients_map.random import make_random_map
from tests.unit.model_factories import (
    TINY_HF_FACTORIES,
    make_known_saliency_fixture,
    make_taylor_vs_gradient_fixture,
)
from tests.unit.validators import (
    assert_pruning_is_lowest_saliency,
    assert_pruning_sets_differ,
    assert_restore_is_exact,
    assert_sparsity_achieved,
)

# Sparsity levels used for the multi-level sweep test.
_SWEEP_SPARSITIES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


# ---------------------------------------------------------------------------
# Fixture: parametrised over architectures
# ---------------------------------------------------------------------------


@pytest.fixture(params=list(TINY_HF_FACTORIES.keys()))
def tiny_causal_lm(request: pytest.FixtureRequest) -> nn.Module:
    """Fresh tiny causal-LM created from config for each architecture."""
    return TINY_HF_FACTORIES[request.param]()


# ---------------------------------------------------------------------------
# Test 1: known-saliency fixture — pruning selects the analytically correct group
# ---------------------------------------------------------------------------


def test_pruning_selects_globally_lowest_saliency() -> None:
    """apply_pruning must zero exactly the lowest-saliency group.

    Fixture: _TwoGroupModel with 24 params total
      important.weight   (16 params, saliency = 100.0)
      unimportant.weight  (8 params, saliency = 0.001)

    At sparsity = 8/24 ≈ 0.333:
      - unimportant.weight must be all-zero
      - important.weight must be entirely non-zero
      - assert_pruning_is_lowest_saliency must pass
    """
    model, saliency = make_known_saliency_fixture()
    snapshot = save_original_weights(model)
    apply_pruning(model, saliency, sparsity_fraction=8 / 24)

    unimportant_vals = model.unimportant.weight.data
    important_vals = model.important.weight.data

    assert unimportant_vals.count_nonzero().item() == 0, (
        "❌ test_pruning_selects_globally_lowest_saliency: "
        f"unimportant.weight has {unimportant_vals.count_nonzero().item()} non-zero values "
        "after pruning — expected all zeros"
    )
    assert important_vals.count_nonzero().item() == important_vals.numel(), (
        "❌ test_pruning_selects_globally_lowest_saliency: "
        "important.weight was partially pruned — it should be fully preserved"
    )
    assert_pruning_is_lowest_saliency(model, saliency, snapshot)

    print(
        "✅ test_pruning_selects_globally_lowest_saliency: "
        "unimportant group fully pruned, important group fully preserved"
    )


# ---------------------------------------------------------------------------
# Test 2: Taylor vs. gradient — different saliency maps prune different weights
# ---------------------------------------------------------------------------


def test_taylor_vs_gradient_produce_different_pruning() -> None:
    """Gradient and Taylor saliency prune different individual weights.

    Fixture: _SingleLinearModel, fc.weight = [[10.0, 0.1]]
      gradient saliency = [[0.5, 2.0]]  → prunes fc.weight[0,0] (score 0.5)
      Taylor saliency   = [[5.0, 0.2]]  → prunes fc.weight[0,1] (score 0.2)

    After apply_pruning at 50% sparsity (1 of 2 params):
    - gradient model: fc.weight[0,0] == 0, fc.weight[0,1] != 0
    - Taylor   model: fc.weight[0,0] != 0, fc.weight[0,1] == 0
    - assert_pruning_is_lowest_saliency must hold for both
    - assert_pruning_sets_differ must hold across the two models
    """
    model_grad, grad_saliency, taylor_saliency = make_taylor_vs_gradient_fixture()
    model_taylor = copy.deepcopy(model_grad)
    snapshot = save_original_weights(model_grad)

    apply_pruning(model_grad, grad_saliency, sparsity_fraction=0.5)
    apply_pruning(model_taylor, taylor_saliency, sparsity_fraction=0.5)

    assert_pruning_is_lowest_saliency(model_grad, grad_saliency, snapshot)
    assert_pruning_is_lowest_saliency(model_taylor, taylor_saliency, snapshot)
    assert_pruning_sets_differ(model_grad, model_taylor, list(grad_saliency.keys()))

    # Explicit position checks to make the test self-documenting.
    w_grad = model_grad.fc.weight.data.flatten()
    w_taylor = model_taylor.fc.weight.data.flatten()

    assert w_grad[0].item() == 0.0 and w_grad[1].item() != 0.0, (
        "❌ test_taylor_vs_gradient_produce_different_pruning: "
        "gradient model should prune w[0] (score 0.5), keep w[1] (score 2.0)"
    )
    assert w_taylor[1].item() == 0.0 and w_taylor[0].item() != 0.0, (
        "❌ test_taylor_vs_gradient_produce_different_pruning: "
        "Taylor model should prune w[1] (score 0.2), keep w[0] (score 5.0)"
    )

    print(
        "✅ test_taylor_vs_gradient_produce_different_pruning: "
        "gradient prunes w[0], Taylor prunes w[1] — correct divergence"
    )


# ---------------------------------------------------------------------------
# Test 3: full pipeline — random saliency, single sparsity, real HF model
# ---------------------------------------------------------------------------


def test_full_pipeline_single_sparsity(tiny_causal_lm: nn.Module) -> None:
    """apply_pruning on a real HF model with random saliency satisfies all invariants.

    Uses make_random_map to generate a saliency map that covers all params,
    then applies pruning at 40% sparsity and validates:
    - assert_pruning_is_lowest_saliency  (global optimality)
    - assert_sparsity_achieved           (actual sparsity ≈ target)
    """
    model = tiny_causal_lm
    saliency = make_random_map(model, seed=42)
    snapshot = save_original_weights(model)

    apply_pruning(model, saliency, sparsity_fraction=0.4)

    assert_pruning_is_lowest_saliency(model, saliency, snapshot)
    assert_sparsity_achieved(model, saliency, target=0.4, tol=0.02)

    print(
        "✅ test_full_pipeline_single_sparsity: "
        "pruning optimal and sparsity achieved on real HF model"
    )


# ---------------------------------------------------------------------------
# Test 4: multi-sparsity sweep with weight restore
# ---------------------------------------------------------------------------


def test_multi_sparsity_sweep_with_restore(tiny_causal_lm: nn.Module) -> None:
    """Sweep over _SWEEP_SPARSITIES, prune, validate, then restore and verify exactly.

    At each level we check:
    - apply_pruning returns a positive n_zeroed (for sparsity > 0)
    - assert_pruning_is_lowest_saliency
    - assert_sparsity_achieved
    After all levels we verify restore_original_weights returns the model to its
    exact original state via assert_restore_is_exact.
    """
    model = tiny_causal_lm
    saliency = make_random_map(model, seed=7)
    original = save_original_weights(model)

    for sparsity in _SWEEP_SPARSITIES:
        restore_original_weights(model, original)
        n_zeroed = apply_pruning(model, saliency, sparsity_fraction=sparsity)

        if sparsity > 0:
            assert n_zeroed > 0, (
                f"❌ test_multi_sparsity_sweep_with_restore: "
                f"apply_pruning returned 0 zeroed weights at sparsity={sparsity}"
            )
            # original is the pre-pruning state (model was just restored from it)
            assert_pruning_is_lowest_saliency(model, saliency, original)
            assert_sparsity_achieved(model, saliency, target=sparsity, tol=0.02)

    # Final restore and exact equality check.
    restore_original_weights(model, original)
    assert_restore_is_exact(model, original)

    print(
        f"✅ test_multi_sparsity_sweep_with_restore: "
        f"{len(_SWEEP_SPARSITIES)} sparsity levels validated, restore exact"
    )
