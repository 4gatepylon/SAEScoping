"""Shared property validators for gradient-map and pruning-pipeline tests.

Every function raises AssertionError with a descriptive ❌ message on failure.
Callers can also enable pytest -s output to see ✅ messages on success.

All functions are pure (no side effects) and take plain torch tensors / nn.Module.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Gradient-map validators
# ---------------------------------------------------------------------------


def assert_gradient_map_covers_all_params(
    model: nn.Module,
    grad_map: dict[str, torch.Tensor],
) -> None:
    """Every trainable parameter key appears in grad_map (no missing, no extras)."""
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    missing = expected - set(grad_map.keys())
    extra = set(grad_map.keys()) - expected
    assert not missing, (
        f"❌ assert_gradient_map_covers_all_params: "
        f"{len(missing)} trainable param(s) absent from grad_map: {sorted(missing)[:5]}"
    )
    assert not extra, (
        f"❌ assert_gradient_map_covers_all_params: "
        f"{len(extra)} unexpected key(s) in grad_map: {sorted(extra)[:5]}"
    )


def assert_gradient_map_nonneg(grad_map: dict[str, torch.Tensor]) -> None:
    """All tensor values are >= 0 (expected after abs-mode EMA accumulation)."""
    for name, tensor in grad_map.items():
        min_val = tensor.min().item()
        assert min_val >= -1e-6, (
            f"❌ assert_gradient_map_nonneg: '{name}' contains negative value {min_val:.6f}"
        )


def assert_hook_fires_match_steps(model: nn.Module, n_steps: int) -> None:
    """model._hook_fires[name] == n_steps for every hooked trainable parameter."""
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        fires = model._hook_fires.get(name, 0)
        assert fires == n_steps, (
            f"❌ assert_hook_fires_match_steps: '{name}' fired {fires} times, "
            f"expected {n_steps}"
        )


def assert_weights_unchanged_from_snapshot(
    model: nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> None:
    """Every parameter value is byte-for-byte equal to the pre-hook snapshot."""
    for name, param in model.named_parameters():
        assert name in snapshot, (
            f"❌ assert_weights_unchanged_from_snapshot: '{name}' not in snapshot"
        )
        assert torch.equal(param.data.cpu(), snapshot[name].cpu()), (
            f"❌ assert_weights_unchanged_from_snapshot: '{name}' changed "
            f"(hooks must not modify weights)"
        )


def assert_grad_maps_differ(
    map_a: dict[str, torch.Tensor],
    map_b: dict[str, torch.Tensor],
    label_a: str = "map_a",
    label_b: str = "map_b",
) -> None:
    """At least one tensor differs between map_a and map_b (e.g. abs vs signed)."""
    if set(map_a.keys()) != set(map_b.keys()):
        return  # different key sets → definitely different
    for k in map_a:
        if not torch.equal(map_a[k], map_b[k]):
            return  # found a differing tensor → pass
    raise AssertionError(
        f"❌ assert_grad_maps_differ: {label_a} and {label_b} are identical — "
        "expected them to differ (e.g. abs vs signed EMA on mixed-sign gradients)"
    )


# ---------------------------------------------------------------------------
# Pruning validators
# ---------------------------------------------------------------------------


def assert_pruning_is_lowest_saliency(
    model: nn.Module,
    saliency_scores: dict[str, torch.Tensor],
) -> None:
    """Global optimality: every zeroed weight has saliency <= every kept weight.

    This is the key correctness invariant: after a correct pruning pass, the
    maximum saliency score among all pruned (zero) weights must be no greater
    than the minimum saliency score among all kept (non-zero) weights, across
    ALL scored parameters jointly.

    Ties at the threshold are handled with a small floating-point tolerance.
    """
    zero_scores: list[torch.Tensor] = []
    nonzero_scores: list[torch.Tensor] = []
    for name, param in model.named_parameters():
        if name not in saliency_scores:
            continue
        scores_flat = saliency_scores[name].flatten().cpu().float()
        weights_flat = param.data.flatten().cpu()
        zero_mask = weights_flat == 0
        if zero_mask.any():
            zero_scores.append(scores_flat[zero_mask])
        if (~zero_mask).any():
            nonzero_scores.append(scores_flat[~zero_mask])

    if not zero_scores or not nonzero_scores:
        return  # nothing pruned or nothing kept — trivially satisfied

    max_pruned = torch.cat(zero_scores).max().item()
    min_kept = torch.cat(nonzero_scores).min().item()
    assert max_pruned <= min_kept + 1e-6, (
        f"❌ assert_pruning_is_lowest_saliency: "
        f"a pruned weight has saliency {max_pruned:.6f} which exceeds "
        f"a kept weight's saliency {min_kept:.6f}. "
        "The globally lowest-saliency weights should always be pruned first."
    )


def assert_sparsity_achieved(
    model: nn.Module,
    saliency_scores: dict[str, torch.Tensor],
    target: float,
    tol: float = 0.02,
) -> None:
    """Actual sparsity fraction (within scored params) is within `tol` of `target`."""
    total = sum(s.numel() for s in saliency_scores.values())
    zeroed = sum(
        int((param.data == 0).sum().item())
        for name, param in model.named_parameters()
        if name in saliency_scores
    )
    actual = zeroed / total if total > 0 else 0.0
    assert abs(actual - target) <= tol, (
        f"❌ assert_sparsity_achieved: actual sparsity {actual:.4f} vs "
        f"target {target:.4f} (tolerance {tol})"
    )


def assert_restore_is_exact(
    model: nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> None:
    """After restore_original_weights, every parameter equals the snapshot exactly."""
    for name, param in model.named_parameters():
        assert name in snapshot, (
            f"❌ assert_restore_is_exact: '{name}' not in snapshot"
        )
        assert torch.equal(param.data.cpu(), snapshot[name].cpu()), (
            f"❌ assert_restore_is_exact: '{name}' not restored correctly after pruning"
        )


def assert_pruning_sets_differ(
    model_a: nn.Module,
    model_b: nn.Module,
    saliency_keys: list[str],
) -> None:
    """The set of zeroed weights differs between model_a and model_b.

    Used to verify that gradient and Taylor scoring produce different pruning sets.
    """
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())
    for name in saliency_keys:
        if name not in params_a or name not in params_b:
            continue
        mask_a = (params_a[name].data == 0).cpu()
        mask_b = (params_b[name].data == 0).cpu()
        if not torch.equal(mask_a, mask_b):
            return  # found at least one differing pruning mask → pass
    raise AssertionError(
        "❌ assert_pruning_sets_differ: gradient and Taylor pruning zeroed "
        "exactly the same weights — expected different pruning sets."
    )
