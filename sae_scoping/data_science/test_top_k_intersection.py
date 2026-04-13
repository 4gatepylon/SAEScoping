"""Tests for top_k_overlap + AUC using synthetic firing-rate distributions."""

from __future__ import annotations

import pytest
import torch

from sae_scoping.data_science.top_k_intersection import (
    default_ks,
    overlap_curve_auc,
    pairwise_overlap_auc_matrix,
    top_k_overlap_curve,
)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.sum()


# ---- top_k_overlap_curve ---------------------------------------------------


def test_identical_distributions_full_overlap() -> None:
    torch.manual_seed(0)
    d = _normalize(torch.rand(1024))
    curve = top_k_overlap_curve(d, d, ks=[1, 10, 100, 1024])
    assert torch.equal(curve, torch.ones(4, dtype=torch.float64))


def test_disjoint_top_k_zero_overlap() -> None:
    n = 100
    a = torch.zeros(n)
    a[:10] = 0.1  # mass on indices 0..9
    b = torch.zeros(n)
    b[-10:] = 0.1  # mass on indices 90..99
    curve = top_k_overlap_curve(a, b, ks=[1, 5, 10])
    assert torch.equal(curve, torch.zeros(3, dtype=torch.float64))


def test_full_k_equals_n_is_one() -> None:
    torch.manual_seed(1)
    a = _normalize(torch.rand(64))
    b = _normalize(torch.rand(64))
    curve = top_k_overlap_curve(a, b, ks=[64])
    assert curve[0] == 1.0


def test_partial_overlap() -> None:
    n = 20
    a = torch.zeros(n)
    b = torch.zeros(n)
    a[:5] = 0.2  # top-5 of a: {0..4}
    b[3:8] = 0.2  # top-5 of b: {3..7}, overlap {3, 4} -> 2/5
    curve = top_k_overlap_curve(a, b, ks=[5])
    assert curve[0].item() == pytest.approx(2 / 5)


def test_default_ks_log_spaced() -> None:
    assert torch.equal(default_ks(16), torch.tensor([1, 2, 4, 8, 16]))
    assert torch.equal(
        default_ks(1000),
        torch.tensor([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]),
    )


def test_returns_tensor_of_expected_shape() -> None:
    torch.manual_seed(2)
    a = _normalize(torch.rand(16))
    b = _normalize(torch.rand(16))
    curve = top_k_overlap_curve(a, b)
    assert curve.shape == (5,)  # default_ks(16) has 5 entries
    assert curve.dtype == torch.float64


def test_rejects_non_1d() -> None:
    a = torch.ones(4, 4) / 16
    b = torch.ones(16) / 16
    with pytest.raises(ValueError, match="1-D"):
        top_k_overlap_curve(a, b)


def test_rejects_width_mismatch() -> None:
    a = _normalize(torch.ones(10))
    b = _normalize(torch.ones(11))
    with pytest.raises(ValueError, match="Width mismatch"):
        top_k_overlap_curve(a, b)


def test_rejects_nonnormalized() -> None:
    a = torch.ones(10)  # sums to 10
    b = _normalize(torch.ones(10))
    with pytest.raises(ValueError, match="does not sum to 1"):
        top_k_overlap_curve(a, b)


def test_rejects_negative_entries() -> None:
    a = torch.zeros(10)
    a[0] = 2.0
    a[1] = -1.0  # sums to 1 but has a negative
    b = _normalize(torch.ones(10))
    with pytest.raises(ValueError, match="negative"):
        top_k_overlap_curve(a, b)


def test_rejects_k_out_of_range() -> None:
    a = _normalize(torch.ones(10))
    b = _normalize(torch.ones(10))
    with pytest.raises(ValueError, match="out of range"):
        top_k_overlap_curve(a, b, ks=[11])


# ---- overlap_curve_auc -----------------------------------------------------


def test_auc_all_ones_is_one() -> None:
    ks = [1, 2, 4, 8, 16]
    curve = torch.ones(5, dtype=torch.float64)
    # Anchor (0,0) then ones everywhere => integral from 0 to 1 of step fn ≈
    # trapezoid between (0,0) and (1/16, 1) + flat 1 up to (1,1)
    # = 0.5 * (1/16) * 1 + (1 - 1/16) * 1 = 0.5/16 + 15/16 = 0.96875
    assert overlap_curve_auc(curve, ks, n=16) == pytest.approx(1 - 0.5 / 16)


def test_auc_all_zeros_is_zero() -> None:
    ks = [1, 2, 4, 8, 16]
    curve = torch.zeros(5, dtype=torch.float64)
    assert overlap_curve_auc(curve, ks, n=16) == 0.0


def test_auc_identical_distribution_via_curve() -> None:
    torch.manual_seed(3)
    n = 128
    d = _normalize(torch.rand(n))
    ks = default_ks(n)
    curve = top_k_overlap_curve(d, d, ks=ks)
    auc = overlap_curve_auc(curve, ks, n=n)
    # Curve is all ones; AUC = 1 - 0.5/n (from the (0,0) anchor to first point)
    assert auc == pytest.approx(1 - 0.5 / n)


def test_auc_monotonic_in_curve_values() -> None:
    ks = [1, 2, 4]
    low = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    high = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64)
    assert overlap_curve_auc(high, ks, n=4) > overlap_curve_auc(low, ks, n=4)


def test_auc_curve_length_mismatch() -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        overlap_curve_auc(torch.ones(3), [1, 2], n=16)


# ---- pairwise_overlap_auc_matrix -------------------------------------------


def test_matrix_symmetric_diag_one() -> None:
    torch.manual_seed(4)
    dists = {
        "a": _normalize(torch.rand(32)),
        "b": _normalize(torch.rand(32)),
        "c": _normalize(torch.rand(32)),
    }
    names, m = pairwise_overlap_auc_matrix(dists)
    assert names == ["a", "b", "c"]
    assert m.shape == (3, 3)
    assert torch.allclose(m.diag(), torch.ones(3, dtype=torch.float64))
    assert torch.allclose(m, m.T)


def test_matrix_rejects_width_mismatch() -> None:
    dists = {
        "a": _normalize(torch.ones(10)),
        "b": _normalize(torch.ones(20)),
    }
    with pytest.raises(ValueError, match="share width"):
        pairwise_overlap_auc_matrix(dists)


def test_matrix_rejects_too_few() -> None:
    dists = {"a": _normalize(torch.ones(10))}
    with pytest.raises(ValueError, match="at least 2"):
        pairwise_overlap_auc_matrix(dists)
