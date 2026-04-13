"""Top-k overlap between two SAE firing-rate distributions and AUC similarity.

Given two 1-D probability distributions over SAE neurons (firing rates
normalized to sum to 1), sweep k and compute |topk(A) ∩ topk(B)| / k for each
k. Then reduce the sweep to a single similarity via trapezoidal AUC on the
normalized x-axis k/N (anchored at (0, 0)).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from safetensors import safe_open

_SAFETENSORS_KEY = "distribution"


def load_distribution(path: str | Path, key: str = _SAFETENSORS_KEY) -> torch.Tensor:
    """Load a 1-D firing-rate distribution from a .safetensors file."""
    with safe_open(str(path), framework="pt") as f:
        return f.get_tensor(key)


def _validate(dist: torch.Tensor, name: str, atol: float) -> None:
    if not isinstance(dist, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(dist)}")
    if dist.dim() != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(dist.shape)}")
    if not torch.is_floating_point(dist):
        raise ValueError(f"{name} must be floating point, got dtype {dist.dtype}")
    if (dist < 0).any():
        raise ValueError(f"{name} has negative entries")
    total = float(dist.sum())
    if abs(total - 1.0) > atol:
        raise ValueError(f"{name} does not sum to 1 within atol={atol}: sum={total}")


def default_ks(n: int) -> torch.Tensor:
    """Log-spaced ks: 1, 2, 4, ..., then N (if not already a power of 2)."""
    ks: list[int] = []
    k = 1
    while k < n:
        ks.append(k)
        k *= 2
    ks.append(n)
    return torch.tensor(ks, dtype=torch.long)


def top_k_overlap_curve(
    dist_a: torch.Tensor,
    dist_b: torch.Tensor,
    ks: Sequence[int] | torch.Tensor | None = None,
    atol: float = 1e-3,
) -> torch.Tensor:
    """Return a 1-D tensor of |topk(A) ∩ topk(B)| / k values, in `ks` order.

    Both distributions must be 1-D, non-negative, same length (== SAE width),
    and sum to 1 (±atol). If `ks` is None, uses `default_ks(N)`.
    """
    _validate(dist_a, "dist_a", atol)
    _validate(dist_b, "dist_b", atol)
    if dist_a.numel() != dist_b.numel():
        raise ValueError(
            f"Width mismatch: dist_a has {dist_a.numel()} entries, "
            f"dist_b has {dist_b.numel()}; all distributions in an analysis "
            f"must share the SAE width."
        )

    n = dist_a.numel()
    ks_tensor = (
        default_ks(n) if ks is None else torch.as_tensor(list(ks), dtype=torch.long)
    )
    if ((ks_tensor < 1) | (ks_tensor > n)).any():
        raise ValueError(f"ks out of range [1, {n}]: {ks_tensor.tolist()}")

    out = torch.empty(ks_tensor.numel(), dtype=torch.float64)
    for i, k in enumerate(ks_tensor.tolist()):
        idx_a = torch.topk(dist_a, k).indices
        idx_b = torch.topk(dist_b, k).indices
        inter = torch.isin(idx_a, idx_b).sum().item()
        out[i] = inter / k
    return out


def overlap_curve_auc(
    curve: torch.Tensor,
    ks: Sequence[int] | torch.Tensor,
    n: int,
) -> float:
    """Reduce an overlap curve to a single similarity in [0, 1] via trapezoidal
    integration over x = k/N, anchored at (0, 0).

    At k=N the overlap is always 1.0 (both top-N sets contain all neurons).
    The anchor at (0, 0) penalizes curves that drop sharply at small k.
    """
    if curve.dim() != 1:
        raise ValueError(f"curve must be 1-D, got shape {tuple(curve.shape)}")
    ks_tensor = torch.as_tensor(list(ks), dtype=torch.long)
    if ks_tensor.numel() != curve.numel():
        raise ValueError(
            f"Length mismatch: curve has {curve.numel()} points, "
            f"ks has {ks_tensor.numel()}"
        )
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    # Anchor at (0, 0); x-axis normalized by N.
    xs = torch.cat([torch.zeros(1, dtype=torch.float64), ks_tensor.to(torch.float64) / n])
    ys = torch.cat([torch.zeros(1, dtype=torch.float64), curve.to(torch.float64)])
    # Trapezoidal rule: sum 0.5 * (y_i + y_{i+1}) * (x_{i+1} - x_i)
    return float(torch.trapz(ys, xs))


def pairwise_overlap_auc_matrix(
    dists: dict[str, torch.Tensor],
    ks: Sequence[int] | torch.Tensor | None = None,
    atol: float = 1e-3,
) -> tuple[list[str], torch.Tensor]:
    """Compute the NxN AUC-similarity matrix over a set of named distributions.

    Diagonal entries are 1.0 (self-similarity). Off-diagonal entries use
    `overlap_curve_auc` on `top_k_overlap_curve`. Matrix is symmetric.
    Raises if the distributions do not all share the same width.
    """
    if len(dists) < 2:
        raise ValueError(f"Need at least 2 distributions, got {len(dists)}")
    widths = {name: d.numel() for name, d in dists.items()}
    if len(set(widths.values())) != 1:
        raise ValueError(f"All distributions must share width; got {widths}")
    n = next(iter(widths.values()))
    ks_tensor = default_ks(n) if ks is None else torch.as_tensor(list(ks), dtype=torch.long)

    names = list(dists.keys())
    m = len(names)
    matrix = torch.eye(m, dtype=torch.float64)
    for i in range(m):
        for j in range(i + 1, m):
            curve = top_k_overlap_curve(dists[names[i]], dists[names[j]], ks=ks_tensor, atol=atol)
            sim = overlap_curve_auc(curve, ks_tensor, n)
            matrix[i, j] = sim
            matrix[j, i] = sim
    return names, matrix
