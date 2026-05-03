"""Pairwise zero-set overlap between safetensors at a sparsity sweep.

For each tensor, the bottom int(N*p) entries by value are treated as zeroed
at sparsity p. Overlap = |zero(A) ∩ zero(B)| / |zero(A)|. Ranks are computed
once per tensor and reused across sparsities — order-independent, but we
iterate ascending for readable output. With rank-threshold pruning,
|zero(A)| == |zero(B)| = sum_t int(N_t * p) by construction.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import click
import torch
from safetensors import safe_open
from tabulate import tabulate

DEFAULT_SPARSITIES = "0.1,0.2,0.3,0.5,0.7,0.9"


def _parse_sparsities(spec: str) -> list[float]:
    out = sorted({float(x) for x in spec.split(",") if x.strip()})
    if not all(0 < p < 1 for p in out):
        raise click.BadParameter(f"Sparsities must be in (0,1); got {out}")
    return out


def _flat_rank(t: torch.Tensor) -> torch.Tensor:
    """rank[i] = ascending position of t.flatten()[i]. Smaller value -> smaller rank -> pruned first."""
    flat = t.reshape(-1)
    sorted_idx = torch.argsort(flat)
    rank = torch.empty_like(sorted_idx)
    rank[sorted_idx] = torch.arange(flat.numel(), device=flat.device)
    return rank


def overlap_by_sparsity(a: Path, b: Path, sparsities: list[float], device: str) -> dict[float, float]:
    n_zero = {p: 0 for p in sparsities}
    n_int = {p: 0 for p in sparsities}
    with safe_open(str(a), framework="pt", device="cpu") as fa, \
         safe_open(str(b), framework="pt", device="cpu") as fb:
        ka, kb = set(fa.keys()), set(fb.keys())
        if ka != kb:
            raise click.ClickException(f"Key mismatch: {a} vs {b}")
        for k in sorted(ka):
            ta = fa.get_tensor(k).to(device=device).reshape(-1)
            tb = fb.get_tensor(k).to(device=device).reshape(-1)
            if ta.shape != tb.shape:
                raise click.ClickException(f"Shape mismatch on '{k}': {ta.shape} vs {tb.shape}")
            ra, rb = _flat_rank(ta), _flat_rank(tb)
            N = ta.numel()
            for p in sparsities:  # rank reused; sorted asc only matters for output
                k_prune = int(N * p)
                if k_prune == 0:
                    continue
                inter = int(((ra < k_prune) & (rb < k_prune)).sum().item())
                n_zero[p] += k_prune  # |zero(A)| == |zero(B)| == k_prune by construction
                n_int[p] += inter
    return {p: (n_int[p] / n_zero[p] if n_zero[p] else 0.0) for p in sparsities}


def _find_maps(root: Path) -> list[tuple[str, Path]]:
    paths = sorted(root.glob("**/*.safetensors"))
    if len(paths) < 2:
        raise click.ClickException(f"Need ≥2 safetensors under {root}, found {len(paths)}")
    return [(p.parent.name, p) for p in paths]


@click.command()
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--sparsities", default=DEFAULT_SPARSITIES, show_default=True)
@click.option("--device", default="cpu", show_default=True)
def main(root: Path, sparsities: str, device: str) -> None:
    """Pairwise zero-set overlap of all safetensors under ROOT, swept over sparsity."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise click.ClickException("CUDA requested but not available")
    sparsity_list = _parse_sparsities(sparsities)
    maps = _find_maps(root)
    click.echo(f"Found {len(maps)} maps under {root}: {[lbl for lbl, _ in maps]}")

    rows = []
    for (la, pa), (lb, pb) in combinations(maps, 2):
        click.echo(f"  {la} vs {lb} ...")
        ov = overlap_by_sparsity(pa, pb, sparsity_list, device)
        for p in sparsity_list:
            rows.append([f"{la} vs {lb}", p, ov[p]])
    print(tabulate(rows, headers=["pair", "sparsity", "overlap"], floatfmt=".4f"))


if __name__ == "__main__":
    # Run once per model:
#   python separation_overlap.py deleteme-cache-old-v1/google--gemma-2-9b-it
#   python separation_overlap.py deleteme-cache-old-v1/google--gemma-3-12b-it
    main()

