"""find_threshold.py

Given a distribution.safetensors file, find the threshold h such that
exactly (approximately) N neurons are kept (i.e. distribution >= h).

Usage:
    python find_threshold.py <path_to_distribution.safetensors> [--n-neurons 2000]

Example:
    python find_threshold.py \\
        ../distributions_cache/ignore_padding_True/chemistry/layer_31--width_16k--canonical/distribution.safetensors \\
        --n-neurons 2000
"""

from __future__ import annotations

import click
import torch
from safetensors.torch import load_file


@click.command()
@click.argument("dist_path", type=click.Path(exists=True))
@click.option("--n-neurons", "-n", type=int, default=2000,
              help="Target number of neurons to keep. Default: 2000.")
def main(dist_path: str, n_neurons: int) -> None:
    data = load_file(dist_path)
    dist: torch.Tensor = data["distribution"]
    d_sae = len(dist)

    # Sort descending to find the value at position n_neurons-1
    sorted_vals, _ = torch.sort(dist, descending=True)

    if n_neurons > d_sae:
        raise click.BadParameter(
            f"n_neurons={n_neurons} exceeds SAE width {d_sae}"
        )

    threshold_exact = sorted_vals[n_neurons - 1].item()
    n_kept_at_threshold = int((dist >= threshold_exact).sum().item())

    # For cleaner reporting also show nearby round values
    candidates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3]

    print(f"\nDistribution: {dist_path}")
    print(f"SAE width   : {d_sae}")
    print(f"Target      : {n_neurons} neurons kept")
    print(f"\nExact threshold to keep {n_kept_at_threshold} neurons: h = {threshold_exact:.6e}")
    print()
    print(f"{'Threshold':<14}  {'Neurons kept':>12}  {'Fraction':>10}")
    print("-" * 42)
    for h in candidates:
        n = int((dist >= h).sum().item())
        print(f"{h:<14.2e}  {n:>12d}  {n / d_sae:>10.4f}")

    # Binary search for a threshold that gives exactly n_neurons
    lo, hi = 0.0, sorted_vals[0].item()
    for _ in range(60):
        mid = (lo + hi) / 2
        n = int((dist >= mid).sum().item())
        if n > n_neurons:
            lo = mid
        else:
            hi = mid
    h_bsearch = (lo + hi) / 2
    n_bsearch = int((dist >= h_bsearch).sum().item())
    print(f"\nBinary-search threshold for ~{n_neurons} neurons: h = {h_bsearch:.6e}  -> keeps {n_bsearch}")


if __name__ == "__main__":
    main()
