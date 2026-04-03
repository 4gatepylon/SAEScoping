"""find_threshold_2026_04_03.py

For every distribution.safetensors found under distributions_cache/ and
downloaded/, produce:

  threshold_info_2026_04_03/
    <label>.png          — side-by-side linear + log threshold curve, annotated
    <label>.json         — operating points at the marked n / h values
    comparison.png       — all curves overlaid for cross-topic comparison

The threshold at n neurons is simply sorted_distribution[n-1] (descending order),
i.e. the n-th largest firing-rate value.

Usage (from biology_pruning_3e-4_2026_03_30/ directory):
    python jobs_2026_04_03/find_threshold_2026_04_03.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from matplotlib.transforms import blended_transform_factory
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

V_NEURONS    = [2000, 4000, 8000]            # vertical marker lines (neuron counts)
H_THRESHOLDS = [1e-4, 2e-4, 3e-4, 4e-4]     # horizontal marker lines (thresholds)

# Colors for intersection dots: first 3 → vertical, last 4 → horizontal
_DOT_COLORS = [
    "#1f77b4",  # n=2000  (blue)
    "#2ca02c",  # n=4000  (green)
    "#ff7f0e",  # n=8000  (orange)
    "#9467bd",  # h=1e-4  (purple)
    "#d62728",  # h=2e-4  (red)
    "#8c564b",  # h=3e-4  (brown)
    "#e377c2",  # h=4e-4  (pink)
]

_OUTPUT_DIR_NAME = "threshold_info_2026_04_03"

_SEARCH_GLOBS = [
    "distributions_cache/**/distribution.safetensors",
    "downloaded/**/distribution.safetensors",
]

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _label_from_path(path: Path, parent: Path) -> str:
    """Extract a human-readable label from a distribution file path."""
    rel = path.relative_to(parent)
    parts = rel.parts
    # Look for a known subject name in the path parts
    subjects = {"biology", "chemistry", "physics", "math", "apps", "ultrachat"}
    subject = next((p for p in parts if p in subjects), None)
    # Look for an SAE id (contains '--')
    sae_parts = [p for p in parts if "--" in p]
    sae = sae_parts[-1].replace("--", "/") if sae_parts else ""
    # Tag as 'downloaded' if coming from the downloaded/ tree
    source = "downloaded" if parts[0] == "downloaded" else "new"
    label = subject or "unknown"
    if source == "downloaded":
        label += " [downloaded]"
    if sae:
        label += f"  ({sae})"
    return label


def find_distributions(parent: Path) -> list[tuple[str, Path]]:
    results = []
    for glob in _SEARCH_GLOBS:
        for p in sorted(parent.glob(glob)):
            results.append((_label_from_path(p, parent), p))
    return results


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def load_distribution(path: Path) -> torch.Tensor:
    return load_file(str(path))["distribution"]


def sorted_curve(dist: torch.Tensor) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (fractions, threshold_values, d_sae).

    fractions[k]         = (k+1) / d_sae   — fraction of neurons kept
    threshold_values[k]  = sorted_dist[k]  — threshold that keeps exactly k+1 neurons
    """
    d_sae = len(dist)
    sv = torch.sort(dist, descending=True).values.numpy()
    fracs = np.arange(1, d_sae + 1, dtype=np.float64) / d_sae
    return fracs, sv, d_sae


def compute_operating_points(
    sv: np.ndarray,
    dist: torch.Tensor,
    d_sae: int,
) -> dict:
    """Compute intersection coordinates for each marker line.

    Returns a dict with:
      vertical:   list of {n_neurons, threshold, frac}
      horizontal: list of {h_threshold, n_neurons, frac}
    """
    v_pts = []
    for n in V_NEURONS:
        if n <= d_sae:
            h = float(sv[n - 1])
            v_pts.append({"n_neurons": n, "threshold": h, "frac": n / d_sae})

    h_pts = []
    dist_np = dist.numpy()
    for h in H_THRESHOLDS:
        n_kept = int((dist_np >= h).sum())
        h_pts.append({"h_threshold": h, "n_neurons": n_kept, "frac": n_kept / d_sae})

    return {"vertical_markers": v_pts, "horizontal_markers": h_pts, "d_sae": d_sae}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _annotate_axes(ax, sv: np.ndarray, d_sae: int, yscale: str) -> list:
    """Add marker lines and intersection dots to one axes. Returns legend handles."""
    handles = []

    # ---- vertical lines (n = 2000, 4000, 8000) ----
    for n in V_NEURONS:
        if n > d_sae:
            continue
        frac = n / d_sae
        ax.axvline(frac, color="red", linestyle="--", linewidth=0.9, alpha=0.55, zorder=1)
        # Label at top of axes using mixed (data-x, axes-y) transform
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(frac, 1.01, f"n={n:,}", transform=trans,
                color="red", fontsize=7, ha="center", va="bottom", rotation=90)

    # ---- horizontal lines (h values) ----
    for h in H_THRESHOLDS:
        ax.axhline(h, color="red", linestyle=":", linewidth=0.9, alpha=0.55, zorder=1)
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.005, h, f"{h:.0e}", transform=trans,
                color="red", fontsize=7, ha="left", va="center")

    # ---- vertical intersection dots ----
    color_iter = iter(_DOT_COLORS)
    for n in V_NEURONS:
        c = next(color_iter)
        if n > d_sae:
            continue
        frac = n / d_sae
        h_at_n = float(sv[n - 1])
        sc = ax.scatter(frac, h_at_n, color=c, marker="o", s=70, zorder=6,
                        label=f"n={n:,}  →  h={h_at_n:.3e}")
        handles.append(sc)

    # ---- horizontal intersection dots ----
    for h in H_THRESHOLDS:
        c = next(color_iter)
        n_kept = int((sv >= h).sum())
        frac = n_kept / d_sae
        sc = ax.scatter(frac, h, color=c, marker="s", s=70, zorder=6,
                        label=f"h={h:.1e}  →  n={n_kept:,}")
        handles.append(sc)

    return handles


def plot_single(
    label: str,
    fracs: np.ndarray,
    sv: np.ndarray,
    dist: torch.Tensor,
    d_sae: int,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    fig.suptitle(f"Threshold vs Neurons Kept — {label}", fontsize=11, y=1.02)

    for ax, yscale in zip(axes, ("linear", "log")):
        ax.plot(fracs, sv, color="steelblue", linewidth=1.4, zorder=2, label="_curve")
        handles = _annotate_axes(ax, sv, d_sae, yscale)

        ax.set_yscale(yscale)
        if yscale == "log":
            pos_vals = sv[sv > 0]
            if pos_vals.size:
                ax.set_ylim(bottom=max(pos_vals.min() * 0.3, 1e-9))

        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction of neurons kept", fontsize=9)
        ax.set_ylabel("Threshold (firing-rate fraction)", fontsize=9)
        ax.set_title(f"{'Linear' if yscale == 'linear' else 'Log'} y-axis  |  d_sae={d_sae:,}",
                     fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)

        # Secondary x-axis showing absolute neuron count
        secax = ax.secondary_xaxis(
            "top",
            functions=(lambda f: f * d_sae, lambda n: n / d_sae),
        )
        secax.set_xlabel("Number of neurons kept", fontsize=8)
        secax.tick_params(labelsize=7)

        ax.legend(handles=handles, fontsize=7, loc="upper right",
                  title="Intersection points", title_fontsize=7,
                  framealpha=0.85)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot  -> {out_path}")


def plot_comparison(
    all_curves: list[tuple[str, np.ndarray, np.ndarray, int]],
    out_path: Path,
) -> None:
    """Overlay all curves on one figure for cross-topic comparison."""
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), constrained_layout=True)
    fig.suptitle("Threshold Curves — All Distributions (comparison)", fontsize=11, y=1.02)

    for ax, yscale in zip(axes, ("linear", "log")):
        for i, (label, fracs, sv, d_sae) in enumerate(all_curves):
            ax.plot(fracs, sv, color=cmap(i % 10), linewidth=1.4, label=label, zorder=2)

        # Horizontal marker lines only (less clutter for comparison)
        for h in H_THRESHOLDS:
            ax.axhline(h, color="red", linestyle=":", linewidth=0.8, alpha=0.45, zorder=1)
            trans = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(1.005, h, f"{h:.0e}", transform=trans,
                    color="red", fontsize=7, ha="left", va="center")

        for n in V_NEURONS:
            # Use d_sae of the first curve (all should be same SAE)
            d_sae_ref = all_curves[0][3] if all_curves else 16384
            frac = n / d_sae_ref
            ax.axvline(frac, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)
            trans = blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(frac, 1.01, f"n={n:,}", transform=trans,
                    color="gray", fontsize=7, ha="center", va="bottom", rotation=90)

        ax.set_yscale(yscale)
        if yscale == "log":
            all_pos = np.concatenate([sv[sv > 0] for _, _, sv, _ in all_curves])
            if all_pos.size:
                ax.set_ylim(bottom=max(all_pos.min() * 0.3, 1e-9))

        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction of neurons kept", fontsize=9)
        ax.set_ylabel("Threshold (firing-rate fraction)", fontsize=9)
        ax.set_title(f"{'Linear' if yscale == 'linear' else 'Log'} y-axis", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  comparison plot -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Resolve paths relative to this script's parent's parent
    # (i.e. biology_pruning_3e-4_2026_03_30/ when script is in jobs_2026_04_03/)
    script_dir = Path(__file__).resolve().parent
    parent = script_dir.parent
    out_dir = parent / _OUTPUT_DIR_NAME

    dist_files = find_distributions(parent)
    if not dist_files:
        print("No distribution.safetensors files found. Run run_distributions.sh first.")
        return

    print(f"Found {len(dist_files)} distribution file(s):")
    for lbl, p in dist_files:
        print(f"  {lbl}")
        print(f"    {p.relative_to(parent)}")
    print()

    all_curves: list[tuple[str, np.ndarray, np.ndarray, int]] = []

    for label, dist_path in dist_files:
        print(f"Processing: {label}")
        dist = load_distribution(dist_path)
        fracs, sv, d_sae = sorted_curve(dist)
        ops = compute_operating_points(sv, dist, d_sae)

        # Safe filename: strip brackets, parens, slashes, spaces
        safe = re.sub(r"[^\w\-]", "_", label).strip("_")
        safe = re.sub(r"_+", "_", safe)

        # Plot
        plot_single(label, fracs, sv, dist, d_sae, out_dir / f"{safe}.png")

        # JSON
        json_path = out_dir / f"{safe}.json"
        with open(json_path, "w") as f:
            json.dump(ops, f, indent=2)
        print(f"  json  -> {json_path}")

        all_curves.append((label, fracs, sv, d_sae))
        print()

    # Comparison plot
    if len(all_curves) > 1:
        plot_comparison(all_curves, out_dir / "comparison.png")

    print(f"\nAll outputs in: {out_dir}")


if __name__ == "__main__":
    main()
