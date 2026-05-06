"""Feature overlap scatter: x=SAE neuron overlap, y=OOD performance.

Each (scope_domain, elicitation_domain) pair produces two points at the same
x-coordinate (overlap), connected by a vertical line segment:
  - bottom point: raw/scoped performance (before elicitation)
  - top point: elicited performance (after attack)

All pairs overlaid on the same plot, same color. The line segment helps
identify which raw/elicited points belong together.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config_schemas import PlotConfig
from .data_loading import get_score


def _load_overlap(overlap_csv: str, config_path: Path | None = None) -> pd.DataFrame:
    """Load the overlap CSV. Columns: model, scope_domain, elicitation_domain, overlap."""
    p = Path(overlap_csv)
    if not p.is_absolute() and config_path is not None:
        p = config_path.parent / p
    df = pd.read_csv(p)
    required = {"model", "scope_domain", "elicitation_domain", "overlap"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Overlap CSV missing columns: {missing}")
    return df


def plot_feature_overlap_scatter(
    df: pd.DataFrame, config: PlotConfig, model_id: str, output_dir: Path,
    config_path: Path | None = None,
) -> Path:
    fig_config = config.figures.feature_overlap_scatter
    if fig_config is None:
        raise ValueError("feature_overlap_scatter not configured")

    overlap_df = _load_overlap(fig_config.overlap_csv, config_path)
    overlap_df = overlap_df[overlap_df["model"] == model_id]

    domain_ids = [d.id for d in config.domains.entries]

    vanilla_scores: dict[str, float] = {}
    if fig_config.relative:
        for m in config.methods:
            if not m.requires_scope_domain:
                for d in domain_ids:
                    v = get_score(df, config, model_id, m.id, elicitation_domain=d)
                    if v is not None and v > 0:
                        vanilla_scores[d] = v
                break

    raw_method = fig_config.raw_method
    elicited_method = fig_config.elicited_method
    raw_cfg = config.get_method(raw_method)
    elicited_cfg = config.get_method(elicited_method)

    points = []
    for _, orow in overlap_df.iterrows():
        scope_d = orow["scope_domain"]
        elicit_d = orow["elicitation_domain"]
        overlap = float(orow["overlap"])

        if scope_d == elicit_d:
            continue

        raw_s = get_score(df, config, model_id, raw_method, elicitation_domain=elicit_d, scope_domain=scope_d)
        elicit_s = get_score(df, config, model_id, elicited_method, elicitation_domain=elicit_d, scope_domain=scope_d)

        if raw_s is None or elicit_s is None:
            continue

        if fig_config.relative and elicit_d in vanilla_scores:
            van = vanilla_scores[elicit_d]
            raw_s = raw_s / van
            elicit_s = elicit_s / van

        points.append({
            "scope_domain": scope_d,
            "elicitation_domain": elicit_d,
            "overlap": overlap,
            "raw": raw_s,
            "elicited": elicit_s,
        })

    if not points:
        raise ValueError(f"No valid (scope, elicit) pairs found for model={model_id}")

    from .figure_ood_bar import ELICIT_COLORS
    domain_color_map = {d: ELICIT_COLORS[i % len(ELICIT_COLORS)] for i, d in enumerate(domain_ids)}

    fig, ax = plt.subplots(figsize=(8, 8))

    for p in points:
        ax.plot(
            [p["overlap"], p["overlap"]], [p["raw"], p["elicited"]],
            color="#aaaaaa", linewidth=1.0, alpha=0.6, zorder=1,
        )

    for p in points:
        ax.scatter(
            p["overlap"], p["raw"],
            color=domain_color_map[p["scope_domain"]],
            marker="o", s=60, zorder=3, edgecolors="white", linewidths=0.5,
        )
        ax.scatter(
            p["overlap"], p["elicited"],
            color=domain_color_map[p["elicitation_domain"]],
            marker="^", s=60, zorder=3, edgecolors="white", linewidths=0.5,
        )

    from matplotlib.lines import Line2D
    legend_handles = []
    legend_handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=8, label=f"● {raw_cfg.display_name} (colored by scope domain)"))
    legend_handles.append(Line2D([0], [0], marker="^", color="w", markerfacecolor="gray", markersize=8, label=f"▲ {elicited_cfg.display_name} (colored by elicit domain)"))
    legend_handles.append(Line2D([0], [0], color="#aaaaaa", linewidth=1, label="Connecting segment"))
    legend_handles.append(Line2D([0], [0], linestyle="none", label=""))
    for d in domain_ids:
        legend_handles.append(Line2D([0], [0], marker="s", color="w", markerfacecolor=domain_color_map[d], markersize=8, label=config.domains.get_display_name(d)))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    ax.set_xlabel("Feature Overlap (SAE neuron overlap between domains)", fontsize=11)
    y_label = "Relative OOD Performance" if fig_config.relative else "OOD Performance"
    ax.set_ylabel(y_label, fontsize=11)

    model_display = next((m.display_name for m in config.models if m.id == model_id), model_id)
    ax.set_title(fig_config.title_template.format(model=model_display), fontsize=13, fontweight="bold")

    ax.legend(handles=legend_handles, fontsize=8, loc="upper left", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / f"feature_overlap_scatter_{model_id}.png"
    fig.savefig(out_path, dpi=config.output.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
