"""OOD grouped bar plot: x-axis is (scope_domain, method), overlaid bars per elicitation domain."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from .config_schemas import PlotConfig
from .data_loading import get_score

ELICIT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def plot_ood_bar(df: pd.DataFrame, config: PlotConfig, model_id: str, output_dir: Path) -> Path:
    fig_config = config.figures.ood_bar
    if fig_config is None:
        raise ValueError("ood_bar not configured")

    domain_ids = [d.id for d in config.domains.entries]
    method_ids = fig_config.methods
    n_domains = len(domain_ids)
    n_methods = len(method_ids)

    vanilla_scores: dict[str, float] = {}
    if fig_config.relative:
        vanilla_method = None
        for m in config.methods:
            if not m.requires_scope_domain:
                vanilla_method = m
                break
        if vanilla_method:
            for d in domain_ids:
                v = get_score(df, config, model_id, vanilla_method.id, elicitation_domain=d)
                if v is not None and v > 0:
                    vanilla_scores[d] = v

    elicit_color_map = {d: ELICIT_COLORS[i % len(ELICIT_COLORS)] for i, d in enumerate(domain_ids)}

    n_groups = n_domains * n_methods
    group_labels = []
    for scope_d in domain_ids:
        for mid in method_ids:
            method_cfg = config.get_method(mid)
            scope_label = config.domains.get_display_name(scope_d)
            group_labels.append(f"{scope_label}\n{method_cfg.display_name}")

    bar_width = 0.8 / n_domains
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(12, n_groups * 0.9), 6))

    for elicit_idx, elicit_d in enumerate(domain_ids):
        vals = []
        for scope_d in domain_ids:
            for mid in method_ids:
                method_cfg = config.get_method(mid)
                if method_cfg.requires_scope_domain:
                    s = get_score(df, config, model_id, mid, elicitation_domain=elicit_d, scope_domain=scope_d)
                else:
                    s = get_score(df, config, model_id, mid, elicitation_domain=elicit_d)

                if s is not None and fig_config.relative and elicit_d in vanilla_scores:
                    s = s / vanilla_scores[elicit_d]
                vals.append(s if s is not None else 0)

        positions = x + (elicit_idx - n_domains / 2 + 0.5) * bar_width
        ax.bar(
            positions, vals, bar_width,
            label=config.domains.get_display_name(elicit_d),
            color=elicit_color_map[elicit_d],
            edgecolor="white", linewidth=0.3,
        )

    for scope_idx in range(1, n_domains):
        sep_x = scope_idx * n_methods - 0.5
        ax.axvline(x=sep_x, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=8, ha="center")
    y_label = "Relative Performance" if fig_config.relative else "Score"
    ax.set_ylabel(y_label, fontsize=11)

    model_display = next((m.display_name for m in config.models if m.id == model_id), model_id)
    ax.set_title(fig_config.title_template.format(model=model_display), fontsize=13, fontweight="bold")

    ax.legend(title="Elicitation Domain", loc="upper right", fontsize=8, title_fontsize=9)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    if fig_config.relative:
        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    out_path = output_dir / f"ood_bar_{model_id}.png"
    fig.savefig(out_path, dpi=config.output.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
