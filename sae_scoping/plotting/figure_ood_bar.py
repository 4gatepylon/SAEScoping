"""OOD grouped bar plot: x-axis is (scope_domain, method), overlaid bars per elicitation domain.

Bars are drawn overlaid (tallest in back, shortest in front) at the same x position,
so the ranking is visible by height. Each x-tick is one (scope_domain, method) combo
corresponding to a column of the OOD table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config_schemas import PlotConfig
from .data_loading import get_score

ELICIT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

METHOD_LABEL_TEMPLATES = {
    "scoped": "Scoped onto {domain}",
    "scoped_recovered": "Scoped onto {domain}\n+ Recovery",
    "pgd": "Pruned for {domain}\n+ Recovery",
    "sft": "SFT on {domain}",
}


def _method_label(method_id: str, method_display: str, domain_display: str) -> str:
    template = METHOD_LABEL_TEMPLATES.get(method_id)
    if template:
        return template.format(domain=domain_display)
    return f"{method_display}\n({domain_display})"


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
        for m in config.methods:
            if not m.requires_scope_domain:
                for d in domain_ids:
                    v = get_score(df, config, model_id, m.id, elicitation_domain=d)
                    if v is not None and v > 0:
                        vanilla_scores[d] = v
                break

    elicit_color_map = {d: ELICIT_COLORS[i % len(ELICIT_COLORS)] for i, d in enumerate(domain_ids)}

    groups = []
    for scope_d in domain_ids:
        for mid in method_ids:
            method_cfg = config.get_method(mid)
            domain_display = config.domains.get_display_name(scope_d)
            label = _method_label(mid, method_cfg.display_name, domain_display)
            groups.append({"scope_domain": scope_d, "method": mid, "label": label})

    n_groups = len(groups)
    GROUP_SPACING = 0.6
    positions = []
    for i in range(n_groups):
        scope_idx = i // n_methods
        local_idx = i % n_methods
        positions.append(scope_idx * (n_methods + GROUP_SPACING) + local_idx)
    x = np.array(positions)
    BAR_WIDTH = 0.7

    fig, ax = plt.subplots(figsize=(max(14, n_groups * 1.5), 6))

    scores_per_group: list[list[tuple[str, float]]] = []
    for g in groups:
        elicit_vals = []
        for elicit_d in domain_ids:
            method_cfg = config.get_method(g["method"])
            if method_cfg.requires_scope_domain:
                s = get_score(df, config, model_id, g["method"], elicitation_domain=elicit_d, scope_domain=g["scope_domain"])
            else:
                s = get_score(df, config, model_id, g["method"], elicitation_domain=elicit_d)
            if s is not None and fig_config.relative and elicit_d in vanilla_scores:
                s = s / vanilla_scores[elicit_d]
            elicit_vals.append((elicit_d, s if s is not None else 0))
        scores_per_group.append(elicit_vals)

    legend_added = set()
    for group_idx in range(n_groups):
        elicit_vals = scores_per_group[group_idx]
        sorted_vals = sorted(elicit_vals, key=lambda t: t[1], reverse=True)
        for draw_order, (elicit_d, val) in enumerate(sorted_vals):
            lbl = config.domains.get_display_name(elicit_d) if elicit_d not in legend_added else None
            if lbl:
                legend_added.add(elicit_d)
            ax.bar(
                x[group_idx], val, BAR_WIDTH,
                color=elicit_color_map[elicit_d],
                edgecolor="white", linewidth=0.5,
                zorder=3 + draw_order,
                label=lbl,
            )

    from matplotlib.patches import FancyBboxPatch
    for scope_idx in range(n_domains):
        first_gi = scope_idx * n_methods
        last_gi = first_gi + n_methods - 1
        left = x[first_gi] - BAR_WIDTH / 2 - 0.12
        right = x[last_gi] + BAR_WIDTH / 2 + 0.12
        box_width = right - left
        max_h = 0
        for gi in range(first_gi, last_gi + 1):
            for _, val in scores_per_group[gi]:
                max_h = max(max_h, val)
        box_top = max_h * 1.05
        bbox = FancyBboxPatch(
            (left, -0.015), box_width, box_top + 0.015,
            boxstyle="round,pad=0.06",
            facecolor="none", edgecolor="#888888", linewidth=1.2,
            linestyle="-", zorder=1, clip_on=False,
        )
        ax.add_patch(bbox)
        domain_display = config.domains.get_display_name(domain_ids[scope_idx])
        ax.text((left + right) / 2, box_top + 0.04, domain_display, ha="center", va="bottom", fontsize=9, fontweight="bold", color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels([g["label"] for g in groups], fontsize=8, ha="center")
    y_label = "Relative OOD Performance" if fig_config.relative else "OOD Performance"
    ax.set_ylabel(y_label, fontsize=11)

    model_display = next((m.display_name for m in config.models if m.id == model_id), model_id)
    ax.set_title(fig_config.title_template.format(model=model_display), fontsize=13, fontweight="bold")

    handles, labels = ax.get_legend_handles_labels()
    ordered = sorted(zip(labels, handles), key=lambda t: domain_ids.index(next(d for d in domain_ids if config.domains.get_display_name(d) == t[0])))
    ax.legend([h for _, h in ordered], [l for l, _ in ordered], title="Elicitation Domain", loc="upper right", fontsize=8, title_fontsize=9)

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
