"""Figure 1: In-domain bar plot showing relative performance per domain."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config_schemas import PlotConfig
from .data_loading import get_score


def plot_in_domain_bar(df: pd.DataFrame, config: PlotConfig, model_id: str, output_dir: Path) -> Path:
    fig_config = config.figures.in_domain_bar
    if fig_config is None:
        raise ValueError("in_domain_bar not configured")

    domain_ids = [d.id for d in config.domains.entries]
    domain_labels = [config.domains.get_display_name(d) for d in domain_ids]
    method_ids = fig_config.methods
    baseline_id = fig_config.baseline_method

    vanilla_scores = {}
    for d in domain_ids:
        v = get_score(df, config, model_id, baseline_id, elicitation_domain=d)
        if v is None:
            raise ValueError(f"Missing baseline ({baseline_id}) score for model={model_id}, domain={d}")
        if v == 0:
            raise ValueError(f"Baseline score is 0 for model={model_id}, domain={d} — cannot compute relative performance")
        vanilla_scores[d] = v

    relative_scores: dict[str, list[float | None]] = {}
    for mid in method_ids:
        scores = []
        for d in domain_ids:
            method_cfg = config.get_method(mid)
            if method_cfg.requires_scope_domain:
                s = get_score(df, config, model_id, mid, elicitation_domain=d, scope_domain=d)
            else:
                s = get_score(df, config, model_id, mid, elicitation_domain=d)
            if s is not None:
                scores.append(s / vanilla_scores[d])
            else:
                scores.append(None)
        relative_scores[mid] = scores

    n_domains = len(domain_ids)
    n_methods = len(method_ids)
    x = np.arange(n_domains)
    bar_width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_domains * 1.5), 5))

    for i, mid in enumerate(method_ids):
        method_cfg = config.get_method(mid)
        vals = relative_scores[mid]
        positions = x + (i - n_methods / 2 + 0.5) * bar_width
        plot_vals = [v if v is not None else 0 for v in vals]
        bars = ax.bar(
            positions,
            plot_vals,
            bar_width,
            label=method_cfg.display_name,
            color=method_cfg.color,
            edgecolor="white",
            linewidth=0.5,
        )
        for j, v in enumerate(vals):
            if v is None:
                ax.text(positions[j], 0.02, "N/A", ha="center", va="bottom", fontsize=7, color="gray")

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.0, label=config.get_method(baseline_id).display_name)

    ax.set_xticks(x)
    ax.set_xticklabels(domain_labels, fontsize=10)
    ax.set_ylabel("Relative In-Domain Performance", fontsize=11)
    model_display = next((m.display_name for m in config.models if m.id == model_id), model_id)
    ax.set_title(fig_config.title_template.format(model=model_display), fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / f"in_domain_bar_{model_id}.png"
    fig.savefig(out_path, dpi=config.output.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
