"""Figure 2: OOD table — scope_domain (cols) x elicitation_domain (rows)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from .config_schemas import PlotConfig
from .data_loading import get_score


def _build_ood_grid(df: pd.DataFrame, config: PlotConfig, model_id: str) -> tuple[list[list[dict | None]], list[str], list[str]]:
    """Build the grid data. Returns (grid, row_domain_ids, col_domain_ids).

    Each cell is {"ours": float, "baseline": float, "baseline_method": str} or None.
    """
    fig_config = config.figures.ood_table
    domain_ids = [d.id for d in config.domains.entries]

    vanilla_scores = {}
    for d in domain_ids:
        v = get_score(df, config, model_id, fig_config.our_method.replace("scoped_recovered", "vanilla").replace(fig_config.our_method, "vanilla"), elicitation_domain=d)
        # Actually, just get vanilla
        baseline_method_cfg = None
        for m in config.methods:
            if not m.requires_scope_domain:
                baseline_method_cfg = m
                break
        if baseline_method_cfg:
            v = get_score(df, config, model_id, baseline_method_cfg.id, elicitation_domain=d)
        if v is not None and v > 0:
            vanilla_scores[d] = v

    grid = []
    for elicit_d in domain_ids:
        row = []
        for scope_d in domain_ids:
            ours = get_score(df, config, model_id, fig_config.our_method, elicitation_domain=elicit_d, scope_domain=scope_d)
            best_baseline_score = None
            best_baseline_method = None
            for comp_mid in fig_config.comparison_methods:
                comp_cfg = config.get_method(comp_mid)
                if comp_cfg.requires_scope_domain:
                    s = get_score(df, config, model_id, comp_mid, elicitation_domain=elicit_d, scope_domain=scope_d)
                else:
                    s = get_score(df, config, model_id, comp_mid, elicitation_domain=elicit_d)
                if s is not None and (best_baseline_score is None or s < best_baseline_score):
                    best_baseline_score = s
                    best_baseline_method = comp_mid

            if ours is None and best_baseline_score is None:
                row.append(None)
            else:
                van = vanilla_scores.get(elicit_d)
                cell = {
                    "ours": ours / van if (ours is not None and van) else ours,
                    "baseline": best_baseline_score / van if (best_baseline_score is not None and van) else best_baseline_score,
                    "baseline_method": best_baseline_method,
                }
                row.append(cell)
        grid.append(row)

    return grid, domain_ids, domain_ids


def render_ood_table_latex(df: pd.DataFrame, config: PlotConfig, model_id: str, output_dir: Path) -> Path | None:
    if not config.output.latex:
        return None
    fig_config = config.figures.ood_table
    grid, row_ids, col_ids = _build_ood_grid(df, config, model_id)

    col_labels = [config.domains.get_display_name(d) for d in col_ids]
    row_labels = [config.domains.get_display_name(d) for d in row_ids]

    lines = []
    lines.append(r"\begin{tabular}{l|" + "c" * len(col_ids) + "}")
    lines.append(r"\hline")
    lines.append(r"\textbf{Elicit $\downarrow$ / Scope $\rightarrow$} & " + " & ".join(rf"\textbf{{{c}}}" for c in col_labels) + r" \\")
    lines.append(r"\hline")

    for i, row_label in enumerate(row_labels):
        cells = []
        for j, cell in enumerate(grid[i]):
            if cell is None:
                cells.append("--")
            else:
                ours_str = f"{cell['ours']:.2f}" if cell["ours"] is not None else "--"
                base_str = f"{cell['baseline']:.2f}" if cell["baseline"] is not None else "--"
                is_diagonal = (row_ids[i] == col_ids[j])
                if is_diagonal:
                    cells.append(rf"\cellcolor{{gray!20}}{ours_str} / {base_str}")
                elif cell["ours"] is not None and cell["baseline"] is not None:
                    we_win = cell["ours"] <= cell["baseline"]
                    color = "green!20" if we_win else "yellow!20"
                    bold_ours = rf"\textbf{{{ours_str}}}" if we_win else ours_str
                    bold_base = base_str if we_win else rf"\textbf{{{base_str}}}"
                    cells.append(rf"\cellcolor{{{color}}}{bold_ours} / {bold_base}")
                else:
                    cells.append(f"{ours_str} / {base_str}")

        lines.append(rf"{row_label} & " + " & ".join(cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    out_path = output_dir / f"ood_table_{model_id}.tex"
    out_path.write_text("\n".join(lines))
    return out_path


def render_ood_table_png(df: pd.DataFrame, config: PlotConfig, model_id: str, output_dir: Path) -> Path:
    fig_config = config.figures.ood_table
    grid, row_ids, col_ids = _build_ood_grid(df, config, model_id)

    col_labels = [config.domains.get_display_name(d) for d in col_ids]
    row_labels = [config.domains.get_display_name(d) for d in row_ids]
    n_rows, n_cols = len(row_ids), len(col_ids)

    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.8), max(4, n_rows * 1.2)))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_xticklabels(col_labels, fontsize=9, fontweight="bold")
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(row_labels, fontsize=9, fontweight="bold")
    ax.tick_params(length=0)

    GREEN = "#c6efce"
    YELLOW = "#fff2cc"
    GRAY = "#e0e0e0"
    WHITE = "#ffffff"

    for i in range(n_rows):
        for j in range(n_cols):
            cell = grid[i][j]
            is_diagonal = (row_ids[i] == col_ids[j])

            if cell is None:
                color = WHITE
                text = "--"
                fontweight = "normal"
                fontcolor = "gray"
            else:
                ours_val = cell["ours"]
                base_val = cell["baseline"]
                ours_str = f"{ours_val:.2f}" if ours_val is not None else "--"
                base_str = f"{base_val:.2f}" if base_val is not None else "--"

                if is_diagonal:
                    color = GRAY
                elif ours_val is not None and base_val is not None:
                    color = GREEN if ours_val <= base_val else YELLOW
                else:
                    color = WHITE

                text = f"{ours_str} / {base_str}"
                fontweight = "bold"
                fontcolor = "black"

            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="gray", linewidth=0.5)
            ax.add_patch(rect)
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=8, fontweight=fontweight, color=fontcolor)

    for i in range(n_rows + 1):
        ax.axhline(y=i, color="gray", linewidth=0.5)
    for j in range(n_cols + 1):
        ax.axvline(x=j, color="gray", linewidth=0.5)

    model_display = next((m.display_name for m in config.models if m.id == model_id), model_id)
    ax.set_title(fig_config.title_template.format(model=model_display), fontsize=12, fontweight="bold", pad=20)

    our_method_display = config.get_method(fig_config.our_method).display_name
    comp_display = " / ".join(config.get_method(m).display_name for m in fig_config.comparison_methods)
    ax.text(0.5, -0.02, f"Cell: {our_method_display} / best of ({comp_display}). Bold = lower. Green = ours wins.",
            transform=ax.transAxes, ha="center", fontsize=7, color="gray")

    ax.set_aspect("equal")
    fig.tight_layout()
    out_path = output_dir / f"ood_table_{model_id}.png"
    fig.savefig(out_path, dpi=config.output.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ood_table(df: pd.DataFrame, config: PlotConfig, model_id: str, output_dir: Path) -> list[Path]:
    paths = []
    paths.append(render_ood_table_png(df, config, model_id, output_dir))
    tex = render_ood_table_latex(df, config, model_id, output_dir)
    if tex:
        paths.append(tex)
    return paths
