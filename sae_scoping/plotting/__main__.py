"""CLI entry point: python -m sae_scoping.plotting --config X --data Y --output-dir Z"""

from __future__ import annotations

from pathlib import Path

import click

from .data_loading import load_config, load_data
from .figure_in_domain_bar import plot_in_domain_bar
from .figure_feature_overlap import plot_feature_overlap_scatter
from .figure_ood_bar import plot_ood_bar
from .figure_ood_table import plot_ood_table

FIGURE_REGISTRY: dict[str, tuple[str, callable]] = {
    "in_domain_bar": ("in_domain_bar", plot_in_domain_bar),
    "ood_table": ("ood_table", plot_ood_table),
    "ood_bar": ("ood_bar", plot_ood_bar),
    "feature_overlap_scatter": ("feature_overlap_scatter", plot_feature_overlap_scatter),
}

FIGURES_NEEDING_CONFIG_PATH = {"feature_overlap_scatter"}


def _available_figures(config):
    return [name for name, (attr, _) in FIGURE_REGISTRY.items() if getattr(config.figures, attr, None) is not None]


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True), help="Path to YAML config file")
@click.option("--data", "data_path", required=True, type=click.Path(exists=True), help="Path to CSV data file")
@click.option("--output-dir", "output_dir", required=True, type=click.Path(), help="Directory for output figures")
@click.option("--figure", "figures", multiple=True, default=None, help="Which figures to generate (default: all configured). Options: " + ", ".join(FIGURE_REGISTRY.keys()))
def main(config_path: str, data_path: str, output_dir: str, figures: tuple[str, ...]) -> None:
    config = load_config(config_path)
    df = load_data(data_path, config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not figures:
        figures = _available_figures(config)

    for model in config.models:
        click.echo(f"Generating figures for {model.display_name}...")
        for fig_name in figures:
            if fig_name not in FIGURE_REGISTRY:
                click.echo(f"  Skipping unknown figure: {fig_name}")
                continue
            attr, plot_fn = FIGURE_REGISTRY[fig_name]
            if getattr(config.figures, attr, None) is None:
                click.echo(f"  Skipping unconfigured figure: {fig_name}")
                continue
            kwargs = {}
            if fig_name in FIGURES_NEEDING_CONFIG_PATH:
                kwargs["config_path"] = Path(config_path)
            result = plot_fn(df, config, model.id, out, **kwargs)
            if isinstance(result, list):
                for p in result:
                    click.echo(f"  -> {p}")
            else:
                click.echo(f"  -> {result}")

    click.echo("Done.")


if __name__ == "__main__":
    main()
