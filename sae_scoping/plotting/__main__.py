"""CLI entry point: python -m sae_scoping.plotting --config X --data Y --output-dir Z"""

from __future__ import annotations

from pathlib import Path

import click

from .data_loading import load_config, load_data
from .figure_in_domain_bar import plot_in_domain_bar
from .figure_ood_table import plot_ood_table


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True), help="Path to YAML config file")
@click.option("--data", "data_path", required=True, type=click.Path(exists=True), help="Path to CSV data file")
@click.option("--output-dir", "output_dir", required=True, type=click.Path(), help="Directory for output figures")
@click.option("--figure", "figures", multiple=True, default=None, help="Which figures to generate (default: all configured). Options: in_domain_bar, ood_table")
def main(config_path: str, data_path: str, output_dir: str, figures: tuple[str, ...]) -> None:
    config = load_config(config_path)
    df = load_data(data_path, config)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not figures:
        figures = []
        if config.figures.in_domain_bar:
            figures.append("in_domain_bar")
        if config.figures.ood_table:
            figures.append("ood_table")

    for model in config.models:
        click.echo(f"Generating figures for {model.display_name}...")
        for fig_name in figures:
            if fig_name == "in_domain_bar" and config.figures.in_domain_bar:
                path = plot_in_domain_bar(df, config, model.id, out)
                click.echo(f"  -> {path}")
            elif fig_name == "ood_table" and config.figures.ood_table:
                paths = plot_ood_table(df, config, model.id, out)
                for p in paths:
                    click.echo(f"  -> {p}")
            else:
                click.echo(f"  Skipping unknown/unconfigured figure: {fig_name}")

    click.echo("Done.")


if __name__ == "__main__":
    main()
