"""Click CLI utilities for YAML config loading."""

from __future__ import annotations

import click
import yaml


def load_yaml_config(ctx: click.Context, param: click.Parameter, value: str | None) -> None:
    """Click callback that loads a YAML file into ctx.default_map.

    Use with: @click.option("--config", is_eager=True, expose_value=False, callback=load_yaml_config, type=click.Path(exists=True))
    """
    if value is None:
        return
    with open(value) as f:
        ctx.default_map = yaml.safe_load(f) or {}
