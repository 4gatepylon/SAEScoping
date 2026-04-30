"""Click CLI utilities for YAML config loading."""

from __future__ import annotations

import sys
from pathlib import Path

import click
import yaml

_AUTO_CONFIG_NAMES = ("config.yaml", "config.yml")


def load_yaml_config(ctx: click.Context, param: click.Parameter, value: str | None) -> None:
    """Click callback that loads a YAML file into ctx.default_map.

    If no --config is passed, auto-detects config.yaml next to the script.
    Use with: @click.option("--config", is_eager=True, expose_value=False, callback=load_yaml_config, type=click.Path(exists=True))
    """
    if value is None:
        script_dir = Path(sys.argv[0]).resolve().parent
        for name in _AUTO_CONFIG_NAMES:
            candidate = script_dir / name
            if candidate.exists():
                value = str(candidate)
                print(f"[config] Auto-detected {candidate}")
                break
    if value is None:
        return
    with open(value) as f:
        ctx.default_map = yaml.safe_load(f) or {}
