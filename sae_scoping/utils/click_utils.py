"""Click CLI utilities for YAML config loading."""

from __future__ import annotations

import sys
from typing import Callable, Literal
from pathlib import Path

import click
import yaml

_AUTO_CONFIG_NAMES = ("config.yaml", "config.yml")


def parse_comma_separated_floats(
    value: str | list | tuple | None,
    default: list[float] | None = None,
    sort: bool = True,
    sort_key: Callable[[float], object] | None = None,
    duplicates: Literal["raise", "dedup", "none"] = "raise",
) -> list[float]:
    """Parse a comma-separated string or list into a list of floats.

    Handles CLI strings ("0.2,0.4,0.6") and YAML lists ([0.2, 0.4, 0.6]).
    Returns *default* (or []) when *value* is None/empty.

    Duplicate handling is applied first, then sorting. When *sort* is True
    the output is ascending; call .reverse() on the result for descending.

    Args:
        value: Raw input (string, list/tuple of numbers, or None).
        default: Fallback when *value* is None/empty.
        sort: Sort ascending. Pass *sort_key* to customise the ordering.
        sort_key: Key function forwarded to sorted() (ignored if sort=False).
        duplicates: "raise" (default) raises on dupes, "dedup" keeps first
            occurrence, "none" allows duplicates through unchanged.
    """
    if value is None or value == "":
        return default if default is not None else []
    raw = [float(x) for x in (value.split(",") if isinstance(value, str) else value)]
    if duplicates != "none":
        seen: set[float] = set()
        deduped: list[float] = []
        for x in raw:
            if x in seen:
                if duplicates == "raise":
                    raise ValueError(f"Duplicate value: {x}")
                continue
            seen.add(x)
            deduped.append(x)
        raw = deduped
    return sorted(raw, key=sort_key) if sort else raw


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
