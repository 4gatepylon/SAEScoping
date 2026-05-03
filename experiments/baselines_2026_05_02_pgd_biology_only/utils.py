"""Shared helpers for callee scripts (calibrate, pgd_or_elicit)."""

from __future__ import annotations

import os
from pathlib import Path

import click

from interface import StepSpec


def resolve_artifacts_root(spec: StepSpec) -> Path:
    """Return the experiment artifacts directory, creating it if needed."""
    base = os.environ.get("SAESCOPING_ARTIFACTS_LOCATION")
    if not base:
        raise click.ClickException("SAESCOPING_ARTIFACTS_LOCATION not set.")
    root = Path(base) / spec.artifacts_subdir
    root.mkdir(parents=True, exist_ok=True)
    return root


def maybe_init_wandb(
    spec: StepSpec,
    artifacts_root: Path,
    *,
    name: str,
    config: dict,
    no_wandb: bool = False,
    tags: list[str] | None = None,
):
    """Init a W&B run if enabled, else return None."""
    if not spec.wandb.enabled or no_wandb:
        return None
    import wandb

    os.environ["WANDB_DIR"] = str(artifacts_root / "wandb")
    return wandb.init(project=spec.wandb.project, name=name, config=config, tags=tags)
