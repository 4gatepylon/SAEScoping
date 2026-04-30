"""Helpers for initializing W&B with arg > env > default precedence.

Kept deliberately small: this module only resolves init kwargs. The runner
imports `wandb` itself and calls `wandb.init(**resolved, config=...)`,
`wandb.define_metric(...)`, `wandb.log(...)`, and `wandb.finish()`.
"""

from __future__ import annotations

import os
from typing import Any, Optional

_DEFAULT_PROJECT = "saescoping"
_DEFAULT_MODE = "online"


def resolve_wandb_settings(
    *,
    project: Optional[str] = None,
    entity: Optional[str] = None,
    mode: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[str] = None,
) -> dict[str, Any]:
    """Resolve `wandb.init(...)` kwargs with precedence: explicit arg > env > default.

    Args:
        project: --wandb-project arg, falls back to $WANDB_PROJECT, then "saescoping".
        entity:  --wandb-entity arg, falls back to $WANDB_ENTITY, then omitted (W&B
                 picks the API key's default user/team).
        mode:    --wandb-mode arg, falls back to $WANDB_MODE, then "online".
        name:    --wandb-name arg, falls back to None (W&B autogenerates).
        tags:    comma-separated string from arg, parsed to a list. Empty/None → no tags.

    Returns:
        Dict suitable for `wandb.init(**settings, config=...)`. Keys absent for
        unset optional fields (entity, name, tags) — W&B then applies its own
        defaults / autogeneration.
    """
    settings: dict[str, Any] = {
        "project": project or os.environ.get("WANDB_PROJECT") or _DEFAULT_PROJECT,
        "mode": mode or os.environ.get("WANDB_MODE") or _DEFAULT_MODE,
    }
    entity_resolved = entity or os.environ.get("WANDB_ENTITY")
    if entity_resolved:
        settings["entity"] = entity_resolved
    if name:
        settings["name"] = name
    if tags:
        parsed = [t.strip() for t in tags.split(",") if t.strip()]
        if parsed:
            settings["tags"] = parsed
    return settings
