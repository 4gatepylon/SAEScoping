from __future__ import annotations

import os
from contextlib import contextmanager

@contextmanager
def wandb_context(project_name: str, run_name: str) -> None:
    import wandb
    wandb.init(project=project_name, name=run_name)
    old_project = os.environ.get("WANDB_PROJECT", None)
    old_run_name = os.environ.get("WANDB_RUN_NAME", None)
    os.environ["WANDB_PROJECT"] = project_name
    os.environ["WANDB_RUN_NAME"] = run_name
    yield
    if old_project is not None:
        os.environ["WANDB_PROJECT"] = old_project
    if old_run_name is not None:
        os.environ["WANDB_RUN_NAME"] = old_run_name
    wandb.finish()