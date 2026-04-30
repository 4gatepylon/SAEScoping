"""Helpers for organizing per-run artifact directories.

Layout convention (callers decide what to write into each step):
    $ARTIFACTS_ROOT/outputs/{run_id}/
      metadata.json          # run-level metadata (one per run)
      step_NNN/
        step_metadata.json   # per-step facts (sparsity, loss, ...)
        judgements.jsonl     # streamed via JsonlSink
        inference.jsonl      # streamed via JsonlSink
        scores.json          # final aggregated scores

`run_id` is `YYYY-MM-DD_HHMMSS_xxxxxxxx` (uuid4 hex prefix) — sortable
lexicographically by start time, collision-resistant for parallel starts.

Crash semantics for streaming logs (judgements/inference): see `JsonlSink`
in sae_scoping.evaluation.utils. Every flushed row survives a Python crash
or SIGKILL; durability against kernel panic / power loss is not guaranteed.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

_ARTIFACTS_ENV = "SAESCOPING_ARTIFACTS_LOCATION"


def resolve_artifacts_root(arg_value: Optional[str] = None) -> Path:
    """Resolve the artifacts root with precedence: explicit arg > env > '.'."""
    return Path(arg_value or os.environ.get(_ARTIFACTS_ENV, "."))


def make_run_id() -> str:
    return f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def make_run_dir(artifacts_root: Path, run_id: str) -> Path:
    """Create $artifacts_root/outputs/{run_id}/ and return it."""
    run_dir = artifacts_root / "outputs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_step_dir(run_dir: Path, step_idx: int) -> Path:
    """Create $run_dir/step_NNN/ and return it (3-digit zero-padded)."""
    step_dir = run_dir / f"step_{step_idx:03d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    return step_dir


def get_git_sha(cwd: Optional[Path] = None) -> Optional[str]:
    """Best-effort `git rev-parse HEAD`. Returns None if not a git repo or git missing."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except FileNotFoundError:
        return None
