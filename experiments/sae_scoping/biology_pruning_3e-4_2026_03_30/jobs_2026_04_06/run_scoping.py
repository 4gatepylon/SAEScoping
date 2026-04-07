"""run_scoping.py

Train domain-scoped models FROM SCRATCH on google/gemma-2-9b-it using
domain-matched SAE distributions.  This is the same process that originally
produced the biology-scoped models in downloaded/.

Runs serially on CUDA 0 (7 runs total):
  - Physics  × h=1e-4, 2e-4, 3e-4   (physics SAE distribution)
  - Math     × h=1e-4, 2e-4, 3e-4   (math SAE distribution)
  - Chemistry × h=2e-4               (chemistry SAE distribution)

Each run: max 2000 steps or 1 epoch, save every 1000 steps.

Usage:
    python jobs_2026_04_06/run_scoping.py
    python jobs_2026_04_06/run_scoping.py --gpu 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "jobs_2026_04_03"))
from _launcher import Job, launch_on_gpus

_ROOT = Path(__file__).resolve().parent.parent

_BASE_MODEL = "google/gemma-2-9b-it"
_WANDB_PROJECT = "sae-scoping-2026-04-06"

_DISTRIBUTIONS: dict[str, str] = {
    "physics": (
        "distributions_cache/ignore_padding_True/physics/"
        "layer_31--width_16k--canonical/distribution.safetensors"
    ),
    "math": (
        "distributions_cache/ignore_padding_True/math/"
        "layer_31--width_16k--canonical/distribution.safetensors"
    ),
    "chemistry": (
        "distributions_cache/ignore_padding_True/chemistry/"
        "layer_31--width_16k--canonical/distribution.safetensors"
    ),
}

# (subject, threshold) pairs in execution order
_RUNS: list[tuple[str, str]] = [
    ("physics", "1e-4"),
    ("physics", "2e-4"),
    ("physics", "3e-4"),
    ("math", "1e-4"),
    ("math", "2e-4"),
    ("math", "3e-4"),
    ("chemistry", "2e-4"),
]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--gpu", type=int, default=0, help="GPU id. Default: 0")
def main(gpu: int) -> None:
    """Train domain-scoped models from base gemma-2-9b-it with domain-matched SAEs."""
    # Validate distributions exist
    for subject, dist_path in _DISTRIBUTIONS.items():
        if not (_ROOT / dist_path).exists():
            raise click.ClickException(
                f"{subject} distribution not found: {dist_path}\n"
                "Run jobs_2026_04_03/run_distributions.sh first."
            )

    jobs: list[Job] = []
    for subject, h in _RUNS:
        dist_path = _DISTRIBUTIONS[subject]
        jobs.append(Job(
            cmd=[
                "python", "script_train_gemma9b_sae.py",
                "--sae", "--subset", subject,
                "-c", _BASE_MODEL, "-p", dist_path, "-h", h,
                "-b", "2", "-a", "8", "-s", "2000",
                "--eval-every", "200",
                "--utility-eval-every", "200",
                "--save-every", "1000",
                "--output-dir", f"./outputs/scoped/{subject}/h{h}",
                "-w", _WANDB_PROJECT,
                "--wandb-run-name", f"scoped/{subject}/h{h}",
            ],
            label=f"scoping: {subject}  h={h}  (from base gemma-2-9b-it)",
        ))

    n_failed = launch_on_gpus(jobs, [gpu], cwd=_ROOT)
    sys.exit(n_failed)


if __name__ == "__main__":
    main()
