"""run_elicitation.py

SAE elicitation training: chemistry and physics on the biology-scoped model.

The biology neuron distribution is kept active — this tests whether OOD
knowledge can be elicited despite the biology-specific SAE filter.
Training runs for 8000 steps (4× the original 2000-step baseline).

Both OOD utility (utility_eval/ood/judge) and biology utility
(utility_eval/biology/judge) are logged as separate W&B series so we can
detect side-effects on biology capability.

Usage:
    python jobs_2026_04_03/run_elicitation.py GPU [GPU...]

Examples:
    python jobs_2026_04_03/run_elicitation.py 1 2   # two GPUs, one subject each
    python jobs_2026_04_03/run_elicitation.py 1     # single GPU, sequential
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))
from _launcher import Job, launch_on_gpus

_ROOT = Path(__file__).resolve().parent.parent

_CKPT = "downloaded/model_layers_31_h0.0003/outputs/checkpoint-2000"
_DIST = (
    "downloaded/deleteme_cache_bio_only/ignore_padding_True/biology/"
    "layer_31--width_16k--canonical/distribution.safetensors"
)
_WANDB_PROJECT = "sae-elicitation-ood-2026-04-03"
_SUBSETS = ["chemistry", "physics"]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("gpus", nargs=-1, required=True, type=int, metavar="GPU [GPU...]")
def main(gpus: tuple[int, ...]) -> None:
    """SAE elicitation training on chemistry and physics using the biology-scoped model."""
    jobs = [
        Job(
            cmd=[
                "python", "script_train_gemma9b_sae.py",
                "--sae", "--subset", subset,
                "-c", _CKPT, "-p", _DIST, "-h", "0.0003",
                "-b", "2", "-a", "8", "-s", "8000",
                "--eval-every", "200",
                "--utility-eval-every", "200",
                "--biology-utility-eval-every", "200",
                "--save-every", "1000",
                "--output-dir", f"./outputs/sae_bio_dist_8k/after_31/{subset}/checkpoint-2000",
                "-w", _WANDB_PROJECT,
                "--wandb-run-name", f"sae_bio_dist_8k/{subset}/h0.0003/ckpt-2000",
            ],
            label=f"SAE elicitation: {subset}  (8000 steps)",
        )
        for subset in _SUBSETS
    ]

    n_failed = launch_on_gpus(jobs, list(gpus), cwd=_ROOT)
    sys.exit(n_failed)


if __name__ == "__main__":
    main()
