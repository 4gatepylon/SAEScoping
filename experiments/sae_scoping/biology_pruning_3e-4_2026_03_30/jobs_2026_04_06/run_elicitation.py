"""run_elicitation.py

Adversarial OOD elicitation against biology-scoped models.

Tries to elicit OOD knowledge (chemistry, physics, math) through the biology
SAE filter.  The biology distribution stays active — the question is whether
longer/harder training can break through the scope.

Order: all h=3e-4 runs first, then all h=2e-4 runs.
Within each threshold: chemistry, physics, math.
Runs serially on CUDA 3 (6 runs total).

Each run: max 16000 steps or 2 epochs (whichever shorter), full dataset,
save every 4000 steps.

Usage:
    python jobs_2026_04_06/run_elicitation.py
    python jobs_2026_04_06/run_elicitation.py --gpu 3
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "jobs_2026_04_03"))
from _launcher import Job, launch_on_gpus

_ROOT = Path(__file__).resolve().parent.parent

_BIO_DIST = (
    "downloaded/deleteme_cache_bio_only/ignore_padding_True/biology/"
    "layer_31--width_16k--canonical/distribution.safetensors"
)
_WANDB_PROJECT = "sae-elicitation-ood-2026-04-06"

# (threshold_str, checkpoint_path) in execution order: 3e-4 first, then 2e-4
_BASES: list[tuple[str, str]] = [
    ("3e-4", "downloaded/model_layers_31_h0.0003/outputs/checkpoint-2000"),
    ("2e-4", "downloaded/model_layers_31_h0.0002/outputs/checkpoint-2000"),
]

# Subject order within each threshold: chemistry first, physics second, math third
_SUBJECTS = ["chemistry", "physics", "math"]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--gpu", type=int, default=3, help="GPU id. Default: 3")
def main(gpu: int) -> None:
    """Adversarial elicitation: OOD training through biology SAE filter."""
    # Validate biology distribution exists
    if not (_ROOT / _BIO_DIST).exists():
        raise click.ClickException(f"Biology distribution not found: {_BIO_DIST}")

    # Validate checkpoints exist
    for h, ckpt in _BASES:
        if not (_ROOT / ckpt).exists():
            raise click.ClickException(f"Checkpoint not found: {ckpt}")

    jobs: list[Job] = []
    for h, ckpt in _BASES:
        for subject in _SUBJECTS:
            jobs.append(Job(
                cmd=[
                    "python", "script_train_gemma9b_sae.py",
                    "--sae", "--subset", subject,
                    "-c", ckpt, "-p", _BIO_DIST, "-h", h,
                    "-b", "2", "-a", "8", "-s", "16000",
                    "--num-epochs", "2",
                    "--eval-every", "200",
                    "--utility-eval-every", "200",
                    "--biology-utility-eval-every", "200",
                    "--save-every", "4000",
                    "--output-dir", f"./outputs/elicitation_bio_h{h}/{subject}",
                    "-w", _WANDB_PROJECT,
                    "--wandb-run-name", f"elicitation/bio_h{h}/{subject}",
                ],
                label=f"elicitation: {subject} on bio h={h}  (max 16K steps / 2 epochs)",
            ))

    n_failed = launch_on_gpus(jobs, [gpu], cwd=_ROOT)
    sys.exit(n_failed)


if __name__ == "__main__":
    main()
