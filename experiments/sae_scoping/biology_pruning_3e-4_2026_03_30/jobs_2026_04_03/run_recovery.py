"""run_recovery.py

Recovery training: fine-tune the biology-scoped model on chemistry/physics using
a DOMAIN-MATCHED SAE distribution.  This lets us compare how easily the model
recovers OOD performance when given the "right" neurons vs. the biology-locked
ones used in run_elicitation.py.

Prerequisites:
    1. Run run_distributions.sh/py to produce distributions_cache/.
    2. Run find_threshold_2026_04_03.py to choose thresholds.

Usage:
    python jobs_2026_04_03/run_recovery.py GPU [GPU...] \\
        --chem-thresholds 3e-4[,4e-4,...] \\
        --phys-thresholds 3e-4[,4e-4,...]

Examples:
    # 2 GPUs, 2 thresholds per domain → 4 jobs (2 per GPU sequentially)
    python jobs_2026_04_03/run_recovery.py 1 2 \\
        --chem-thresholds 3e-4,4e-4 --phys-thresholds 3e-4,4e-4

    # Single threshold per domain, one GPU
    python jobs_2026_04_03/run_recovery.py 1 \\
        --chem-thresholds 3e-4 --phys-thresholds 3e-4
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))
from _launcher import Job, launch_on_gpus

_ROOT = Path(__file__).resolve().parent.parent

_CKPT = "downloaded/model_layers_31_h0.0003/outputs/checkpoint-2000"
_CHEM_DIST = (
    "distributions_cache/ignore_padding_True/chemistry/"
    "layer_31--width_16k--canonical/distribution.safetensors"
)
_PHYS_DIST = (
    "distributions_cache/ignore_padding_True/physics/"
    "layer_31--width_16k--canonical/distribution.safetensors"
)
_WANDB_PROJECT = "sae-elicitation-ood-2026-04-03"


def _parse_thresholds(s: str) -> list[str]:
    return [t.strip() for t in s.split(",") if t.strip()]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("gpus", nargs=-1, required=True, type=int, metavar="GPU [GPU...]")
@click.option(
    "--chem-thresholds", required=True,
    help="Comma-separated chemistry thresholds, e.g. 3e-4,4e-4",
)
@click.option(
    "--phys-thresholds", required=True,
    help="Comma-separated physics thresholds, e.g. 3e-4,4e-4",
)
def main(
    gpus: tuple[int, ...],
    chem_thresholds: str,
    phys_thresholds: str,
) -> None:
    """Recovery training on chemistry/physics with domain-matched SAE distributions."""
    chem_hs = _parse_thresholds(chem_thresholds)
    phys_hs = _parse_thresholds(phys_thresholds)

    # Validate distributions exist before queuing jobs
    for label, dist_path in [("chemistry", _CHEM_DIST), ("physics", _PHYS_DIST)]:
        if not (_ROOT / dist_path).exists():
            raise click.ClickException(
                f"{label} distribution not found: {dist_path}\n"
                "Run jobs_2026_04_03/run_distributions.sh first."
            )

    jobs: list[Job] = []
    for subset, hs, dist in [
        ("chemistry", chem_hs, _CHEM_DIST),
        ("physics",   phys_hs, _PHYS_DIST),
    ]:
        for h in hs:
            tag = f"{subset}_dist_h{h}"
            jobs.append(Job(
                cmd=[
                    "python", "script_train_gemma9b_sae.py",
                    "--sae", "--subset", subset,
                    "-c", _CKPT, "-p", dist, "-h", h,
                    "-b", "2", "-a", "8", "-s", "8000",
                    "--eval-every", "200",
                    "--utility-eval-every", "200",
                    "--biology-utility-eval-every", "200",
                    "--save-every", "1000",
                    "--output-dir", f"./outputs/recovery_{tag}/after_31/{subset}/checkpoint-2000",
                    "-w", _WANDB_PROJECT,
                    "--wandb-run-name", f"recovery/{tag}/ckpt-2000",
                ],
                label=f"recovery: {subset}  h={h}",
            ))

    n_failed = launch_on_gpus(jobs, list(gpus), cwd=_ROOT)
    sys.exit(n_failed)


if __name__ == "__main__":
    main()
