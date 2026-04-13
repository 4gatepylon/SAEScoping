"""run_distributions.py

Compute SAE neuron firing-rate distributions for chemistry and physics using
the biology-trained checkpoint (h=3e-4).

Usage:
    python jobs_2026_04_03/run_distributions.py GPU [GPU...]

Examples:
    python jobs_2026_04_03/run_distributions.py 1 2   # two GPUs, one domain each
    python jobs_2026_04_03/run_distributions.py 1     # single GPU, sequential

Outputs:
    distributions_cache/ignore_padding_True/chemistry/layer_31--width_16k--canonical/distribution.safetensors
    distributions_cache/ignore_padding_True/physics/layer_31--width_16k--canonical/distribution.safetensors
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))
from _launcher import Job, launch_on_gpus

_ROOT = Path(__file__).resolve().parent.parent   # biology_pruning folder

_CKPT = "downloaded/model_layers_31_h0.0003/outputs/checkpoint-2000"
_DOMAINS = ["chemistry", "physics", "math"]


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("gpus", nargs=-1, required=True, type=int, metavar="GPU [GPU...]")
def main(gpus: tuple[int, ...]) -> None:
    """Compute SAE distributions for chemistry and physics on the biology-trained model."""
    jobs = [
        Job(
            cmd=[
                "python", "script_cache_distributions.py",
                "--datasets", domain,
                "--checkpoint", _CKPT,
                "--layer", "31",
                "--ignore-padding",
                "--n-samples", "2000",
                "--batch-size", "4",
                "--output-dir", "./distributions_cache",
            ],
            label=f"distributions: {domain}",
        )
        for domain in _DOMAINS
    ]

    n_failed = launch_on_gpus(jobs, list(gpus), cwd=_ROOT)

    if n_failed == 0:
        print(
            "\nNext: inspect threshold curves and choose thresholds for recovery training:\n"
            "  python jobs_2026_04_03/find_threshold_2026_04_03.py\n"
            "\nThen launch recovery training:\n"
            "  bash jobs_2026_04_03/run_recovery.sh GPU ... "
            "--chem-thresholds 3e-4 --phys-thresholds 3e-4"
        )

    sys.exit(n_failed)


if __name__ == "__main__":
    main()
