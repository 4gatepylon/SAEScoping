"""
make_eval_config.py

Generate the eval_config.json for post-hoc inference & grading of all
checkpoints produced by the 2026-03-30 biology pruning experiment.

Usage:
    python make_eval_config.py              # prints to stdout
    python make_eval_config.py -o config.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click


DIST_PATH = (
    "downloaded/deleteme_cache_bio_only/ignore_padding_True/"
    "biology/layer_31--width_16k--canonical/distribution.safetensors"
)

SUBSETS = ["physics", "chemistry", "math"]
OOD_STEPS = [500, 1000, 1500, 2000]
BASE_THRESHOLDS = ["0.0002", "0.0003", "0.0004"]


def _make_jobs() -> list[dict]:
    jobs: list[dict] = []

    # 1) Vanilla OOD-trained models — no SAE during inference.
    #    Trained without SAE starting from the h0.0003 base checkpoint.
    for subset in SUBSETS:
        for step in OOD_STEPS:
            jobs.append({
                "checkpoint_path": (
                    f"outputs/vanilla/after_31/{subset}/"
                    f"checkpoint-2000/checkpoint-{step}"
                ),
                "tag": f"vanilla/{subset}/ckpt-{step}",
                "use_sae": False,
            })

    # 2) SAE OOD-trained models — SAE hooked during inference (h=0.0003).
    #    Trained with the biology-pruned SAE active at layer 31.
    for subset in SUBSETS:
        for step in OOD_STEPS:
            jobs.append({
                "checkpoint_path": (
                    f"outputs/sae_h0.0003/after_31/{subset}/"
                    f"checkpoint-2000/checkpoint-{step}"
                ),
                "tag": f"sae_h0.0003/{subset}/ckpt-{step}",
                "use_sae": True,
                "dist_path": DIST_PATH,
                "threshold": 0.0003,
            })

    # 3) Base models WITH SAE — matches how they were originally trained.
    for h in BASE_THRESHOLDS:
        jobs.append({
            "checkpoint_path": f"downloaded/model_layers_31_h{h}/outputs/checkpoint-2000",
            "tag": f"base_with_sae/h{h}",
            "use_sae": True,
            "dist_path": DIST_PATH,
            "threshold": float(h),
        })

    # 4) Base models WITHOUT SAE — raw capability baseline.
    for h in BASE_THRESHOLDS:
        jobs.append({
            "checkpoint_path": f"downloaded/model_layers_31_h{h}/outputs/checkpoint-2000",
            "tag": f"base_no_sae/h{h}",
            "use_sae": False,
        })

    return jobs


def make_config() -> dict:
    return {
        "jobs": _make_jobs(),
        "output_dir": "./eval_results",
        "max_eval_samples": 50,
        "batch_size": 4,
        "max_new_tokens": 256,
    }


_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_OUTPUT = _SCRIPT_DIR / "eval_config.json"


@click.command()
@click.option("-o", "--output", type=click.Path(path_type=Path), default=_DEFAULT_OUTPUT,
              show_default=True, help="Output path for eval config JSON.")
def main(output: Path):
    """Print or write eval_config.json for all experiment checkpoints."""
    config = make_config()
    text = json.dumps(config, indent=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    n = len(config["jobs"])
    click.echo(f"Wrote {output} ({n} jobs)", err=True)


if __name__ == "__main__":
    main()
