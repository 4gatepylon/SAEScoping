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
from pathlib import Path

import click


# Path to the biology SAE neuron distribution applied as the SAE filter during inference.
_BIOLOGY_SAE_DIST_PATH = (
    "downloaded/deleteme_cache_bio_only/ignore_padding_True/"
    "biology/layer_31--width_16k--canonical/distribution.safetensors"
)

# Maps each OOD elicitation subject to the two things we want to measure:
#   1. the subject itself  — did elicitation succeed? (the safety-case question)
#   2. biology             — did in-domain capability degrade? (side-effect check)
_ELICITATION_SUBJECT_EVALS: dict[str, list[str]] = {
    "physics":   ["physics",   "biology"],
    "chemistry": ["chemistry", "biology"],
    "math":      ["math",      "biology"],
}

# Base models are reference points, so we evaluate them on everything.
_BASE_MODEL_EVAL_SUBJECTS = ["physics", "chemistry", "math", "biology"]

# Checkpoint steps saved during OOD elicitation training.
_ELICITATION_CKPT_STEPS = [500, 1000, 1500, 2000]

# SAE pruning thresholds used for the base biology-scoped model variants.
_BASE_SAE_THRESHOLDS = ["0.0002", "0.0003", "0.0004"]


def _make_jobs() -> list[dict]:
    jobs: list[dict] = []

    # 1) Vanilla OOD-elicited models — no SAE during inference.
    #    Trained without SAE active, starting from the h=0.0003 biology-scoped base.
    for subject, eval_subjects in _ELICITATION_SUBJECT_EVALS.items():
        for step in _ELICITATION_CKPT_STEPS:
            jobs.append({
                "checkpoint_path": (
                    f"outputs/vanilla/after_31/{subject}/"
                    f"checkpoint-2000/checkpoint-{step}"
                ),
                "tag": f"vanilla/{subject}/ckpt-{step}",
                "use_sae": False,
                "eval_subsets": eval_subjects,
            })

    # 2) SAE-active OOD-elicited models — biology SAE hooked at layer 31 (h=0.0003).
    #    Trained with the biology-pruned SAE active; tests elicitation difficulty.
    for subject, eval_subjects in _ELICITATION_SUBJECT_EVALS.items():
        for step in _ELICITATION_CKPT_STEPS:
            jobs.append({
                "checkpoint_path": (
                    f"outputs/sae_h0.0003/after_31/{subject}/"
                    f"checkpoint-2000/checkpoint-{step}"
                ),
                "tag": f"sae_h0.0003/{subject}/ckpt-{step}",
                "use_sae": True,
                "dist_path": _BIOLOGY_SAE_DIST_PATH,
                "threshold": 0.0003,
                "eval_subsets": eval_subjects,
            })

    # 3) Base biology-scoped models WITH SAE active — matches original training setup.
    for h in _BASE_SAE_THRESHOLDS:
        jobs.append({
            "checkpoint_path": f"downloaded/model_layers_31_h{h}/outputs/checkpoint-2000",
            "tag": f"base_with_sae/h{h}",
            "use_sae": True,
            "dist_path": _BIOLOGY_SAE_DIST_PATH,
            "threshold": float(h),
            "eval_subsets": _BASE_MODEL_EVAL_SUBJECTS,
        })

    # 4) Base biology-scoped models WITHOUT SAE — raw capability baseline.
    for h in _BASE_SAE_THRESHOLDS:
        jobs.append({
            "checkpoint_path": f"downloaded/model_layers_31_h{h}/outputs/checkpoint-2000",
            "tag": f"base_no_sae/h{h}",
            "use_sae": False,
            "eval_subsets": _BASE_MODEL_EVAL_SUBJECTS,
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
