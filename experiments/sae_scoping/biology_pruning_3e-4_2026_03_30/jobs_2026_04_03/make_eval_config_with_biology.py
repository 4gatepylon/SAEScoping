"""make_eval_config_with_biology.py

Generate an eval config that adds "biology" to every job's eval_subsets on top
of the existing OOD subsets (physics, chemistry, math).

Running generate_and_grade.py with this config + its built-in caching means only
the missing biology results will be computed; all existing OOD results are skipped.

Usage:
    python make_eval_config_with_biology.py              # prints path, writes default location
    python make_eval_config_with_biology.py -o out.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

# Reuse the existing job list from jobs_inference_2026_04_02
_INFERENCE_DIR = Path(__file__).resolve().parent.parent / "jobs_inference_2026_04_02"
sys.path.insert(0, str(_INFERENCE_DIR))
from make_eval_config import _make_jobs  # noqa: E402

_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "eval_config_with_biology.json"


def make_config() -> dict:
    jobs = _make_jobs()
    # Add biology to every job's eval_subsets
    for job in jobs:
        subsets = list(job.get("eval_subsets", ["physics", "chemistry", "math"]))
        if "biology" not in subsets:
            subsets.append("biology")
        job["eval_subsets"] = subsets
    return {
        "jobs": jobs,
        "output_dir": "./eval_results",
        "max_eval_samples": 50,
        "batch_size": 4,
        "max_new_tokens": 256,
    }


@click.command()
@click.option("-o", "--output", type=click.Path(path_type=Path), default=_DEFAULT_OUTPUT,
              show_default=True, help="Output path for eval config JSON.")
def main(output: Path) -> None:
    """Write eval_config_with_biology.json (adds biology subset to all existing jobs)."""
    config = make_config()
    text = json.dumps(config, indent=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    n = len(config["jobs"])
    click.echo(f"Wrote {output} ({n} jobs, each with biology in eval_subsets)", err=True)


if __name__ == "__main__":
    main()
