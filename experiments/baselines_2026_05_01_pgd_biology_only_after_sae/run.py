"""Subprocess wrapper for experiments/baselines_2026_04_29/sweep_wanda.py.

Calibrates Wanda on biology, judges on all 4 StemQA domains (in-domain +
3 OOD), single sparsity 0.65, PGD recovery restricted to layers 32..end
(after the deepest GemmaScope SAE on the aruna branch). Sister of
experiments/baselines_2026_04_30_pgd_after_sae/, narrowed to one sparsity
and one calibration domain.

Five caller-level knobs; everything else lives in the YAML:
    --size {full,mini}  pick config_full.yaml vs config_mini.yaml in this folder
    --model             override model_id (default from YAML: google/gemma-2-9b-it)
    --batch-size        override calibration.batch_size (set 1 for gemma-3-12b-it)
    --grad-accum        override pgd.gradient_accumulation_steps
    --device            logical device passed to sweep_wanda.py (default cuda:0;
                        physical GPU pinned by CUDA_VISIBLE_DEVICES)

PGD / W&B / LLM-judge are always enabled — that is the point of this folder.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import click

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_SCRIPT = REPO_ROOT / "experiments" / "baselines_2026_04_29" / "sweep_wanda.py"
CONFIG_DIR = Path(__file__).resolve().parent


@click.command()
@click.option(
    "--size",
    type=click.Choice(["full", "mini"]),
    default="full",
    show_default=True,
    help="Pick config_full.yaml (production) or config_mini.yaml (smoke).",
)
@click.option(
    "--model",
    "model_id",
    default=None,
    help="Override model_id, e.g. google/gemma-3-12b-it. Defaults to whatever the YAML says.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Override calibration.batch_size (use 1 for gemma-3-12b-it, 2 for gemma-2-9b-it).",
)
@click.option(
    "--grad-accum",
    "grad_accum",
    type=int,
    default=None,
    help="Override pgd.gradient_accumulation_steps.",
)
@click.option(
    "--device",
    default="cuda:0",
    show_default=True,
    help="Logical device passed to sweep_wanda.py. Pin the physical GPU via CUDA_VISIBLE_DEVICES.",
)
def main(
    size: str,
    model_id: Optional[str],
    batch_size: Optional[int],
    grad_accum: Optional[int],
    device: str,
) -> None:
    cfg = CONFIG_DIR / f"config_{size}.yaml"
    if not cfg.exists():
        raise click.ClickException(f"Config not found: {cfg}")

    cmd = [
        sys.executable,
        str(SWEEP_SCRIPT),
        "--config", str(cfg),
        "--device", device,
        "--enable-llm-judge",
        "--enable-wandb",
        "--enable-pgd",
    ]
    if model_id is not None:
        cmd += ["--model-id", model_id]
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]
    if grad_accum is not None:
        cmd += ["--gradient-accumulation-steps", str(grad_accum)]

    print("[run.py] exec:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
