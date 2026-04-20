"""
Parallel sweep launcher — distributes jobs across GPUs.

Each job = one (method, model) pair running sweep_sparsity.py on a single GPU.
Jobs are queued and assigned to GPUs as they become free.

Usage:
  # Run all methods on gemma-2-2b-it across GPUs 0,2,7:
  python launch_sweeps.py --gpus 0,2,7 --model google/gemma-2-2b-it

  # Run specific methods on multiple models:
  python launch_sweeps.py --gpus 0,2,3,7 \
      --methods wanda,random,sparse_llm \
      --models google/gemma-2-2b-it,google/gemma-2-9b-it

  # All methods, all models (will queue and run as GPUs free up):
  python launch_sweeps.py --gpus 0,2,3,7 --all

  # Dry run (show what would be launched):
  python launch_sweeps.py --gpus 0,2,7 --all --dry-run

  # With custom sparsity levels:
  python launch_sweeps.py --gpus 0,2 --model google/gemma-2-2b-it \
      --sparsity-levels 0.1,0.3,0.5,0.7,0.9

  # Skip LLM judge (faster):
  python launch_sweeps.py --gpus 0,2 --all --no-judge
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click

ALL_METHODS = ["wanda", "random", "sparse_llm", "taylor", "gradient"]
METHODS_NO_SALIENCY = ["wanda", "random", "sparse_llm"]  # Don't need pre-computed maps

ALL_MODELS = [
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-3-12b-it",
]

# Per-model tuning for memory constraints.
# saliency_path_template uses {subset} placeholder filled at runtime.
MODEL_CONFIGS = {
    "google/gemma-2-2b-it": {
        "n_calibration": 128,
        "max_seq_len": 1024,
        "sparse_llm_iterations": 4,
        "sparse_llm_n_calibration": 64,
        "saliency_path_template": "./saliency_maps/gemma2_2b_{subset}/ema_grads.safetensors",
    },
    "google/gemma-2-9b-it": {
        "n_calibration": 128,
        "max_seq_len": 1024,
        "sparse_llm_iterations": 4,
        "sparse_llm_n_calibration": 32,
        "saliency_path_template": "./saliency_maps/gemma2_9b_{subset}/ema_grads.safetensors",
    },
    "google/gemma-3-12b-it": {
        "n_calibration": 128,
        "max_seq_len": 1024,
        "sparse_llm_iterations": 2,
        "sparse_llm_n_calibration": 16,
        "saliency_path_template": "./saliency_maps/gemma3_12b_{subset}/ema_grads.safetensors",
    },
}


@dataclass
class Job:
    method: str
    model: str
    gpu_id: int = -1  # Assigned at launch time
    extra_args: list[str] | None = None

    @property
    def name(self) -> str:
        model_slug = self.model.replace("/", "--")
        return f"{self.method}/{model_slug}"


def run_job(job: Job, common_args: list[str]) -> tuple[str, int, str]:
    """Run a single sweep job. Returns (job_name, return_code, output_snippet)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)

    cfg = MODEL_CONFIGS.get(job.model, MODEL_CONFIGS["google/gemma-2-2b-it"])

    cmd = [
        sys.executable, str(Path(__file__).parent / "sweep_sparsity.py"),
        "--method", job.method,
        "--model", job.model,
        "--device", "cuda:0",  # Always 0 since CUDA_VISIBLE_DEVICES handles mapping
    ]

    # Method-specific calibration settings
    if job.method == "sparse_llm":
        cmd += ["--n-calibration", str(cfg["sparse_llm_n_calibration"])]
        cmd += ["--sparse-llm-iterations", str(cfg["sparse_llm_iterations"])]
    else:
        cmd += ["--n-calibration", str(cfg["n_calibration"])]

    cmd += ["--max-seq-len", str(cfg["max_seq_len"])]
    cmd += common_args

    if job.extra_args:
        cmd += job.extra_args

    print(f"[GPU {job.gpu_id}] Launching: {job.name}")
    print(f"  CMD: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=7200,  # 2hr timeout
        )
        # Get last few lines of output for summary
        output_lines = (result.stdout + result.stderr).strip().split("\n")
        snippet = "\n".join(output_lines[-5:])
        return job.name, result.returncode, snippet
    except subprocess.TimeoutExpired:
        return job.name, -1, "TIMEOUT (2h)"
    except Exception as e:
        return job.name, -1, str(e)


@click.command()
@click.option("--gpus", required=True, help="Comma-separated GPU IDs (e.g. 0,2,3,7)")
@click.option("--methods", default=None, help="Comma-separated methods (default: wanda,random,sparse_llm)")
@click.option("--models", default=None, help="Comma-separated model IDs")
@click.option("--model", default=None, help="Single model (shorthand for --models)")
@click.option("--all", "run_all", is_flag=True, help="All methods x all models")
@click.option("--dataset-name", default="4gate/StemQAMixture")
@click.option("--dataset-subset", default="biology")
@click.option("--sparsity-levels", default=None, help="Custom sparsity levels")
@click.option("--saliency-path", default=None, help="For taylor/gradient methods")
@click.option("--no-judge", is_flag=True, help="Skip LLM judge")
@click.option("--dry-run", is_flag=True, help="Print jobs without running")
@click.option("--wandb-project", default="sae-scoping-baselines")
def main(
    gpus, methods, models, model, run_all, dataset_name, dataset_subset,
    sparsity_levels, saliency_path, no_judge, dry_run, wandb_project,
):
    gpu_ids = [int(g.strip()) for g in gpus.split(",")]
    n_gpus = len(gpu_ids)

    # Resolve methods
    if run_all:
        method_list = METHODS_NO_SALIENCY if saliency_path is None else ALL_METHODS
    elif methods:
        method_list = [m.strip() for m in methods.split(",")]
    else:
        method_list = METHODS_NO_SALIENCY

    # Resolve models
    if run_all:
        model_list = ALL_MODELS
    elif models:
        model_list = [m.strip() for m in models.split(",")]
    elif model:
        model_list = [model]
    else:
        model_list = ["google/gemma-2-2b-it"]

    # Build job list — resolve saliency paths per model for taylor/gradient
    jobs: list[Job] = []
    for mdl in model_list:
        cfg = MODEL_CONFIGS.get(mdl, MODEL_CONFIGS["google/gemma-2-2b-it"])
        for meth in method_list:
            extra = []
            if meth in ("taylor", "gradient"):
                # Resolve saliency path: explicit flag > per-model template
                sal_path = saliency_path
                if sal_path is None and "saliency_path_template" in cfg:
                    sal_path = cfg["saliency_path_template"].format(subset=dataset_subset)
                if sal_path is None or not Path(sal_path).exists():
                    print(f"WARNING: method={meth} for {mdl} needs gradient map at {sal_path}.")
                    print(f"  Run the sweep shell script first, or: python -m sae_scoping.training.saliency.grad run --model-id {mdl} --output {sal_path}")
                    print(f"  Skipping this job.")
                    continue
                extra += ["--saliency-path", sal_path]
            jobs.append(Job(method=meth, model=mdl, extra_args=extra if extra else None))

    # Build common args
    common_args = [
        "--dataset-name", dataset_name,
        "--dataset-subset", dataset_subset,
        "--wandb-project", wandb_project,
    ]
    if sparsity_levels:
        common_args += ["--sparsity-levels", sparsity_levels]
    if no_judge:
        common_args += ["--no-judge"]

    print(f"\n{'='*70}")
    print(f"Sweep launcher: {len(jobs)} jobs across {n_gpus} GPUs ({gpu_ids})")
    print(f"{'='*70}")
    for i, job in enumerate(jobs):
        print(f"  [{i+1}] {job.name}")

    if dry_run:
        print("\n(Dry run — no jobs launched)")
        return

    # Launch with process pool (one worker per GPU)
    print(f"\nLaunching with {n_gpus} parallel workers...")
    completed = []
    failed = []

    # Assign GPUs round-robin and run in parallel
    with ProcessPoolExecutor(max_workers=n_gpus) as executor:
        futures = {}
        for i, job in enumerate(jobs):
            job.gpu_id = gpu_ids[i % n_gpus]
            future = executor.submit(run_job, job, common_args)
            futures[future] = job

        for future in as_completed(futures):
            job = futures[future]
            name, rc, snippet = future.result()
            if rc == 0:
                print(f"\n[DONE] {name} (GPU {job.gpu_id})")
                completed.append(name)
            else:
                print(f"\n[FAIL] {name} (GPU {job.gpu_id}, rc={rc})")
                print(f"  {snippet}")
                failed.append(name)

    # Summary
    print(f"\n{'='*70}")
    print(f"Results: {len(completed)} completed, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
