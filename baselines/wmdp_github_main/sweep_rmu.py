#!/usr/bin/env python3
"""
RMU Hyperparameter Sweep Script

Runs RMU unlearning experiments across multiple GPUs with wandb logging.

Usage:
    python sweep_rmu.py --wandb_project rmu-2026-01-28
    python sweep_rmu.py --wandb_project rmu-2026-01-28 --gpus 0,1,2,3
    python sweep_rmu.py --config my_sweep_config.json
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import wandb


@dataclass
class SweepConfig:
    """Configuration for a single sweep run."""
    model_name: str
    steering_coeff: float
    alpha: float
    layer_id: int
    layer_ids: str
    param_ids: str
    lr: float
    max_num_batches: int
    batch_size: int
    gpu_id: int
    output_dir: str
    run_name: str


def load_sweep_config(config_path: str) -> dict:
    """Load sweep configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def get_sweep_configs(sweep_params: dict, output_base: str) -> list[SweepConfig]:
    """Generate all sweep configurations from parameters."""
    configs = []

    models = sweep_params["models"]
    steering_coeffs = sweep_params["steering_coeffs"]
    alphas = sweep_params["alphas"]
    layer_configs = sweep_params["layer_configs"]
    lrs = sweep_params["lrs"]
    max_num_batches = sweep_params["max_num_batches"]
    batch_size = sweep_params["batch_size"]
    param_ids = sweep_params.get("param_ids", "6")

    for model, sc, alpha, layer_cfg, lr in itertools.product(
        models, steering_coeffs, alphas, layer_configs, lrs
    ):
        layer_id = layer_cfg["layer_id"]
        layer_ids = layer_cfg["layer_ids"]

        # Create intelligent run name
        model_short = model.split("/")[-1].replace("-", "_")
        run_name = f"{model_short}_sc{int(sc)}_a{int(alpha)}_l{layer_id}_lr{lr:.0e}"

        output_dir = os.path.join(output_base, run_name)

        configs.append(SweepConfig(
            model_name=model,
            steering_coeff=sc,
            alpha=alpha,
            layer_id=layer_id,
            layer_ids=layer_ids,
            param_ids=param_ids,
            lr=lr,
            max_num_batches=max_num_batches,
            batch_size=batch_size,
            gpu_id=-1,  # Will be assigned later
            output_dir=output_dir,
            run_name=run_name,
        ))

    return configs


def run_single_experiment(config: SweepConfig, wandb_project: str, script_dir: str) -> dict:
    """Run a single RMU experiment and evaluation."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)
    os.environ["PYTHONPATH"] = script_dir

    # Initialize wandb
    run = wandb.init(
        project=wandb_project,
        name=config.run_name,
        config={
            "model": config.model_name,
            "steering_coeff": config.steering_coeff,
            "alpha": config.alpha,
            "layer_id": config.layer_id,
            "layer_ids": config.layer_ids,
            "param_ids": config.param_ids,
            "lr": config.lr,
            "max_num_batches": config.max_num_batches,
            "batch_size": config.batch_size,
        },
        reinit=True,
    )

    results = {
        "config": config.__dict__,
        "status": "failed",
        "wmdp_acc": None,
        "wmdp_bio_acc": None,
        "wmdp_cyber_acc": None,
        "wmdp_chem_acc": None,
        "mmlu_acc": None,
    }

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(config.gpu_id),
        "PYTHONPATH": script_dir,
    }

    try:
        # Run RMU training
        train_cmd = [
            sys.executable, "-m", "rmu.unlearn",
            "--model_name_or_path", config.model_name,
            "--output_dir", config.output_dir,
            "--layer_id", str(config.layer_id),
            "--layer_ids", config.layer_ids,
            "--param_ids", config.param_ids,
            "--batch_size", str(config.batch_size),
            "--max_num_batches", str(config.max_num_batches),
            "--steering_coeffs", f"{config.steering_coeff},{config.steering_coeff}",
            "--alpha", f"{config.alpha},{config.alpha}",
            "--lr", str(config.lr),
        ]

        print(f"[GPU {config.gpu_id}] Starting training: {config.run_name}")
        train_start = time.time()

        train_result = subprocess.run(
            train_cmd,
            cwd=script_dir,
            env=env,
            capture_output=True,
            text=True,
        )

        train_time = time.time() - train_start
        wandb.log({"train_time_seconds": train_time})

        if train_result.returncode != 0:
            print(f"[GPU {config.gpu_id}] Training failed: {train_result.stderr[-500:]}")
            wandb.log({"error": train_result.stderr[-1000:]})
            run.finish()
            return results

        print(f"[GPU {config.gpu_id}] Training completed in {train_time:.1f}s, starting eval...")

        # Run evaluation
        eval_output_dir = os.path.join(config.output_dir, "eval")
        eval_cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={config.output_dir},dtype=bfloat16",
            "--tasks", "wmdp",
            "--batch_size", "8",
            "--output_path", eval_output_dir,
        ]

        eval_start = time.time()
        eval_result = subprocess.run(
            eval_cmd,
            env=env,
            capture_output=True,
            text=True,
        )
        eval_time = time.time() - eval_start
        wandb.log({"eval_time_seconds": eval_time})

        if eval_result.returncode != 0:
            print(f"[GPU {config.gpu_id}] Evaluation failed: {eval_result.stderr[-500:]}")
            wandb.log({"error": eval_result.stderr[-1000:]})
            run.finish()
            return results

        # Parse WMDP evaluation results
        results_file = os.path.join(eval_output_dir, "results.json")
        if os.path.exists(results_file):
            with open(results_file) as f:
                eval_results = json.load(f)

            results["wmdp_acc"] = eval_results["results"]["wmdp"]["acc,none"]
            results["wmdp_bio_acc"] = eval_results["results"]["wmdp_bio"]["acc,none"]
            results["wmdp_cyber_acc"] = eval_results["results"]["wmdp_cyber"]["acc,none"]
            results["wmdp_chem_acc"] = eval_results["results"]["wmdp_chem"]["acc,none"]

            wandb.log({
                "wmdp_acc": results["wmdp_acc"],
                "wmdp_bio_acc": results["wmdp_bio_acc"],
                "wmdp_cyber_acc": results["wmdp_cyber_acc"],
                "wmdp_chem_acc": results["wmdp_chem_acc"],
            })

            print(f"[GPU {config.gpu_id}] {config.run_name} WMDP={results['wmdp_acc']:.4f}, running MMLU...")

        # Run MMLU evaluation
        mmlu_output_dir = os.path.join(config.output_dir, "eval_mmlu")
        mmlu_cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={config.output_dir},dtype=bfloat16",
            "--tasks", "mmlu",
            "--batch_size", "8",
            "--output_path", mmlu_output_dir,
        ]

        mmlu_start = time.time()
        mmlu_result = subprocess.run(
            mmlu_cmd,
            env=env,
            capture_output=True,
            text=True,
        )
        mmlu_time = time.time() - mmlu_start
        wandb.log({"mmlu_eval_time_seconds": mmlu_time})

        if mmlu_result.returncode != 0:
            print(f"[GPU {config.gpu_id}] MMLU evaluation failed: {mmlu_result.stderr[-500:]}")
            wandb.log({"mmlu_error": mmlu_result.stderr[-1000:]})
        else:
            # Parse MMLU results
            mmlu_results_file = os.path.join(mmlu_output_dir, "results.json")
            if os.path.exists(mmlu_results_file):
                with open(mmlu_results_file) as f:
                    mmlu_results = json.load(f)

                results["mmlu_acc"] = mmlu_results["results"]["mmlu"]["acc,none"]
                wandb.log({"mmlu_acc": results["mmlu_acc"]})

        results["status"] = "success"
        print(f"[GPU {config.gpu_id}] {config.run_name} completed: WMDP={results['wmdp_acc']:.4f}, MMLU={results.get('mmlu_acc', 'N/A')}")

    except Exception as e:
        print(f"[GPU {config.gpu_id}] Error: {e}")
        wandb.log({"error": str(e)})

    run.finish()
    return results


def run_experiment_wrapper(args):
    """Wrapper for ProcessPoolExecutor."""
    config, wandb_project, script_dir = args
    return run_single_experiment(config, wandb_project, script_dir)


def main():
    parser = argparse.ArgumentParser(description="RMU Hyperparameter Sweep")
    parser.add_argument("--wandb_project", type=str, default="rmu-2026-01-28")
    parser.add_argument("--gpus", type=str, default="0,1,2,3", help="Comma-separated GPU IDs")
    parser.add_argument("--config", type=str, default="sweep_config.json",
                        help="Path to sweep config JSON file")
    parser.add_argument("--output_base", type=str, default="models/sweep")
    parser.add_argument("--dry_run", action="store_true", help="Print configs without running")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    script_dir = Path(__file__).parent.absolute()

    # Load sweep config
    config_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
    print(f"Loading sweep config from: {config_path}")
    sweep_params = load_sweep_config(config_path)

    # Generate all configurations
    output_base = os.path.join(script_dir, args.output_base)
    configs = get_sweep_configs(sweep_params, output_base)

    print(f"Generated {len(configs)} configurations to sweep")
    print(f"Using GPUs: {gpus}")
    print(f"Wandb project: {args.wandb_project}")

    if args.dry_run:
        print("\nDry run - configurations that would be run:")
        for c in configs:
            print(f"  {c.run_name}")
        return

    # Skip already completed runs
    completed_runs = set()
    for config in configs:
        results_file = os.path.join(config.output_dir, "eval", "results.json")
        if os.path.exists(results_file):
            completed_runs.add(config.run_name)
            print(f"Skipping completed: {config.run_name}")

    configs = [c for c in configs if c.run_name not in completed_runs]
    print(f"Remaining configurations: {len(configs)}")

    if not configs:
        print("All configurations already completed!")
        return

    # Assign GPUs round-robin
    for i, config in enumerate(configs):
        config.gpu_id = gpus[i % len(gpus)]

    # Run in parallel across GPUs
    all_results = []
    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = []

        # Submit all tasks
        for config in configs:
            future = executor.submit(
                run_experiment_wrapper,
                (config, args.wandb_project, str(script_dir))
            )
            futures.append((future, config))

        # Collect results
        for future, config in futures:
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"Failed {config.run_name}: {e}")
                all_results.append({"config": config.__dict__, "status": "error", "error": str(e)})

    # Save summary
    summary_file = os.path.join(output_base, "sweep_summary.json")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "="*80)
    print("SWEEP SUMMARY")
    print("="*80)
    successful = [r for r in all_results if r.get("status") == "success"]
    if successful:
        # Sort by WMDP accuracy (lower is better for unlearning)
        successful.sort(key=lambda x: x.get("wmdp_acc", 1.0))
        print(f"{'Run Name':<45} {'WMDP':>7} {'Bio':>7} {'Cyber':>7} {'Chem':>7} {'MMLU':>7}")
        print("-"*88)
        for r in successful[:10]:  # Top 10
            mmlu = r.get('mmlu_acc')
            mmlu_str = f"{mmlu:>7.4f}" if mmlu else "   N/A "
            print(f"{r['config']['run_name']:<45} "
                  f"{r['wmdp_acc']:>7.4f} "
                  f"{r['wmdp_bio_acc']:>7.4f} "
                  f"{r['wmdp_cyber_acc']:>7.4f} "
                  f"{r['wmdp_chem_acc']:>7.4f} "
                  f"{mmlu_str}")

    print(f"\nCompleted: {len(successful)}/{len(all_results)}")
    print(f"Results saved to: {summary_file}")


if __name__ == "__main__":
    main()
