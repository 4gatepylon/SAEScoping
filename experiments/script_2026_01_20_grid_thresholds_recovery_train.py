from __future__ import annotations

import functools
from typing import Any
import wandb
import tqdm
import os
import traceback
import click
from multiprocessing import Pool

"""
Grid search over SAE neuron firing rate thresholds for recovery training.

This script runs recovery training multiple times with different threshold values,
storing a bounded number of checkpoints per threshold. Uses the biology dataset
and layer 31 SAE by default.
"""

# Include 0.0 for all SAE features kept

# In [1]: from safetensors import safe_open
#    ...: 
#    ...: path = "/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/deleteme_cache_bio_only/ignore_padd
#       ⋮ ing_True/biology/layer_31--width_16k--canonical/distribution.safetensors"
#    ...: 
#    ...: with safe_open(path, framework="pt") as f:
#    ...:     dist = f.get_tensor("distribution")
#    ...:     print(f"Min value: {dist.min().item()}")
#    ...: 
# Min value: 0.0
# ...
# In [3]: dist
# Out[3]: 
# tensor([1.1932e-05, 4.4196e-05, 2.3805e-05,  ..., 1.0953e-04, 4.9817e-06,
#         5.7810e-05])
# In [4]: dist.max()
# Out[4]: tensor(0.0045)
# In [5]: dist[dist > 0].min()
# Out[5]: tensor(3.0878e-09)
# Go from 1 feature to all features
DEFAULT_THRESHOLDS = [4.5e-3, 1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0]

# Distribution path for layer 31 SAE (from vibecheck script, also in the wandb, via this command:
# ```
# /mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/script_2025_12_08_train_gemma9b_sae.py -h 1e-4 -p deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors --max-steps 40_000 -b 4 -a 8
# ```)
from pathlib import Path

DEFAULT_DIST_PATH = (
    Path("/mnt/align4_drive2/adrianoh")
    / "git"
    / "ScopeBench"
    / "sae_training"
    / "deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors"
)
assert DEFAULT_DIST_PATH.exists()

def run_multiple_experiments(_experiment_kwargs: tuple[int, list[dict]]) -> None:
    gpu_id, experiment_kwargs = _experiment_kwargs
    if len(experiment_kwargs) == 0:
        print(f"WARNING: No experiments to run on GPU {gpu_id}")
        return
    os.environ["GRADIENT_CHECKPOINTING"] = "0" # Needed by the script _main, allegedly (a gemma thing...?)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from experiments.script_2025_12_08_train_gemma9b_sae import _main as train_gemma9b_sae
    for kwargs in experiment_kwargs:
        assert "output_dir" in kwargs
        root_dir = Path(kwargs["output_dir"])
        assert not root_dir.exists()
        output_dir = root_dir / "outputs"
        error_dir = root_dir / "errors"
        output_dir.mkdir(parents=True, exist_ok=True)
        error_dir.mkdir(parents=True, exist_ok=True)
        kwargs["output_dir"] = output_dir.as_posix()
        try:
            os.environ["WANDB_PROJECT"] = kwargs["wandb_project_name"] # defensive code
            os.environ["WANDB_RUN_NAME"] = kwargs["wandb_run_name"] # defensive code
            train_gemma9b_sae(**kwargs)
        except Exception as e:
            error_string = str(e)
            traceback_string = traceback.format_exc()
            error_file = error_dir / f"error.log"
            traceback_file = error_dir / f"traceback.log"
            error_file.parent.mkdir(parents=True, exist_ok=True)
            traceback_file.parent.mkdir(parents=True, exist_ok=True)
            error_file.write_text(error_string)
            traceback_file.write_text(traceback_string)
            print("="*60)
            print(f"Error running experiment {kwargs}: {e}")
            print("="*60)
        finally:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Error finishing WandB: {e}")


@click.command()
@click.option(
    "--thresholds",
    "-h",
    multiple=True,
    type=float,
    default=DEFAULT_THRESHOLDS,
    help="Thresholds to sweep over (min firing rate to keep neuron)",
)
@click.option(
    "--dist-path",
    "-p",
    type=str,
    default=DEFAULT_DIST_PATH.as_posix(),
    help="Path to distribution.safetensors",
)
@click.option("--batch-size", "-b", type=int, default=2, help="Training batch size")
@click.option("--accum", "-a", type=int, default=8, help="Gradient accumulation steps")
@click.option("--max-steps", "-s", type=int, default=4000, help="Max training steps")
@click.option(
    "--hookpoint",
    "-hook",
    type=str,
    default="model.layers.31",
    help="Hookpoint to apply SAE at",
)
@click.option(
    "--checkpoint", "-c", type=str, default=None, help="Checkpoint to resume from"
)
@click.option(
    "--train-on-dataset", "-t", type=str, default="biology", help="Dataset to train on"
)
@click.option(
    "--wandb-project-name",
    "-w",
    type=str,
    # We should reuse the original to compare plots more easily
    # default="gemma-scope-9b-recovery-train-sweep-2026-01-20",
    default="gemma-scope-9b-recovery-train-initial-2025-12-08",
    help="Wandb project name",
)
@click.option("--save-every", "-se", type=int, default=1500, help="Save every n steps")
@click.option(
    # NOTE that each save could take up to 50GB; at 2 saves (assuming it saves final by accident)
    # times <= 15 runs -> 30 * 50GB = 1.5TB (which we have as of 2026-01-20)
    "--save-limit", "-sl", type=int, default=2, help="Max checkpoints per run"
)
@click.option("--gpu-ids", "-g", type=str, default="0,1,2,3", help="GPU IDs to use")
def main(
    thresholds: tuple[float, ...],
    dist_path: str,
    batch_size: int,
    accum: int,
    max_steps: int,
    hookpoint: str,
    checkpoint: str | None,
    train_on_dataset: str,
    wandb_project_name: str,
    save_every: int,
    save_limit: int,
    gpu_ids: str,
) -> None:
    r"""
    Run recovery training grid search over multiple thresholds.

    For example run with this command with default thresholds and paths:
    ```
    python3 script_2026_01_20_grid_thresholds_recovery_train.py \
        --gpu-ids '2,3' \
        --batch-size 2 \
        --accum 8 \
        --max-steps 4000 \
        --save-every 2000 \
        --save-limit 2 \
        --train-on-dataset biology \
        --wandb-project-name 'gemma-scope-9b-recovery-train-initial-2025-12-08' \
        --dist-path '/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors'
    """
    print("="*60)
    print("Parsing arguments...")
    thresholds_list = list(thresholds)
    print(
        f"Running grid search over {len(thresholds_list)} thresholds: {thresholds_list}"
    )
    gpu_ids = list(map(int, gpu_ids.split(",")))
    kwargs_list_list: list[list[dict[str, Any]]] = [[] for _ in range(len(gpu_ids))]
    assert len(set(gpu_ids)) == len(gpu_ids)
    assert len(set(thresholds_list)) == len(thresholds_list)
    assert len(gpu_ids) <= len(thresholds_list)
    for i, threshold in tqdm.tqdm(list(enumerate(thresholds_list))):
        kwargs = {
            "dist_path": dist_path,
            "batch_size": batch_size,
            "threshold": threshold,
            "max_steps": max_steps,
            "accum": accum,
            "special_hookpoint": hookpoint,
            "checkpoint": checkpoint,
            "train_on_dataset": train_on_dataset,
            "wandb_project_name": wandb_project_name,
            "save_every": save_every,
            "save_limit": save_limit,
            "output_dir": f"./outputs_gemma9b_h_sweep_2026_01_20/outputs_gemma9b/{train_on_dataset}/{hookpoint.replace('.', '_')}_h{threshold}",
            "save_output": False, # do NOT save the output; that's why we save 2 checkpoints
            "wandb_run_name": f"{train_on_dataset}/{hookpoint.replace('.', '_')}/h{threshold}/google-gemma-9b-it",
        }
        kwargs_list_list[i % len(gpu_ids)].append(kwargs)
    # fmt: off
    assert len(functools.reduce(lambda x, y: x | y, [set(k["threshold"] for k in ks) for ks in kwargs_list_list], set())) == functools.reduce(lambda x, y: x + y, [len(k) for k in kwargs_list_list], 0)
    assert len(functools.reduce(lambda x, y: x | y, [set(k["output_dir"] for k in ks) for ks in kwargs_list_list], set())) == functools.reduce(lambda x, y: x + y, [len(k) for k in kwargs_list_list], 0)
    assert len(functools.reduce(lambda x, y: x | y, [set(k["wandb_run_name"] for k in ks) for ks in kwargs_list_list], set())) == functools.reduce(lambda x, y: x + y, [len(k) for k in kwargs_list_list], 0)
    # fmt: on
    print("="*60)
    print("Lengths of each gpu's list")
    assert len(gpu_ids) == len(kwargs_list_list)
    for gpu_id, kwargs_list in zip(gpu_ids, kwargs_list_list):
        print(f"GPU {gpu_id}: {len(kwargs_list)}, thresholds: {sorted(k['threshold'] for k in kwargs_list)}")
        print("\n - ".join(k["wandb_run_name"] for k in kwargs_list))
    print("OK?")
    click.confirm("Continue?", abort=True)
    print("="*60)
    print("Launching grid search across GPUs...")
    with Pool(len(gpu_ids)) as p:
        p.map(run_multiple_experiments, list(zip(gpu_ids, kwargs_list_list)))
    print("="*60)
    print(f"\nGrid search complete. Trained {len(thresholds_list)} models.")


if __name__ == "__main__":
    main()
