from __future__ import annotations
import uuid
import os
from pathlib import Path
import re
import click
import traceback
from beartype import beartype
from typing import Callable, Any
import orjson
from copy import deepcopy
import random
from multiprocessing import Pool
from tempfile import TemporaryDirectory


class TooManyHValuesError(Exception):
    """Raised when there are too many h values in the folder."""

    pass


class CannotReturnExactlyNumPerHError(Exception):
    """Raised when it's not possible to return exactly `num_per_h` checkpoints per `h` value."""

    pass


@beartype
def extract_threshold_from_checkpoint_path(checkpoint_path: str) -> float:
    """
    Given a path to a checkpoint in a tree like the following:
    ```
    /mnt/align4_drive2/adrianoh/git/SAEScoping/experiments/
        outputs_gemma9b_h_sweep_2026_01_20/outputs_gemma9b/biology/
            model_layers_31_h0.0005/outputs/
                checkpoint-2000/ <--- extract this folder
            model_layers_31_h1e-05/outputs/
                checkpoint-3000/ <--- extract this folder
            ...

    ```
    extract from a single path (i.e. the /mnt/.../model_layers_31_h0.0005/outputs/checkpoint-2000/) the h value. It
    is guaranteed to be as a str(<float>) from python, so it could be `<digits>.<digits>e<int>` or `<digits>.<digits>`.
    """
    # Match h followed by a float (decimal or scientific notation)
    # Pattern: _h followed by digits, optional decimal, optional scientific notation
    h_find_pattern = r"_h(\d+\.?\d*(?:e[+-]?\d+)?)"
    match = re.search(h_find_pattern, checkpoint_path, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not extract h value from path: {checkpoint_path}")
    return float(match.group(1))


@beartype
def get_checkpoints(
    folder: Path,
    num_per_h: int | None = None,
    prioritize_step_callable: Callable[[int], float] = lambda step: random.random(),
    num_per_must_be_exact: bool = False,
) -> list[Path]:
    """
    Given a tree like the following:
    ```
    <folder>/
        model_layers_31_h0.0005/outputs/
            checkpoint-2000
        model_layers_31_h1e-05/outputs/
            checkpoint-2000/
            checkpoint-3000/
        model_layers_31_h1e-03/outputs/
            <empty>
        model_layers_31_h1e-01/outputs/
            checkpoint-4000
    ```
    extract checkpoints following the following constraints:
    - Each subfolder must have a unique `h` value or we will always raise a TooManyHValuesError.
    - Get exactly `num_per_h` checkpoints per `h` value if possible and if `num_per_must_be_exact` is True.
        Otherwise, if `num_per_must_be_exact` is True and its not possible raise `CannotReturnExactlyNumPerHError`.
        Otherwise, return as many checkpoints as possible so long as they are under the `num_per_h` limit.
    - Return the first K checkpoints (where K is either `num_per_h` or the number chosen from the procedure ^)
        sorted by the `prioritize_step_callable` function. By default, this will not prioritize. Common choices are
        to prioritize later checkpoints or earlier checkpoints, but anything should be fair game.
    """
    if num_per_h is not None and num_per_h <= 0:
        raise ValueError(f"num_per_h must be positive, got {num_per_h}")
    if num_per_h is None and num_per_must_be_exact:
        raise ValueError("num_per_h must be specified if num_per_must_be_exact is True")

    # Group subfolders by h value, checking for duplicates
    h_to_subfolder: dict[float, Path] = {}
    for sf in folder.iterdir():
        if not sf.is_dir():
            continue
        try:
            h = extract_threshold_from_checkpoint_path(sf.name)
        except ValueError:
            continue
        if h in h_to_subfolder:
            raise TooManyHValuesError(f"Duplicate h={h} in {sf} and {h_to_subfolder[h]}")
        h_to_subfolder[h] = sf

    # Collect checkpoints per h value
    all_checkpoints: list[Path] = []
    for h, sf in h_to_subfolder.items():
        outputs_dir = sf / "outputs"
        if not outputs_dir.exists():
            if num_per_h is not None and num_per_must_be_exact:
                raise CannotReturnExactlyNumPerHError(f"No outputs directory for h={h}")
            continue

        # Find checkpoints and extract step numbers
        ckpts = [(int(d.name.split("-")[1]), d) for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]

        if num_per_must_be_exact and len(ckpts) < num_per_h:
            assert num_per_h is not None  # Impossible code (defensive wall here)
            raise CannotReturnExactlyNumPerHError(f"Only {len(ckpts)} checkpoints for h={h}, need {num_per_h}")

        # Sort by priority (descending) and take top K
        ckpts.sort(key=lambda x: prioritize_step_callable(x[0]), reverse=True)
        all_checkpoints.extend(path for _, path in ckpts[:num_per_h])  # [1,2,3][:None] == [1,2,3] -> True

    return all_checkpoints


@beartype
def process_kwargs_list(kwargs_list: list[dict[str, Any]], eval_fn: Callable, save_to_dir: Path) -> None:
    for kwargs in kwargs_list:
        data_dict, metadata_dict = eval_fn(**kwargs)
        output_path = save_to_dir / f"{uuid.uuid4()}.json"
        output_path.write_bytes(
            orjson.dumps(
                {
                    "kwargs": kwargs,
                    "data_dict": data_dict,
                    "metadata_dict": metadata_dict,
                }
            )
        )


def worker_fn(args: dict[str, Any]) -> None:
    gpu_id = args["gpu_id"]
    kwargs_list = args["kwargs_list"]
    temp_dir = Path(args["temp_dir"])
    temp_dir.mkdir(parents=True, exist_ok=True)  # Shared across all workers so should be fine imo
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # NOTE: this next should be the first import of the module
    from sae_scoping.evaluation.hardcoded_biology.utility_1click_judgement import evaluate_utility_on_biology_from_file

    process_kwargs_list(kwargs_list, evaluate_utility_on_biology_from_file, temp_dir)


def master_fn(kwargs_list: list[dict[str, Any]], gpu_ids: list[int], output_path: Path) -> None:
    if len(kwargs_list) == 0:
        raise ValueError("No kwargs provided to evaluate")
    gpu_ids = gpu_ids[: len(kwargs_list)]
    if len(gpu_ids) == 0:
        raise ValueError("No GPU IDs provided to compute with")
    with TemporaryDirectory() as temp_dir_str:
        kwargs_list_chunks = [deepcopy(kwargs_list[i :: len(gpu_ids)]) for i in range(len(gpu_ids))]
        assert len(kwargs_list_chunks) == len(gpu_ids) and all(len(chunk) > 0 for chunk in kwargs_list_chunks)
        worker_kwargs_list = [
            # Note we use UUID 4 filenames so there should be no collisions
            {"gpu_id": gpu_id, "kwargs_list": kwargs_list_chunk, "temp_dir": temp_dir_str}
            for gpu_id, kwargs_list_chunk in zip(gpu_ids, kwargs_list_chunks)
        ]
        err, tb, err_obj = None, None, None
        try:
            with Pool(len(gpu_ids)) as pool:
                pool.map(worker_fn, worker_kwargs_list)
        except Exception as e:
            err, tb, err_obj = str(e), traceback.format_exc(), e
        temp_dir = Path(temp_dir_str)
        json_filepaths = list(temp_dir.glob("*.json"))
        dictionary_list = []
        for json_filepath in json_filepaths:
            try:
                dictionary_list.append(orjson.loads(json_filepath.read_bytes()))
            except Exception:
                continue
        output_path.write_bytes(
            orjson.dumps(
                {
                    "results": dictionary_list,
                    "error_str": err,
                    "traceback_str": tb,
                }
            )
        )
        assert sum(int(z is None) for z in [err, tb, err_obj]) in [0, 3]  # All or nothing
        if err_obj is not None:
            raise err_obj


@click.command()
@click.option("--output-path", "-o", type=str, default="biology_utility_cache.json", help="Output path to save results to")
@click.option("--gpu-ids", "-g", type=str, default="0", help="Comma-separated list of GPU IDs to use")
def main(output_path: str, gpu_ids: str) -> None:
    """Evaluate biology utility for OUR model as of 2026-01-23. NOTE that you must pass CUDA_VISIBLE_DEVICES to the script."""
    shared_kwargs = {
        "model_device": "cuda:0",
        "n_samples": 30,
        "judge_model": "gpt-4.1-nano",
        "judge_max_new_tokens": 700,
        "judge_batch_size": 500,
        "judge_batch_completion_kwargs": {},
        "error_threshold": 0.1,
    }
    GEMMA_2_9B_IT_PATH = "google/gemma-2-9b-it"
    GEMMA_2_9B_SCOPED_PATH = "/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b/ultrachat/layer_31_width_16k_canonical_h0.0001_85cac49528/checkpoint-2000"
    GEMMA_2_9B_SCOPED_SWEEP_PATH_FOLDER = Path("/mnt/align4_drive2/adrianoh/git/SAEScoping/experiments/outputs_gemma9b_h_sweep_2026_01_20/outputs_gemma9b/biology")
    PRUNED_SAE_DIST_PATH = "/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors"
    checkpoints = get_checkpoints(GEMMA_2_9B_SCOPED_SWEEP_PATH_FOLDER, num_per_h=1, prioritize_step_callable=lambda step: 2 if step in [2000, 1500] else 1 if step in [4000, 3000] else random.random())
    kwargs_list = [
        {
            **shared_kwargs,
            "model_name_or_path": GEMMA_2_9B_IT_PATH,
            "pruned_sae_dist_path": None,
            "pruned_sae_threshold": 0.0,  # Dummy
        },
        {
            **shared_kwargs,
            "model_name_or_path": GEMMA_2_9B_SCOPED_PATH,
            "pruned_sae_dist_path": PRUNED_SAE_DIST_PATH,  # All SAE-enhanced models use this same dist path
            "pruned_sae_threshold": 1e-4,
        },
    ]
    for checkpoint in checkpoints:
        kwargs_list.append(
            {
                **shared_kwargs,
                "model_name_or_path": checkpoint.as_posix(),
                "pruned_sae_dist_path": PRUNED_SAE_DIST_PATH,  # All SAE-enhanced models use this same dist path
                "pruned_sae_threshold": extract_threshold_from_checkpoint_path(checkpoint.as_posix()),
            }
        )
    # Get and store full results across various gpus
    gpu_ids: list[int] = sorted(set(int(gpu_id) for gpu_id in gpu_ids.split(",")))
    master_fn(kwargs_list, gpu_ids, Path(output_path))


if __name__ == "__main__":
    main()
