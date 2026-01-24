from __future__ import annotations
import warnings
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
from enum import Enum
from pydantic import BaseModel


class TooManyHValuesError(Exception):
    """Raised when there are too many h values in the folder."""

    pass


class CannotReturnExactlyNumPerHError(Exception):
    """Raised when it's not possible to return exactly `num_per_h` checkpoints per `h` value."""

    pass


class CheckpointType(str, Enum):
    SFT = "sft"
    VANILLA = "vanilla"
    SAE_ENHANCED = "sae_enhanced"


class EvalKwargs(BaseModel):
    model_name_or_path: str
    model_device: str
    n_samples: int
    judge_model: str
    judge_max_new_tokens: int
    judge_batch_size: int
    judge_batch_completion_kwargs: dict[str, Any]
    error_threshold: float
    pruned_sae_dist_path: str | None
    pruned_sae_threshold: float


class EvalResult(BaseModel):
    kwargs: EvalKwargs
    data_dict: dict[str, float]  # utility scores
    metadata_dict: dict[str, str | int]  # optional SAE metadata


class EvalOutput(BaseModel):
    results: list[EvalResult]
    error_str: str | None
    traceback_str: str | None


@beartype
def classify_checkpoint_type(model_name_or_path: str, pruned_sae_dist_path: str | None) -> CheckpointType:
    """
    Classify a checkpoint as SFT, VANILLA, or SAE_ENHANCED.

    - VANILLA: HuggingFace model ID (e.g., "google/gemma-2-9b-it")
    - SFT: Path containing "/vanilla/" (no SAE involved)
    - SAE_ENHANCED: Has pruned_sae_dist_path or path contains SAE-related patterns
    """
    # Vanilla Gemma from HF - check if it's a HF model ID (no leading /)
    if not model_name_or_path.startswith("/") and "/" in model_name_or_path:
        assert pruned_sae_dist_path is None
        assert not "/vanilla/" in model_name_or_path
        return CheckpointType.VANILLA

    # SFT checkpoint: path contains /vanilla/
    if "/vanilla/" in model_name_or_path:
        assert pruned_sae_dist_path is None
        return CheckpointType.SFT

    # SAE-enhanced: has pruned_sae_dist_path
    if pruned_sae_dist_path is not None:
        return CheckpointType.SAE_ENHANCED

    raise ValueError(f"Could not classify checkpoint type for {model_name_or_path}")


@beartype
def extract_step_from_checkpoint_path(checkpoint_path: str) -> int:
    """Extract step number from a checkpoint path like '/path/to/checkpoint-1000'."""
    match = re.search(r"checkpoint-(\d+)", checkpoint_path)
    if match is None:
        raise ValueError(f"Could not extract step from path: {checkpoint_path}")
    return int(match.group(1))


@beartype
def get_sft_checkpoints(
    folder: Path | str,
    num_per_config: int | None = None,
    prioritize_step_callable: Callable[[int], float] = lambda step: random.random(),
) -> list[Path]:
    """Implemented by Claude Code and analogous to `get_checkpoints()`, which is for SAE-enhanced checkpoints."""
    folder = Path(folder)
    if not folder.exists():
        return []

    # Find checkpoints and extract step numbers
    ckpts = [(int(d.name.split("-")[1]), d) for d in folder.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]

    if not ckpts:
        return []

    # Sort by priority (descending) and take top K
    ckpts.sort(key=lambda x: prioritize_step_callable(x[0]), reverse=True)
    return [path for _, path in ckpts[:num_per_config]]


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
def process_kwargs_list(kwargs_list: list[EvalKwargs], eval_fn: Callable, save_to_dir: Path) -> None:
    for kwargs in kwargs_list:
        data_dict, metadata_dict = eval_fn(**kwargs.model_dump())
        result = EvalResult(kwargs=kwargs, data_dict=data_dict, metadata_dict=metadata_dict)
        output_path = save_to_dir / f"{uuid.uuid4()}.json"
        output_path.write_bytes(result.model_dump_json().encode())


def worker_fn(args: dict[str, Any]) -> None:
    gpu_id = args["gpu_id"]
    kwargs_list = args["kwargs_list"]
    temp_dir = Path(args["temp_dir"])
    temp_dir.mkdir(parents=True, exist_ok=True)  # Shared across all workers so should be fine imo
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # NOTE: this next should be the first import of the module
    from sae_scoping.evaluation.hardcoded_biology.utility_1click_judgement import evaluate_utility_on_biology_from_file

    process_kwargs_list(kwargs_list, evaluate_utility_on_biology_from_file, temp_dir)


def master_fn(kwargs_list: list[EvalKwargs], gpu_ids: list[int], output_path: Path, previous_results: list[EvalResult] | None = None) -> None:
    """Produce all the EvalResults for the given kwargs list and save them to the output path."""
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
        results = [EvalResult.model_validate_json(fp.read_bytes()) for fp in json_filepaths]
        if previous_results is not None:
            results = previous_results + results
        output = EvalOutput(results=results, error_str=err, traceback_str=tb)
        tmp_output_path = output_path.with_suffix(f".tmp.{uuid.uuid4()}.json")
        try:
            tmp_output_path.write_bytes(output.model_dump_json().encode())
            tmp_output_path.rename(output_path)  # mv to output path (atomic overwrite)
        except Exception as e:
            if err_obj is None:
                err, tb, err_obj = str(e), traceback.format_exc(), e
            else:
                warnings.warn(f"Error writing output to {tmp_output_path} AFTER previous error")
                warnings.warn(traceback.format_exc())
        finally:
            tmp_output_path.unlink(missing_ok=True)  # remove tmp file (may not exist after successful rename)
        assert sum(int(z is None) for z in [err, tb, err_obj]) in [0, 3]  # All or nothing
        if err_obj is not None:
            raise err_obj


@click.command()
@click.option("--output-path", "-o", type=str, default="biology_utility_cache.json", help="Output path to save results to")
@click.option("--gpu-ids", "-g", type=str, default="0", help="Comma-separated list of GPU IDs to use")
@click.option("--input-file", "-i", type=str, default=None, help="Previous results file to resume from (skips already-evaluated kwargs)")
def main(output_path: str, gpu_ids: str, input_file: str | None) -> None:
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
    SFT_VANILLA_FOLDER = Path("/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b/biology/vanilla")

    # SAE-enhanced checkpoints
    checkpoints = get_checkpoints(GEMMA_2_9B_SCOPED_SWEEP_PATH_FOLDER, num_per_h=1, prioritize_step_callable=lambda step: 2 if step in [2000, 1500] else 1 if step in [4000, 3000] else random.random())

    # SFT checkpoints (vanilla, no SAE)
    sft_checkpoints = get_sft_checkpoints(SFT_VANILLA_FOLDER, num_per_config=None, prioritize_step_callable=lambda step: step)  # Prioritize higher steps

    kwargs_list: list[EvalKwargs] = [
        EvalKwargs(**shared_kwargs, model_name_or_path=GEMMA_2_9B_IT_PATH, pruned_sae_dist_path=None, pruned_sae_threshold=0.0),
        EvalKwargs(**shared_kwargs, model_name_or_path=GEMMA_2_9B_SCOPED_PATH, pruned_sae_dist_path=PRUNED_SAE_DIST_PATH, pruned_sae_threshold=1e-4),
    ]
    # Add SAE-enhanced checkpoints
    for checkpoint in checkpoints:
        kwargs_list.append(
            EvalKwargs(
                **shared_kwargs,
                model_name_or_path=checkpoint.as_posix(),
                pruned_sae_dist_path=PRUNED_SAE_DIST_PATH,
                pruned_sae_threshold=extract_threshold_from_checkpoint_path(checkpoint.as_posix()),
            )
        )
    # Add SFT checkpoints (no SAE)
    for sft_checkpoint in sft_checkpoints:
        kwargs_list.append(
            EvalKwargs(
                **shared_kwargs,
                model_name_or_path=sft_checkpoint.as_posix(),
                pruned_sae_dist_path=None,
                pruned_sae_threshold=0.0,
            )
        )

    # Load previous results if input file provided and filter out already-evaluated kwargs
    previous_results: list[EvalResult] | None = None
    if input_file is not None:
        input_path = Path(input_file)
        if input_path.exists():
            previous_output = EvalOutput.model_validate_json(input_path.read_bytes())
            previous_results = previous_output.results
            evaluated_kwargs = [r.kwargs for r in previous_results]
            original_count = len(kwargs_list)
            # Quadratic checking because small data and also because not hashable without more code
            kwargs_list = [k for k in kwargs_list if not any(k == ek for ek in evaluated_kwargs)]  # Filter out already-evaluated kwargs
            skipped_count = original_count - len(kwargs_list)
            if skipped_count > 0:
                click.echo(f"Resuming: skipping {skipped_count} already-evaluated kwargs, {len(kwargs_list)} remaining")

    # Get and store full results across various gpus
    gpu_ids_list: list[int] = sorted(set(int(gpu_id) for gpu_id in gpu_ids.split(",")))
    if len(kwargs_list) == 0:
        click.echo("All kwargs already evaluated, nothing to do")
    else:
        master_fn(kwargs_list, gpu_ids_list, Path(output_path), previous_results=previous_results)


if __name__ == "__main__":
    main()
