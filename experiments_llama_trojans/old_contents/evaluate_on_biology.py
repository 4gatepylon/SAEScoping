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
    SAE_ENHANCED = "sae_enhanced" # NOTE this, in this case, means using Sparsify SAEs instead of SAELens


class EvalKwargs(BaseModel):
    model_name_or_path: str
    sae_path: str | None # SAEs are not actually pruned in this case
    model_device: str
    n_samples: int
    judge_model: str
    judge_max_new_tokens: int
    judge_batch_size: int
    judge_batch_completion_kwargs: dict[str, Any]
    error_threshold: float


class EvalResult(BaseModel):
    kwargs: EvalKwargs
    data_dict: dict[str, float]  # utility scores
    metadata_dict: dict[str, str | int]  # optional SAE metadata


class EvalOutput(BaseModel):
    results: list[EvalResult]
    error_str: str | None
    traceback_str: str | None

# TODO similar but single-GPU version of experiments/script_2026_01_23_evaluate_biology_utility.py

@click.command()
@click.option("--output-path", "-o", type=str, default="biology_utility_cache.json", help="Output path to save results to")
@click.option("--gpu-ids", "-g", type=str, default="0", help="Comma-separated list of GPU IDs to use")
@click.option("--input-file", "-i", type=str, default=None, help="Previous results file to resume from (skips already-evaluated kwargs)")
@click.option("--sae-path-folder", "-spf", type=str, default=None, help="Folder containing SAE-enhanced checkpoints")
@click.option("--model-name-or-path", "-m", type=str, default=None, help="Model name or path to evaluate")
def main(output_path: str, gpu_ids: str, input_file: str | None, sae_path_folder: str | None, model_name_or_path: str | None) -> None:
    raise NotImplementedError("Not implemented yet") # Run inference with sparsify SAEs instead of SAELens


if __name__ == "__main__":
    main()
