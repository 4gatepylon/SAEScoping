from __future__ import annotations
import json
import warnings
import uuid
import os
from pathlib import Path
import re
import click
import traceback
from beartype import beartype
from typing import Callable, Any, Literal
from copy import deepcopy
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
    SAE_ENHANCED_SUBJECT = "sae_enhanced_subject"
    SAE_ENHANCED_ULTRACHAT = "sae_enhanced_ultrachat"


SubjectType = Literal["biology", "chemistry", "physics"]
VALID_SUBJECTS: list[SubjectType] = ["biology", "chemistry", "physics"]


class Checkpoint(BaseModel):
    name_or_path: str  # Path to the checkpoint
    type: CheckpointType  # Classification
    step: int  # Specific metadata for sorting etc...
    threshold: float | None  # Prune dist path to threshold
    dist_path: str | None  # Load this


class EvalKwargs(BaseModel):
    model_name_or_path: str
    generation_kwargs: dict[str, Any]
    n_samples: int
    judge_model: str
    judge_max_new_tokens: int
    judge_batch_size: int
    judge_batch_completion_kwargs: dict[str, Any]
    error_threshold: float
    pruned_sae_dist_path: str | None
    pruned_sae_threshold: float | None  # None for non-scoped use
    subject: SubjectType  # Added subject field


class EvalResult(BaseModel):
    kwargs: EvalKwargs
    data_dict: dict[str, float]  # utility scores
    metadata_dict: dict[str, str | int]  # optional SAE metadata
    generations_uuid: str | None  # UUID of the generations file (in generations subfolder)


class EvalOutput(BaseModel):
    results: list[EvalResult]
    error_str: str | None
    traceback_str: str | None


class OpenAIMessages(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class JudgementResult(BaseModel):
    judge_name: str
    score: float
    explanation: str
    is_error: bool


class GenerationResult(BaseModel):
    prompt: list[OpenAIMessages]
    response: str
    judgements: list[JudgementResult]


class GenerationsResult(BaseModel):  # Stored in another file adjacent to EvalOutput
    kwargs: EvalKwargs
    generations_uuid: str
    data_dict: dict[str, float]  # utility scores; copied from ^
    metadata_dict: dict[str, str | int]  # optional SAE metadata; copied from ^
    generations: list[GenerationResult]  # List of single generation results


@beartype
def extract_step_from_checkpoint_path(checkpoint_path: str) -> int:
    """Extract step number from a checkpoint path like '/path/to/checkpoint-1000'."""
    match = re.search(r"checkpoint-(\d+)", checkpoint_path)
    if match is None:
        raise ValueError(f"Could not extract step from path: {checkpoint_path}")
    return int(match.group(1))


@beartype
def extract_threshold_from_checkpoint_path(checkpoint_path: str) -> float:
    # Match h followed by a float (decimal or scientific notation)
    # Pattern: _h followed by digits, optional decimal, optional scientific notation
    h_find_pattern = r"_h(\d+\.?\d*(?:e[+-]?\d+)?)"
    match = re.search(h_find_pattern, checkpoint_path, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not extract h value from path: {checkpoint_path}")
    return float(match.group(1))


@beartype
def classify_checkpoint_type(model_name_or_path: str, subject: SubjectType) -> CheckpointType:
    # Vanilla Gemma from HF - check if it's a HF model ID (no leading /)
    if model_name_or_path == "google/gemma-2-9b-it":  # Only gemma is supported
        return CheckpointType.VANILLA
    assert Path(model_name_or_path).is_dir()  # Other options must be checkpoints

    # SFT checkpoint: path contains /vanilla/
    if "/vanilla/" in model_name_or_path:
        return CheckpointType.SFT

    # SAE-enhanced: has pruned_sae_dist_path
    if f"/{subject}/" in model_name_or_path:
        extract_threshold_from_checkpoint_path(model_name_or_path)  # raises if no threshold found
        return CheckpointType.SAE_ENHANCED_SUBJECT
    else:
        extract_threshold_from_checkpoint_path(model_name_or_path)  # raises if no threshold found
        return CheckpointType.SAE_ENHANCED_ULTRACHAT  # Original


# Return the non-vanilla one
@beartype
def _subject_sae_enhanced_checkpoints(subject: SubjectType) -> str:
    parent = Path(f"/mnt/align4_drive2/adrianoh/git/SAEScoping/experiments/outputs_gemma9b/{subject}")
    children = list(parent.iterdir())
    children = [c for c in children if c.is_dir() and not c.stem == "vanilla"]
    assert len(children) == 1, f"Expected 1 child folder for {subject} but got {len(children)}"
    return children[0].as_posix()


CHECKPOINT_PATH_FN_REGISTRY: dict[str, Callable[[SubjectType], str]] = {
    "google/gemma-2-9b-it": lambda _: "google/gemma-2-9b-it",
    "subject_vanilla_checkpoints": lambda subject: f"/mnt/align4_drive2/adrianoh/git/SAEScoping/experiments/outputs_gemma9b/{subject}/vanilla",  # new repo
    "subject_sae_enhanced_checkpoints": _subject_sae_enhanced_checkpoints,  # new repo
    "subject_sae_enhanced_ultrachat": lambda _: "/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b/ultrachat/layer_31_width_16k_canonical_h0.0001_85cac49528/",  # old repo
}
CHECKPOINT_NEEDS_PRUNED_SAE_DIST_PATH_REGISTRY: set[str] = {"subject_sae_enhanced_checkpoints", "subject_sae_enhanced_ultrachat"}
assert CHECKPOINT_NEEDS_PRUNED_SAE_DIST_PATH_REGISTRY.issubset(set(CHECKPOINT_PATH_FN_REGISTRY.keys()))

CHECKPOINT_STEP_FN_REGISTRY: dict[str, Callable[[str], int]] = {
    "google/gemma-2-9b-it": lambda x: 0,  # Not yet attacked
    "subject_vanilla_checkpoints": extract_step_from_checkpoint_path,
    "subject_sae_enhanced_checkpoints": extract_step_from_checkpoint_path,
    "subject_sae_enhanced_ultrachat": lambda x: 0,  # Not yet attacked
}
assert set(CHECKPOINT_PATH_FN_REGISTRY.keys()) == set(CHECKPOINT_STEP_FN_REGISTRY.keys())


def folder_is_checkpoint(checkpoint: str) -> bool:
    g = re.match(r"^(.*checkpoint-\d+)$", checkpoint)
    if g is None:
        return False
    return g.group(1) == checkpoint


def is_vanilla_checkpoint(checkpoint: str) -> bool:
    return checkpoint == "google/gemma-2-9b-it"


def get_relevant_checkpoints(checkpoints: list[str], subject: SubjectType, pruned_sae_dist_path: str | None) -> list[Checkpoint]:
    if len(checkpoints) == 0:
        raise ValueError("No checkpoints provided")
    elif len(checkpoints) == 1:
        # 1. Extract ACTUAL checkpoint and step from registry; might be folder or checkpoint
        checkpoint, registry_key = checkpoints[0], None
        if checkpoint in CHECKPOINT_PATH_FN_REGISTRY:
            registry_key = checkpoint
            checkpoint = CHECKPOINT_PATH_FN_REGISTRY[registry_key](subject)
        # 2. Early-exit (recurse) if we cannot extract step number
        if not is_vanilla_checkpoint(checkpoint) and not folder_is_checkpoint(checkpoint):
            # Recurse
            all_checkpoints = []
            globlings = list(Path(checkpoint).glob("checkpoint-*")) + list(Path(checkpoint).glob("**/checkpoint-*"))
            globlings = [g.as_posix() for g in globlings if g.exists() and g.is_dir()]
            globlings = list(set(globlings))  # Remove duplicates
            for c in globlings:
                all_checkpoints += get_relevant_checkpoints([c], subject, pruned_sae_dist_path)
            return all_checkpoints
        # 3. Extract/classify
        step = extract_step_from_checkpoint_path(checkpoint) if registry_key is None else CHECKPOINT_STEP_FN_REGISTRY[registry_key](checkpoint)  # registry_key => is in registry
        classification = classify_checkpoint_type(checkpoint, subject)
        dist_path = pruned_sae_dist_path if classification in {CheckpointType.SAE_ENHANCED_SUBJECT, CheckpointType.SAE_ENHANCED_ULTRACHAT} else None
        threshold = extract_threshold_from_checkpoint_path(checkpoint) if classification in {CheckpointType.SAE_ENHANCED_SUBJECT, CheckpointType.SAE_ENHANCED_ULTRACHAT} else None
        assert is_vanilla_checkpoint(checkpoint) == (classification == CheckpointType.VANILLA)
        assert is_vanilla_checkpoint(checkpoint) or folder_is_checkpoint(checkpoint)
        # 4. Format output and return
        return [Checkpoint(name_or_path=checkpoint, type=classification, step=step, threshold=threshold, dist_path=dist_path)]

    else:  # recurse on list
        all_checkpoints = []
        for c in checkpoints:
            all_checkpoints += get_relevant_checkpoints([c], subject, pruned_sae_dist_path)
        return all_checkpoints


@beartype
def _convert_raw_generation_to_pydantic(raw: dict[str, Any]) -> GenerationResult:
    """Convert raw generation dict from eval fn to pydantic GenerationResult."""
    prompt = [OpenAIMessages(role=m["role"], content=m["content"]) for m in raw["prompt"]]
    judgements = [
        JudgementResult(
            judge_name=j["_judge_name"],
            score=j["score"],
            explanation=j["explanation"],
            is_error=j["_is_error"],
        )
        for j in raw["judgements"]
    ]
    return GenerationResult(prompt=prompt, response=raw["response"], judgements=judgements)


@beartype
def process_kwargs_list(
    kwargs_list: list[EvalKwargs],
    eval_fn: Callable,
    temp_save_dir: Path,
    generations_dir: Path,
) -> None:
    for kwargs in kwargs_list:
        data_dict, metadata_dict, generations = eval_fn(**kwargs.model_dump())
        gen_uuid = str(uuid.uuid4())

        # Save generations if requested
        generations_pydantic = [_convert_raw_generation_to_pydantic(g) for g in generations]
        generations_result = GenerationsResult(
            kwargs=kwargs,
            generations_uuid=gen_uuid,
            data_dict=data_dict,
            metadata_dict=metadata_dict,
            generations=generations_pydantic,
        )
        gen_output_path = generations_dir / f"{gen_uuid}.json"  # Perma-save
        gen_output_path.write_bytes(generations_result.model_dump_json().encode())

        result = EvalResult(
            kwargs=kwargs,
            data_dict=data_dict,
            metadata_dict=metadata_dict,
            generations_uuid=gen_uuid,
        )
        output_path = temp_save_dir / f"{uuid.uuid4()}.json"  # Temporary before joining together
        output_path.write_bytes(result.model_dump_json().encode())


def worker_fn(args: dict[str, Any]) -> None:
    gpu_id = args["gpu_id"]
    kwargs_list = args["kwargs_list"]
    temp_dir = Path(args["temp_dir"])
    generations_dir = Path(args["generations_dir"])
    temp_dir.mkdir(parents=True, exist_ok=True)  # Shared across all workers so should be fine imo
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # NOTE: this next should be the first import of the module
    from sae_scoping.evaluation.hardcoded_biology.utility_1click_judgement_generic import evaluate_utility_on_subject_from_file

    process_kwargs_list(kwargs_list, evaluate_utility_on_subject_from_file, temp_dir, generations_dir)


def master_fn(
    kwargs_list: list[EvalKwargs],
    gpu_ids: list[int],
    output_path: Path,
    generations_dir: Path,
    previous_results: list[EvalResult] | None = None,
) -> None:
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
            {
                "gpu_id": gpu_id,
                "kwargs_list": kwargs_list_chunk,
                "temp_dir": temp_dir_str,
                "generations_dir": generations_dir.as_posix(),  # Convert to string for multiprocessing
            }
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


DEFAULT_PRUNED_SAE_DIST_PATH = "/mnt/align4_drive2/adrianoh/scope_bench_spring_2026/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors"


@click.command()
@click.option("--subject", "-s", type=click.Choice(VALID_SUBJECTS), default="chemistry", help="Subject to evaluate (biology, chemistry, physics)")
@click.option("--output-path", "-o", type=str, default=None, help="Output path to save results to (defaults to {subject}_utility_cache.json)")
@click.option("--gpu-ids", "-g", type=str, default="0", help="Comma-separated list of GPU IDs to use")
@click.option("--input-file", "-i", type=str, default=None, help="Previous results file to resume from (skips already-evaluated kwargs)")
@click.option(
    "--checkpoints",
    "-c",
    multiple=True,
    help="Folders containing checkpoints",
    default=[
        "google/gemma-2-9b-it",  # Original/vanilla
        "subject_vanilla_checkpoints",  # Special function to "find vanilla checkpoints for my subject; i.e. SFT on gemma9b"
        "subject_sae_enhanced_checkpoints",  # Special function to "find SAE-enhanced checkpoints for my subject; i.e. SAE-enhanced on gemma9b"
        "/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b/ultrachat/layer_31_width_16k_canonical_h0.0001_85cac49528/",  # Our scoped model from which all comes
    ],
)
@click.option(
    "--pruned-sae-dist-path", type=str, default=DEFAULT_PRUNED_SAE_DIST_PATH, help="Path to pruned SAE distribution file. ALL checkpoints that match as having a threshold will ahve this applied."
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option("--generation-kwargs", "-gk", type=str, default=f'{{"do_sample": False, "max_new_tokens": 700}}', help="Generation kwargs to pass to the model. JSON string.")
@click.option("--judge-model", "-jm", type=str, default="gpt-4.1-nano", help="Judge model to use.")
@click.option("--judge-max-new-tokens", "-jmnt", type=int, default=700, help="Max new tokens for judge.")
@click.option("--judge-batch-size", "-jbs", type=int, default=500, help="Batch size for judge.")
@click.option("--judge-batch-completion-kwargs", "-jbc", type=str, default=r"{}", help="Batch completion kwargs for judge. JSON string.")
@click.option("--n-samples", "-ns", type=int, default=30, help="Number of samples to generate.")
@click.option("--debug", "-d", is_flag=True, help="Debug mode.")
def main(
    subject: SubjectType,
    output_path: str | None,
    gpu_ids: str,
    input_file: str | None,
    checkpoints: list[str],
    pruned_sae_dist_path: str | None,
    yes: bool,
    generation_kwargs: str,
    # Judge arguemnts
    judge_model: str,
    judge_max_new_tokens: int,
    judge_batch_size: int,
    judge_batch_completion_kwargs: str,
    n_samples: int,
    debug: bool,
) -> None:
    """Evaluate subject utility for a model. Pass CUDA_VISIBLE_DEVICES to the script."""
    generation_kwargs_parsed: dict[str, Any] = json.loads(generation_kwargs)

    if debug:
        import litellm
        litellm._turn_on_debug()

    # Set default output path based on subject
    output_path_resolved: Path = Path(output_path) if output_path is not None else Path(__file__).parent / f"{subject}_utility_cache"
    output_path_resolved.mkdir(parents=True, exist_ok=True)
    output_overview_path = output_path_resolved / f"{subject}_utility_cache_overview.json"
    output_generations_path = output_path_resolved / f"{subject}_utility_cache_generations"  # Folder of generations
    output_generations_path.mkdir(parents=True, exist_ok=True)  # exist_ok=True for resume support

    judge_batch_completion_kwargs_parsed = json.loads(judge_batch_completion_kwargs)
    shared_kwargs = {
        "generation_kwargs": generation_kwargs_parsed,
        # This is hardcoded for our specific judges and that's fine since they do not change
        "n_samples": n_samples,
        "judge_model": judge_model,  # NOTE this is used for ALL judges
        "judge_max_new_tokens": judge_max_new_tokens,
        "judge_batch_size": judge_batch_size,
        "judge_batch_completion_kwargs": judge_batch_completion_kwargs_parsed,
        "error_threshold": 0.1,
        "subject": subject,
    }

    click.echo(f"Parsing checkpoints {len(checkpoints)} down to checkpoint folders:")
    for c in checkpoints:
        click.echo(f"  {c}")
    if not yes:
        click.confirm(f"Continue?", abort=True)
    else:
        click.echo(f"Continuing...")

    checkpoints_objs: list[Checkpoint] = get_relevant_checkpoints(checkpoints, subject, pruned_sae_dist_path)  # Includes vanilla, sft, sae-enhanced, sae-enhanced+sft (parse type, etc...)
    checkpoints_objs = sorted(checkpoints_objs, key=lambda x: (x.type, x.step))  # Group by types then within those groups by step
    kwargs_list = [EvalKwargs(**shared_kwargs, model_name_or_path=c.name_or_path, pruned_sae_dist_path=c.dist_path, pruned_sae_threshold=c.threshold) for c in checkpoints_objs]
    assert len(set(k.model_name_or_path for k in kwargs_list)) == len(kwargs_list), "Duplicate model names in kwargs list"
    for kw in kwargs_list:
        click.echo(f"  {kw.model_dump_json(indent=4)}")
    if not yes:
        click.confirm(f"Evaluating on {subject} with {len(kwargs_list)} kwargs. Continue?", abort=True)
    else:
        click.echo(f"Evaluating on {subject} with {len(kwargs_list)} kwargs. Continuing...")

    # Load previous results if input file provided and filter out already-evaluated kwargs
    previous_results: list[EvalResult] | None = None
    if input_file is not None:
        input_path = Path(input_file)
        if not input_path.is_file():
            assert input_path.is_dir(), f"Input path {input_path} is not a file or directory (why did you pass it?)"
            input_path = input_path / f"{subject}_utility_cache_overview.json"  # <--- fetch using default output path under folder
        if input_path.exists():
            previous_output = EvalOutput.model_validate_json(input_path.read_bytes())
            previous_results = previous_output.results
            evaluated_kwargs = [r.kwargs for r in previous_results]
            original_count = len(kwargs_list)
            # Quadratic checking because small data and also because not hashable without more code
            kwargs_list = [k for k in kwargs_list if not any(k == ek for ek in evaluated_kwargs)]
            skipped_count = original_count - len(kwargs_list)
            if skipped_count > 0:
                click.echo(f"Resuming: skipping {skipped_count} already-evaluated kwargs, {len(kwargs_list)} remaining")
        else:
            raise ValueError(f"Input file {input_path} does not exist (why did you pass it?)")

    # Get and store full results across various gpus
    gpu_ids_list: list[int] = sorted(set(int(gpu_id) for gpu_id in gpu_ids.split(",")))
    if len(kwargs_list) == 0:
        click.echo("All kwargs already evaluated, nothing to do")
    else:
        master_fn(
            kwargs_list,
            gpu_ids_list,
            output_overview_path,
            output_generations_path,
            previous_results=previous_results,
        )


if __name__ == "__main__":
    main()
