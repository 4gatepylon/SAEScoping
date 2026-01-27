"""Evaluate an LLM on math benchmarks: GSM8K and NuminaMath (AIMO)."""

from __future__ import annotations
import contextlib
import json
import os
import re
import traceback
import uuid
import warnings
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, Callable

import click
import tqdm
from beartype import beartype
from datasets import load_dataset
from pydantic import BaseModel

# TODO(Adriano) this is CC bloatware; we are going to need to consolidate everything into one abstraction soon to not fking die
# =============================================================================
# Pydantic models for multiprocessing
# =============================================================================


class MathEvalKwargs(BaseModel):
    """Arguments for a single math evaluation run."""

    model_path: str
    dataset: str  # "gsm8k" or "numinamath"
    limit: int | None
    batch_size: int
    max_tokens: int
    chat_template_path: str | None
    dist_path: str | None  # SAE distribution path
    threshold: float | None  # SAE pruning threshold
    seed: int
    generous: bool
    output_dir: str


class MathEvalResult(BaseModel):
    """Result from a single math evaluation run."""

    kwargs: MathEvalKwargs
    statistics: dict[str, Any]
    config: dict[str, Any]
    output_file: str  # Path to the full results file


class MathEvalOutput(BaseModel):
    """Combined output from all evaluations."""

    results: list[MathEvalResult]
    error_str: str | None
    traceback_str: str | None


# =============================================================================
# Default paths and model registry
# =============================================================================

# Default SAE distribution path (biology-based, used for math training)
DEFAULT_PRUNED_SAE_DIST_PATH = "/mnt/align4_drive2/adrianoh/scope_bench_spring_2026/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors"

# Base paths
_SAESCOPING_OUTPUTS = Path("/mnt/align4_drive2/adrianoh/git/SAEScoping/experiments/outputs_gemma9b")
_SCOPEBENCH_OUTPUTS = Path("/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b")

# Model shortcut registry - maps friendly names to (path_or_fn, needs_dist_path)
# If needs_dist_path is True, the model requires SAE hooks
MathSubject = Literal["gsm8k", "numinamath"]


@beartype
def _get_sae_enhanced_checkpoint_folder(subject: MathSubject) -> str:
    """Find the SAE-enhanced checkpoint folder for a subject (non-vanilla)."""
    parent = _SAESCOPING_OUTPUTS / subject
    if not parent.exists():
        raise FileNotFoundError(f"Subject folder not found: {parent}")
    children = [c for c in parent.iterdir() if c.is_dir() and c.stem != "vanilla"]
    if len(children) == 0:
        raise FileNotFoundError(f"No SAE-enhanced checkpoint folders found in {parent}")
    if len(children) > 1:
        # Pick the one with highest threshold in name (most recent usually)
        children.sort(key=lambda x: x.stem, reverse=True)
    return children[0].as_posix()


@beartype
def _get_latest_checkpoint(folder: str | Path) -> str:
    """Get the latest checkpoint (highest step number) from a folder."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    checkpoints = [d for d in folder.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {folder}")
    # Sort by step number (descending) and return highest
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]), reverse=True)
    return checkpoints[0].as_posix()


# Registry: name -> (path_resolver, needs_sae_dist_path)
# path_resolver can be: str (direct path), or Callable[[MathSubject], str]
MODEL_REGISTRY: dict[str, tuple[str | Callable[[MathSubject], str], bool]] = {
    # Vanilla Gemma (no finetuning)
    "vanilla": ("google/gemma-2-9b-it", False),
    "google/gemma-2-9b-it": ("google/gemma-2-9b-it", False),
    # SFT checkpoints (finetuned on subject, no SAE)
    "sft": (lambda subj: _get_latest_checkpoint(_SAESCOPING_OUTPUTS / subj / "vanilla"), False),
    "sft_folder": (lambda subj: (_SAESCOPING_OUTPUTS / subj / "vanilla").as_posix(), False),
    # SAE-enhanced checkpoints
    "sae": (lambda subj: _get_latest_checkpoint(_get_sae_enhanced_checkpoint_folder(subj)), True),
    "sae_folder": (lambda subj: _get_sae_enhanced_checkpoint_folder(subj), True),
    # Old ultrachat-based SAE model (from ScopeBench)
    "sae_ultrachat": (
        (_SCOPEBENCH_OUTPUTS / "ultrachat" / "layer_31_width_16k_canonical_h0.0001_85cac49528" / "checkpoint-2000").as_posix(),
        True,
    ),
}


@beartype
def resolve_model_path(model: str, dataset: MathSubject) -> tuple[str, bool]:
    """
    Resolve a model shortcut or path to actual path and whether it needs SAE dist.

    Returns: (model_path, needs_sae_dist_path)
    """
    if model in MODEL_REGISTRY:
        resolver, needs_dist = MODEL_REGISTRY[model]
        if callable(resolver):
            return resolver(dataset), needs_dist
        return resolver, needs_dist

    # Not in registry - treat as direct path
    # Check if it looks like an SAE-enhanced checkpoint (has _h in path)
    needs_dist = bool(re.search(r"_h\d", model))
    return model, needs_dist


@beartype
def extract_threshold_from_path(checkpoint_path: str) -> float | None:
    """Extract threshold (h value) from checkpoint path if present."""
    # Match _h followed by a float (decimal or scientific notation)
    match = re.search(r"_h(\d+\.?\d*(?:e[+-]?\d+)?)", checkpoint_path, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


@beartype
def list_available_checkpoints(subject: MathSubject) -> None:
    """Print available checkpoints for a subject."""
    click.echo(f"\nAvailable checkpoints for {subject}:")
    click.echo("-" * 60)

    # Vanilla
    click.echo("  vanilla (google/gemma-2-9b-it) - Base Gemma model")

    # SFT checkpoints
    sft_folder = _SAESCOPING_OUTPUTS / subject / "vanilla"
    if sft_folder.exists():
        ckpts = sorted([d.name for d in sft_folder.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
        if ckpts:
            click.echo(f"  sft - Latest SFT checkpoint ({ckpts[-1]})")
            click.echo(f"  sft_folder - All SFT checkpoints: {', '.join(ckpts)}")

    # SAE-enhanced checkpoints
    try:
        sae_folder = _get_sae_enhanced_checkpoint_folder(subject)
        sae_path = Path(sae_folder)
        ckpts = sorted([d.name for d in sae_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
        if ckpts:
            click.echo(f"  sae - Latest SAE checkpoint ({ckpts[-1]})")
            click.echo(f"  sae_folder - All SAE checkpoints: {', '.join(ckpts)}")
    except FileNotFoundError:
        click.echo("  sae - No SAE-enhanced checkpoints found")

    click.echo("  sae_ultrachat - Ultrachat-based SAE model")
    click.echo("-" * 60)


# =============================================================================
# Answer extraction functions
# =============================================================================


@beartype
def extract_gsm8k_answer(text: str) -> str | None:
    """
    Extract the final answer from GSM8K format: #### {number}

    Handles:
    - Negative numbers: #### -42
    - Decimals: #### 3.14
    - Commas: #### 1,000,000
    """
    # Look for #### followed by a number (with optional negative, commas, decimal)
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        answer = match.group(1).replace(",", "").strip()
        return answer
    return None


@beartype
def extract_boxed_answer(text: str, location_mode: Literal["any", "last"] = "last") -> str | None:
    r"""
    Extract the content from \boxed{...} format used in NuminaMath.

    Handles nested braces like \boxed{x^{2}+1}.
    """
    # Simple pattern for non-nested content
    simple_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)

    # More complex pattern for potentially nested braces (one level)
    nested_matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)

    # Combine and dedupe while preserving order
    all_matches = []
    seen = set()
    for m in simple_matches + nested_matches:
        if m not in seen:
            all_matches.append(m)
            seen.add(m)

    if not all_matches:
        return None

    if location_mode == "last":
        return all_matches[-1].strip()
    elif location_mode == "any":
        return all_matches[0].strip()  # Return first for "any"
    else:
        raise ValueError(f"Invalid location_mode: {location_mode}")


@beartype
def normalize_numeric_answer(answer: str) -> str:
    """Normalize a numeric answer for comparison."""
    # Remove whitespace
    answer = answer.strip()
    # Remove commas
    answer = answer.replace(",", "")
    # Remove leading zeros before decimal (but keep "0.5" as is)
    if "." in answer:
        parts = answer.split(".")
        if len(parts) == 2:
            int_part = parts[0].lstrip("0") or "0"
            answer = f"{int_part}.{parts[1]}"
    return answer


@beartype
def compare_answers(pred: str | None, gold: str | None, strict: bool = False) -> bool:
    """
    Compare predicted and gold answers.

    For numeric answers, normalizes before comparison.
    For non-numeric (algebraic), does case-insensitive string comparison.
    """
    if pred is None or gold is None:
        return False

    pred_norm = normalize_numeric_answer(pred)
    gold_norm = normalize_numeric_answer(gold)

    # Try numeric comparison first
    try:
        pred_float = float(pred_norm)
        gold_float = float(gold_norm)
        # Use small epsilon for floating point comparison
        return abs(pred_float - gold_float) < 1e-9
    except ValueError:
        pass

    # Fall back to string comparison (case-insensitive for algebraic expressions)
    if strict:
        return pred_norm == gold_norm
    return pred_norm.lower() == gold_norm.lower()


# =============================================================================
# Dataset loading
# =============================================================================

# TODO(adriano) not sure if this instruction should be optimized tbh
MATH_SYSTEM_PROMPT = """You are a helpful math assistant. Solve the problem step by step, showing your work clearly."""

GSM8K_INSTRUCTION = """Solve this math problem step by step. After your solution, write your final numerical answer on a new line in the format: #### <answer>

For example, if the answer is 42, end with:
#### 42"""

NUMINAMATH_INSTRUCTION = r"""Solve this math problem step by step. After your solution, put your final answer inside \boxed{}.

For example, if the answer is 42, end with:
\boxed{42}"""


@beartype
def format_math_prompt(
    question: str,
    dataset_type: Literal["gsm8k", "numinamath"],
    include_system_prompt: bool = True,
) -> list[dict[str, str]]:
    """Format a math problem as chat messages."""
    messages = []

    if include_system_prompt:
        messages.append({"role": "system", "content": MATH_SYSTEM_PROMPT})

    instruction = GSM8K_INSTRUCTION if dataset_type == "gsm8k" else NUMINAMATH_INSTRUCTION
    content = f"{instruction}\n\nProblem: {question}"
    messages.append({"role": "user", "content": content})

    return messages


@beartype
def load_gsm8k_dataset(
    limit: int | None,
    seed: int = 1,
) -> tuple[list[list[dict[str, str]]], list[str], str]:
    """
    Load GSM8K test dataset.

    Returns: (prompts, ground_truth_answers, dataset_name)
    """
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    prompts = []
    ground_truth = []

    for row in dataset:
        prompt = format_math_prompt(row["question"], "gsm8k")
        prompts.append(prompt)

        # Extract answer from the solution (format: "... #### {number}")
        answer = extract_gsm8k_answer(row["answer"])
        if answer is None:
            # Fallback: try to get the last number in the answer
            numbers = re.findall(r"-?[\d,]+\.?\d*", row["answer"])
            answer = numbers[-1].replace(",", "") if numbers else "0"
        ground_truth.append(answer)

    return prompts, ground_truth, "openai/gsm8k (main, test)"


@beartype
def load_numinamath_dataset(
    limit: int | None,
    seed: int = 1,
    exclude_proofs: bool = True,
) -> tuple[list[list[dict[str, str]]], list[str], str]:
    """
    Load NuminaMath dataset for evaluation.

    Returns: (prompts, ground_truth_answers, dataset_name)
    """
    dataset = load_dataset("AI-MO/NuminaMath-1.5", split="train")

    # Filter for valid problems with answers
    dataset = dataset.filter(lambda x: x["problem_is_valid"] and x["solution_is_valid"])
    dataset = dataset.filter(lambda x: x["answer"] is not None and isinstance(x["answer"], str) and len(x["answer"].strip()) > 0)

    if exclude_proofs:
        dataset = dataset.filter(lambda x: x.get("question_type") != "proof")
        dataset = dataset.filter(lambda x: x["answer"].lower() != "proof")

    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    prompts = []
    ground_truth = []

    for row in dataset:
        prompt = format_math_prompt(row["problem"], "numinamath")
        prompts.append(prompt)
        ground_truth.append(row["answer"].strip())

    return prompts, ground_truth, "AI-MO/NuminaMath-1.5 (train)"


@beartype
def load_dataset_by_name(
    dataset: str,
    limit: int | None,
    seed: int = 1,
) -> tuple[list[list[dict[str, str]]], list[str], str]:
    """Router function to load the appropriate dataset."""
    if dataset == "gsm8k":
        return load_gsm8k_dataset(limit=limit, seed=seed)
    elif dataset == "numinamath":
        return load_numinamath_dataset(limit=limit, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: gsm8k, numinamath")


# =============================================================================
# Grading
# =============================================================================


@beartype
def grade_response(
    response: str,
    ground_truth: str,
    dataset_type: Literal["gsm8k", "numinamath"],
) -> tuple[bool, bool, str | None]:
    """
    Grade a model response.

    Returns: (is_correct, is_invalid, extracted_answer)
    """
    if dataset_type == "gsm8k":
        extracted = extract_gsm8k_answer(response)
        if extracted is None:
            # Fallback: try to find boxed answer (some models use this format)
            extracted = extract_boxed_answer(response)
        if extracted is None:
            # Last resort: find the last number in the response
            numbers = re.findall(r"-?[\d,]+\.?\d*", response)
            if numbers:
                extracted = numbers[-1].replace(",", "")
    else:  # numinamath
        extracted = extract_boxed_answer(response)
        if extracted is None:
            # Fallback: try GSM8K format
            extracted = extract_gsm8k_answer(response)

    if extracted is None:
        return False, True, None

    is_correct = compare_answers(extracted, ground_truth)
    return is_correct, False, extracted


@beartype
def grade_response_generous(
    response: str,
    ground_truth: str,
) -> tuple[bool, bool, str | None]:
    """
    Grade a model response using all extractors generously.

    Tries all extraction methods (GSM8K, boxed, last number) and gives credit
    if ANY extraction matches the ground truth.

    Returns: (is_correct, is_invalid, extracted_answer)
    """

    # All extractors to try, in order of preference
    def extract_last_number(r: str) -> str | None:
        numbers = re.findall(r"-?[\d,]+\.?\d*", r)
        return numbers[-1].replace(",", "") if numbers else None

    extractors: list[tuple[str, Callable[[str], str | None]]] = [
        ("gsm8k", extract_gsm8k_answer),
        ("boxed", extract_boxed_answer),
        ("last_number", extract_last_number),
    ]

    # Try each extractor and check if it matches
    all_extracted: list[tuple[str, str | None]] = []
    for name, extractor in extractors:
        extracted = extractor(response)
        all_extracted.append((name, extracted))
        if extracted is not None and compare_answers(extracted, ground_truth):
            return True, False, extracted

    # None matched - return the first non-None extraction (or None if all failed)
    for name, extracted in all_extracted:
        if extracted is not None:
            return False, False, extracted

    # All extractors returned None - invalid response
    return False, True, None


# =============================================================================
# All-checkpoints enumeration
# =============================================================================


@beartype
def get_all_model_configs() -> list[tuple[str, str, bool]]:
    """
    Enumerate all available model configurations for comprehensive evaluation.

    Returns list of (model_name, model_path_or_shortcut, needs_sae_dist).
    The model_name is a friendly identifier for the configuration.
    """
    configs: list[tuple[str, str, bool]] = []

    # 1. Vanilla Gemma (base model, no finetuning)
    configs.append(("vanilla", "google/gemma-2-9b-it", False))

    # 2. SFT checkpoints from SAEScoping (one per subject)
    for subject in ["gsm8k", "numinamath"]:
        sft_folder = _SAESCOPING_OUTPUTS / subject / "vanilla"
        if sft_folder.exists():
            try:
                latest = _get_latest_checkpoint(sft_folder)
                configs.append((f"sft_{subject}", latest, False))
            except FileNotFoundError:
                pass

    # 3. SAE-enhanced checkpoints from SAEScoping (one per subject)
    for subject in ["gsm8k", "numinamath"]:
        try:
            sae_folder = _get_sae_enhanced_checkpoint_folder(subject)  # type: ignore[arg-type]
            latest = _get_latest_checkpoint(sae_folder)
            configs.append((f"sae_{subject}", latest, True))
        except FileNotFoundError:
            pass

    # 4. SAE models from ScopeBench
    scopebench_ultrachat = _SCOPEBENCH_OUTPUTS / "ultrachat"
    if scopebench_ultrachat.exists():
        # Find all SAE-enhanced folders (contain _h in name)
        for folder in scopebench_ultrachat.iterdir():
            if folder.is_dir() and "_h" in folder.name:
                try:
                    latest = _get_latest_checkpoint(folder)
                    # Create friendly name from folder
                    configs.append((f"scopebench_ultrachat_{folder.name[:20]}", latest, True))
                except FileNotFoundError:
                    pass

    return configs


# =============================================================================
# Multiprocessing support
# =============================================================================

GEMMA2_CHAT_TEMPLATE_WITH_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "sae_scoping" / "utils" / "gemma2" / "chat_template_with_system_prompt.jinja"
CHAT_TEMPLATE_PATH_REGISTRY = {
    "gemma-2-9b-it-sys": GEMMA2_CHAT_TEMPLATE_WITH_SYSTEM_PROMPT_PATH,
}


@beartype
def evaluate_single_math_config(kwargs: MathEvalKwargs) -> MathEvalResult:
    """
    Run a single math evaluation. This is the core function that loads a model and evaluates it.

    IMPORTANT: This must be called AFTER setting CUDA_VISIBLE_DEVICES in a worker process.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_scoping.utils.generation.hf_generator import HFGenerator
    from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks

    model_path = kwargs.model_path
    dataset = kwargs.dataset
    dataset_typed: MathSubject = dataset  # type: ignore[assignment]

    # Setup output path
    output_path = Path(kwargs.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create output filename
    model_name_safe = model_path.replace("/", "_").replace("\\", "_")
    if len(model_name_safe) > 50:
        import hashlib

        model_name_safe = hashlib.sha256(model_path.encode()).hexdigest()[:16]
    grading_suffix = "_generous" if kwargs.generous else ""
    output_file = output_path / f"{dataset}_{model_name_safe}{grading_suffix}_results.json"

    # Load dataset
    prompts, ground_truth, dataset_name = load_dataset_by_name(dataset=dataset, limit=kwargs.limit, seed=kwargs.seed)

    # Load model and tokenizer
    model_obj = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Critical for Gemma2
    )
    tokenizer_obj = AutoTokenizer.from_pretrained(model_path)
    tokenizer_obj.padding_side = "left"

    # Apply custom chat template if provided
    chat_template_path = kwargs.chat_template_path
    if chat_template_path is not None:
        if chat_template_path in CHAT_TEMPLATE_PATH_REGISTRY:
            chat_template_path = str(CHAT_TEMPLATE_PATH_REGISTRY[chat_template_path])
        template_path = Path(chat_template_path)
        if template_path.exists():
            tokenizer_obj.chat_template = template_path.read_text()
        else:
            raise FileNotFoundError(f"Chat template not found: {chat_template_path}")  # why pass if no exist wtf

    # Setup SAE hooks if needed
    hooking_context = contextlib.nullcontext()
    sae_metadata: dict[str, Any] = {}

    if kwargs.dist_path is not None:
        from sae_scoping.evaluation.hardcoded_biology.utility_1click_judgement import get_pruned_sae

        threshold = kwargs.threshold if kwargs.threshold is not None else 1e-4
        pruned_sae, hookpoint, n_kept = get_pruned_sae(kwargs.dist_path, threshold, device=model_obj.device)
        hooking_context = named_forward_hooks(model_obj, {hookpoint: partial(filter_hook_fn, pruned_sae)})
        sae_metadata = {
            "dist_path": kwargs.dist_path,
            "threshold": threshold,
            "hookpoint": hookpoint,
            "n_kept": n_kept,
        }

    # Run inference and grading
    completions = []
    correct = 0
    invalid = 0
    total = len(prompts)

    generation_kwargs_dict = {
        "do_sample": False,
        "max_new_tokens": kwargs.max_tokens,
    }

    with hooking_context:
        generator = HFGenerator(model_obj, tokenizer_obj)
        responses_iterator = generator.generate_stream(
            prompts,
            batch_size=kwargs.batch_size,
            generation_kwargs=generation_kwargs_dict,
        )

        for i, (response, gt) in enumerate(zip(responses_iterator, ground_truth)):
            if kwargs.generous:
                is_correct, is_invalid, extracted = grade_response_generous(
                    response=response,
                    ground_truth=gt,
                )
            else:
                is_correct, is_invalid, extracted = grade_response(
                    response=response,
                    ground_truth=gt,
                    dataset_type=dataset_typed,
                )

            correct += int(is_correct)
            invalid += int(is_invalid)

            completions.append(
                {
                    "prompt": prompts[i],
                    "response": response,
                    "ground_truth": gt,
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                    "is_invalid": is_invalid,
                }
            )

    # Compute statistics
    accuracy = correct / total if total > 0 else 0.0
    valid_total = total - invalid
    conditional_accuracy = correct / valid_total if valid_total > 0 else 0.0

    statistics = {
        "dataset": dataset,
        "dataset_name": dataset_name,
        "model": model_path,
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy": accuracy,
        "conditional_accuracy": conditional_accuracy,
    }

    config = {
        "model": model_path,
        "dataset": dataset,
        "limit": kwargs.limit,
        "batch_size": kwargs.batch_size,
        "max_tokens": kwargs.max_tokens,
        "chat_template_path": kwargs.chat_template_path,
        "seed": kwargs.seed,
        "generous_grading": kwargs.generous,
        **sae_metadata,
    }

    # Save full results
    results = {
        "completions": completions,
        "statistics": statistics,
        "config": config,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    return MathEvalResult(
        kwargs=kwargs,
        statistics=statistics,
        config=config,
        output_file=str(output_file),
    )


@beartype
def process_kwargs_list(kwargs_list: list[MathEvalKwargs], save_to_dir: Path) -> None:
    """Process a list of kwargs and save results to temp directory."""
    for kwargs in kwargs_list:
        result = evaluate_single_math_config(kwargs)
        output_path = save_to_dir / f"{uuid.uuid4()}.json"
        output_path.write_bytes(result.model_dump_json().encode())


def worker_fn(args: dict[str, Any]) -> None:
    """Worker function that runs in a subprocess with specific GPU."""
    gpu_id = args["gpu_id"]
    kwargs_list = args["kwargs_list"]
    temp_dir = Path(args["temp_dir"])
    temp_dir.mkdir(parents=True, exist_ok=True)

    # CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing torch/transformers
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Now process kwargs (heavy imports happen inside evaluate_single_math_config)
    process_kwargs_list(kwargs_list, temp_dir)


def master_fn(
    kwargs_list: list[MathEvalKwargs],
    gpu_ids: list[int],
    output_path: Path,
    previous_results: list[MathEvalResult] | None = None,
) -> None:
    """Distribute work across GPUs and collect results."""
    if len(kwargs_list) == 0:
        raise ValueError("No kwargs provided to evaluate")
    gpu_ids = gpu_ids[: len(kwargs_list)]
    if len(gpu_ids) == 0:
        raise ValueError("No GPU IDs provided")

    with TemporaryDirectory() as temp_dir_str:
        # Distribute kwargs across GPUs (round-robin)
        kwargs_list_chunks = [deepcopy(kwargs_list[i :: len(gpu_ids)]) for i in range(len(gpu_ids))]
        # Filter out empty chunks
        non_empty = [(gpu_id, chunk) for gpu_id, chunk in zip(gpu_ids, kwargs_list_chunks) if len(chunk) > 0]

        worker_kwargs_list = [
            {
                "gpu_id": gpu_id,
                "kwargs_list": chunk,
                "temp_dir": temp_dir_str,
            }
            for gpu_id, chunk in non_empty
        ]

        err, tb, err_obj = None, None, None
        try:
            with Pool(len(non_empty)) as pool:
                pool.map(worker_fn, worker_kwargs_list)
        except Exception as e:
            err, tb, err_obj = str(e), traceback.format_exc(), e

        # Collect results from temp directory
        temp_dir = Path(temp_dir_str)
        json_filepaths = list(temp_dir.glob("*.json"))
        results = [MathEvalResult.model_validate_json(fp.read_bytes()) for fp in json_filepaths]

        if previous_results is not None:
            results = previous_results + results

        output = MathEvalOutput(results=results, error_str=err, traceback_str=tb)

        # Atomic write
        tmp_output_path = output_path.with_suffix(f".tmp.{uuid.uuid4()}.json")
        try:
            tmp_output_path.write_bytes(output.model_dump_json().encode())
            tmp_output_path.rename(output_path)
        except Exception as e:
            if err_obj is None:
                err, tb, err_obj = str(e), traceback.format_exc(), e
            else:
                warnings.warn(f"Error writing output to {tmp_output_path} AFTER previous error")
                warnings.warn(traceback.format_exc())
        finally:
            tmp_output_path.unlink(missing_ok=True)

        if err_obj is not None:
            raise err_obj


# =============================================================================
# Main CLI
# =============================================================================


@click.command()
@click.option(
    "--model",
    "-m",
    type=str,
    default="vanilla",
    help="Model path or shortcut (vanilla, sft, sae, sae_ultrachat, sft_folder, sae_folder, 'all', or full path)",
)
@click.option(
    "--dataset",
    "-ds",
    type=click.Choice(["gsm8k", "numinamath", "all"]),
    default="gsm8k",
    help="Dataset to evaluate on (default: gsm8k, use 'all' for both)",
)
@click.option("--limit", "-l", type=int, default=None, help="Limit number of samples to evaluate")
@click.option("--batch-size", "-b", type=int, default=8, help="Batch size for inference (default: 8)")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./math_eval_results",
    help="Directory to save results (default: ./math_eval_results)",
)
@click.option("--max-tokens", "-mt", type=int, default=1024, help="Max tokens for generation")
@click.option(
    "--chat-template-path",
    "-ctp",
    type=str,
    default="gemma-2-9b-it-sys",
    help="Path to chat template (default: gemma-2-9b-it-sys for Gemma with system prompt)",
)
@click.option(
    "--dist-path",
    "-p",
    type=str,
    default=None,
    help=f"Path to distribution safetensors for SAE pruning (default: {DEFAULT_PRUNED_SAE_DIST_PATH} if model needs it)",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=None,
    help="Threshold for SAE pruning (inferred from model path if not provided, default: 1e-4)",
)
@click.option("--seed", type=int, default=1, help="Random seed for dataset shuffling")
@click.option("--clobber", "-c", is_flag=True, default=False, help="Overwrite existing output file")
@click.option("--list-models", "-lm", is_flag=True, default=False, help="List available model shortcuts and exit")
@click.option(
    "--generous",
    "-g",
    is_flag=True,
    default=False,
    help="Use generous grading: give credit if ANY extractor (GSM8K, boxed, last number) matches",
)
@click.option(
    "--gpu-ids",
    "-gpu",
    type=str,
    default=None,
    help="Comma-separated GPU IDs for parallel evaluation (e.g., '0,1,2'). If provided, uses multiprocessing.",
)
@beartype
def main(
    model: str,
    dataset: str,  # Can be "gsm8k", "numinamath", or "all"
    limit: int | None,
    batch_size: int,
    output_dir: str,
    max_tokens: int,
    chat_template_path: str | None,
    dist_path: str | None,
    threshold: float | None,
    seed: int,
    clobber: bool,
    list_models: bool,
    generous: bool,
    gpu_ids: str | None,
) -> None:
    """
    Evaluate a HuggingFace model on math benchmarks.

    Model shortcuts available:
      - vanilla: Base google/gemma-2-9b-it model
      - sft: Latest SFT checkpoint (finetuned on dataset, no SAE)
      - sae: Latest SAE-enhanced checkpoint
      - sae_ultrachat: Ultrachat-based SAE model
      - sft_folder / sae_folder: Evaluate all checkpoints in folder
      - all: Evaluate ALL available models (vanilla, SFT, SAE from SAEScoping + ScopeBench)

    Examples:

    \b
    # Quick eval on GSM8K with vanilla model (100 samples)
    python script_2026_01_25_math_eval.py -ds gsm8k -m vanilla -l 100

    \b
    # Eval on GSM8K with SFT model
    python script_2026_01_25_math_eval.py -ds gsm8k -m sft -l 100

    \b
    # Eval on NuminaMath with SAE-enhanced model
    python script_2026_01_25_math_eval.py -ds numinamath -m sae -l 100

    \b
    # Full eval on GSM8K test set with vanilla model
    python script_2026_01_25_math_eval.py -ds gsm8k -m vanilla

    \b
    # List available model shortcuts
    python script_2026_01_25_math_eval.py -ds gsm8k --list-models

    \b
    # COMPREHENSIVE: Eval ALL models on BOTH datasets with generous grading
    python script_2026_01_25_math_eval.py -ds all -m all -g -l 100

    \b
    # Eval vanilla on both datasets with generous grading
    python script_2026_01_25_math_eval.py -ds all -m vanilla -g -l 100
    """
    # Handle --list-models flag
    if list_models:
        if dataset == "all":
            for ds in ["gsm8k", "numinamath"]:
                list_available_checkpoints(ds)  # type: ignore[arg-type]
        else:
            list_available_checkpoints(dataset)  # type: ignore[arg-type]
        # Also list all model configs
        click.echo("\nAll available model configurations (for -m all):")
        click.echo("-" * 60)
        for name, path, needs_sae in get_all_model_configs():
            sae_marker = " [SAE]" if needs_sae else ""
            click.echo(f"  {name}: {path[:60]}...{sae_marker}" if len(path) > 60 else f"  {name}: {path}{sae_marker}")
        click.echo("-" * 60)
        return

    # =========================================================================
    # Multiprocessing path: when --gpu-ids is provided
    # =========================================================================
    if gpu_ids is not None:
        gpu_ids_list: list[int] = sorted(set(int(g) for g in gpu_ids.split(",")))
        click.echo(f"Using multiprocessing with GPUs: {gpu_ids_list}")

        # Build list of all (model, dataset) combinations to evaluate
        kwargs_list: list[MathEvalKwargs] = []

        # Determine datasets to evaluate
        datasets_to_eval = ["gsm8k", "numinamath"] if dataset == "all" else [dataset]

        # Determine models to evaluate
        if model == "all":
            model_configs = get_all_model_configs()
        else:
            raise NotImplementedError("Single model evaluation is not supported with multiprocessing")  # TODO(Adriano) not read lol
            # Single model - resolve once (use first dataset for path resolution)
            ds_for_resolve: MathSubject = datasets_to_eval[0]  # type: ignore[assignment]
            resolved_path, needs_dist = resolve_model_path(model, ds_for_resolve)

            # Handle folder-based models
            if model.endswith("_folder"):
                folder_path = Path(resolved_path)
                checkpoints = [d for d in folder_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
                model_configs = [(ckpt.name, ckpt.as_posix(), needs_dist) for ckpt in checkpoints]
            else:
                model_configs = [(model, resolved_path, needs_dist)]

        # Build kwargs for each (model, dataset) combination
        for ds in datasets_to_eval:
            for model_name, model_path, needs_dist in model_configs:
                # Resolve dist_path and threshold
                this_dist_path = dist_path
                this_threshold = threshold
                if needs_dist:
                    if this_dist_path is None:
                        this_dist_path = DEFAULT_PRUNED_SAE_DIST_PATH
                    if this_threshold is None:
                        this_threshold = extract_threshold_from_path(model_path)
                        if this_threshold is None:
                            this_threshold = 1e-4

                kwargs_list.append(
                    MathEvalKwargs(
                        model_path=model_path,
                        dataset=ds,
                        limit=limit,
                        batch_size=batch_size,
                        max_tokens=max_tokens,
                        chat_template_path=chat_template_path,
                        dist_path=this_dist_path if needs_dist else None,
                        threshold=this_threshold if needs_dist else None,
                        seed=seed,
                        generous=generous,
                        output_dir=output_dir,
                    )
                )

        if len(kwargs_list) == 0:
            click.echo("No configurations to evaluate")
            return

        click.echo(f"Evaluating {len(kwargs_list)} configurations across {len(gpu_ids_list)} GPUs")
        for i, kw in enumerate(kwargs_list):
            click.echo(f"  [{i + 1}] {kw.dataset} / {kw.model_path[:60]}...")

        # Run evaluation using multiprocessing
        overview_output_path = Path(output_dir) / "math_eval_overview.json"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        master_fn(kwargs_list, gpu_ids_list, overview_output_path)

        # Print summary
        if overview_output_path.exists():
            output = MathEvalOutput.model_validate_json(overview_output_path.read_bytes())
            click.echo(f"\n{'=' * 60}")
            click.echo(f"Completed {len(output.results)} evaluations")
            for result in output.results:
                acc = result.statistics.get("accuracy", 0)
                click.echo(f"  {result.kwargs.dataset} / {result.kwargs.model_path[:40]}... : {acc:.2%}")
            if output.error_str:
                click.echo(f"Errors: {output.error_str}")
            click.echo("=" * 60)
        return

    # =========================================================================
    # Single-process path (original behavior)
    # =========================================================================
    raise NotImplementedError("Single-process path is not supported with multiprocessing")  # TODO(Adriano) not read lol
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sae_scoping.utils.generation.hf_generator import HFGenerator
    from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks

    # Handle --dataset all: iterate over all datasets
    if dataset == "all":
        datasets_to_eval: list[str] = ["gsm8k", "numinamath"]
        click.echo(f"Evaluating on ALL datasets: {datasets_to_eval}")
        for ds in datasets_to_eval:
            click.echo(f"\n{'=' * 60}\nDataset: {ds}\n{'=' * 60}")
            ctx = click.Context(main)
            ctx.invoke(
                main,
                model=model,
                dataset=ds,
                limit=limit,
                batch_size=batch_size,
                output_dir=output_dir,
                max_tokens=max_tokens,
                chat_template_path=chat_template_path,
                dist_path=dist_path,
                threshold=threshold,
                seed=seed,
                clobber=clobber,
                list_models=False,
                generous=generous,
                gpu_ids=None,
            )
        return

    # Handle --model all: iterate over all model configurations
    if model == "all":
        all_configs = get_all_model_configs()
        click.echo(f"Evaluating ALL models ({len(all_configs)} configurations)")
        for model_name, model_path_or_shortcut, needs_sae in all_configs:
            click.echo(f"\n{'=' * 60}\nModel: {model_name}\n{'=' * 60}")
            ctx = click.Context(main)
            try:
                ctx.invoke(
                    main,
                    model=model_path_or_shortcut,
                    dataset=dataset,
                    limit=limit,
                    batch_size=batch_size,
                    output_dir=output_dir,
                    max_tokens=max_tokens,
                    chat_template_path=chat_template_path,
                    dist_path=dist_path if needs_sae else None,
                    threshold=threshold,
                    seed=seed,
                    clobber=clobber,
                    list_models=False,
                    generous=generous,
                    gpu_ids=None,
                )
            except Exception as e:
                click.echo(f"ERROR evaluating {model_name}: {e}")
                continue
        return

    # 0. Resolve model shortcut to actual path
    # Cast dataset to MathSubject for type checking (we've already handled "all" above)
    dataset_typed: MathSubject = dataset  # type: ignore[assignment]
    model_path, needs_dist = resolve_model_path(model, dataset_typed)
    click.echo(f"Resolved model '{model}' -> '{model_path}'")

    # Handle folder-based models (evaluate all checkpoints)
    if model.endswith("_folder"):
        folder_path = Path(model_path)
        checkpoints = sorted(
            [d for d in folder_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {folder_path}")
        click.echo(f"Found {len(checkpoints)} checkpoints to evaluate: {[c.name for c in checkpoints]}")

        for ckpt in checkpoints:
            click.echo(f"\n{'=' * 60}\nEvaluating checkpoint: {ckpt.name}\n{'=' * 60}")
            # Recursively call main with specific checkpoint
            ctx = click.Context(main)
            ctx.invoke(
                main,
                model=ckpt.as_posix(),
                dataset=dataset,
                limit=limit,
                batch_size=batch_size,
                output_dir=output_dir,
                max_tokens=max_tokens,
                chat_template_path=chat_template_path,
                dist_path=dist_path,
                threshold=threshold,
                seed=seed,
                clobber=clobber,
                list_models=False,
                generous=generous,
                gpu_ids=None,
            )
        return

    # Auto-set dist_path if model needs it
    if needs_dist and dist_path is None:
        dist_path = DEFAULT_PRUNED_SAE_DIST_PATH
        click.echo(f"Auto-setting dist_path to: {dist_path}")

    # Auto-set threshold from model path if needed
    if needs_dist and threshold is None:
        threshold = extract_threshold_from_path(model_path)
        if threshold is None:
            threshold = 1e-4  # Default threshold
            click.echo(f"Using default threshold: {threshold}")
        else:
            click.echo(f"Extracted threshold from path: {threshold}")

    # 1. Setup output path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create output filename based on model and dataset
    model_name_safe = model_path.replace("/", "_").replace("\\", "_")
    if len(model_name_safe) > 50:
        import hashlib

        model_name_safe = hashlib.sha256(model_path.encode()).hexdigest()[:16]
    grading_suffix = "_generous" if generous else ""
    output_file = output_path / f"{dataset}_{model_name_safe}{grading_suffix}_results.json"

    if output_file.exists() and not clobber:
        raise FileExistsError(f"Output file already exists: {output_file}. Use --clobber to overwrite.")

    # 2. Load dataset
    click.echo(f"Loading dataset: {dataset}")
    prompts, ground_truth, dataset_name = load_dataset_by_name(dataset=dataset, limit=limit, seed=seed)
    click.echo(f"Loaded {len(prompts)} samples from {dataset_name}")

    # 3. Load model and tokenizer
    click.echo(f"Loading model: {model_path}")
    model_obj = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    tokenizer_obj = AutoTokenizer.from_pretrained(model_path)
    tokenizer_obj.padding_side = "left"

    # Apply custom chat template if provided
    if chat_template_path is not None:
        if chat_template_path in CHAT_TEMPLATE_PATH_REGISTRY:
            chat_template_path = CHAT_TEMPLATE_PATH_REGISTRY[chat_template_path]
        template_path = Path(chat_template_path)
        if not template_path.exists():
            raise FileNotFoundError(f"Chat template not found: {chat_template_path}")
        tokenizer_obj.chat_template = template_path.read_text()
        click.echo(f"Applied custom chat template from: {chat_template_path}")

    # 4. Setup SAE hooks if needed
    hooking_context = contextlib.nullcontext()
    sae_metadata = {}

    if dist_path is not None:
        from sae_scoping.evaluation.hardcoded_biology.utility_1click_judgement import get_pruned_sae

        # Threshold should already be set from earlier auto-detection
        if threshold is None:
            # Last resort: try to extract from model path
            threshold = extract_threshold_from_path(model_path)
            if threshold is None:
                threshold = 1e-4
            click.echo(f"Using threshold: {threshold}")

        pruned_sae, hookpoint, n_kept = get_pruned_sae(dist_path, threshold, device=model_obj.device)
        click.echo(f"SAE hookpoint: {hookpoint}, neurons kept: {n_kept}")

        hooking_context = named_forward_hooks(model_obj, {hookpoint: partial(filter_hook_fn, pruned_sae)})
        sae_metadata = {
            "dist_path": dist_path,
            "threshold": threshold,
            "hookpoint": hookpoint,
            "n_kept": n_kept,
        }

    # 5. Run inference and grading
    grading_mode = "generous (any extractor)" if generous else f"strict ({dataset})"
    click.echo(f"Running inference with batch_size={batch_size}, max_tokens={max_tokens}, grading={grading_mode}")

    completions = []
    correct = 0
    invalid = 0
    total = len(prompts)

    generation_kwargs = {
        "do_sample": False,
        "max_new_tokens": max_tokens,
    }

    with hooking_context:
        generator = HFGenerator(model_obj, tokenizer_obj)
        responses_iterator = generator.generate_stream(
            prompts,
            batch_size=batch_size,
            generation_kwargs=generation_kwargs,
        )

        for i, (response, gt) in tqdm.tqdm(
            enumerate(zip(responses_iterator, ground_truth)),
            total=total,
            desc="Evaluating",
        ):
            if generous:
                is_correct, is_invalid, extracted = grade_response_generous(
                    response=response,
                    ground_truth=gt,
                )
            else:
                is_correct, is_invalid, extracted = grade_response(
                    response=response,
                    ground_truth=gt,
                    dataset_type=dataset_typed,
                )

            correct += int(is_correct)
            invalid += int(is_invalid)

            completions.append(
                {
                    "prompt": prompts[i],
                    "response": response,
                    "ground_truth": gt,
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                    "is_invalid": is_invalid,
                }
            )

    # 6. Compute statistics
    accuracy = correct / total if total > 0 else 0.0
    valid_total = total - invalid
    conditional_accuracy = correct / valid_total if valid_total > 0 else 0.0

    statistics = {
        "dataset": dataset,
        "dataset_name": dataset_name,
        "model": model_path,
        "model_shortcut": model,
        "total": total,
        "correct": correct,
        "invalid": invalid,
        "accuracy": accuracy,
        "conditional_accuracy": conditional_accuracy,
    }

    config = {
        "model": model_path,
        "model_shortcut": model,
        "dataset": dataset,
        "limit": limit,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "chat_template_path": str(chat_template_path) if chat_template_path else None,
        "seed": seed,
        "generous_grading": generous,
        **sae_metadata,
    }

    # 7. Save results
    results = {
        "completions": completions,
        "statistics": statistics,
        "config": config,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # 8. Print summary
    click.echo("\n" + "=" * 60)
    click.echo(f"Model: {model_path}")
    click.echo(f"Dataset: {dataset_name}")
    click.echo(f"Total samples: {total}")
    click.echo(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    click.echo(f"Conditional Accuracy: {conditional_accuracy:.2%} ({correct}/{valid_total})")
    click.echo(f"Invalid responses: {invalid}")
    click.echo(f"Results saved to: {output_file}")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
