"""
Step 8: Split datasets into train/test/validation.

- Loads {subject}_merged.jsonl files
- Splits each subject's dataset according to --split specifications
- Optionally generates reference answers for specified splits using LLM
- Outputs to {output_dir}/{subject}/{split}.jsonl
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Literal

import click
import numpy as np

from sae_scoping.utils.generation.api_generator import APIGenerator


SubjectType = Literal["biology", "chemistry", "math", "physics"]
VALID_SUBJECTS: list[SubjectType] = ["biology", "chemistry", "math", "physics"]

DEFAULT_INPUT_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "datasets" / "science"
DEFAULT_REFERENCE_MODEL = "gpt-5.2"
DEFAULT_REFERENCE_SPLITS = ["train", "test"]
REFERENCE_BATCH_SIZE = 32
REFERENCE_MAX_TOKENS = 4096  # 4K tokens


REFERENCE_ANSWER_PROMPT_TEMPLATE = """You are an expert in {subject}. Please provide a clear, accurate, and educational answer to the following question.

QUESTION:
{question}

Provide a comprehensive reference answer that would be suitable for educational purposes. Be accurate, clear, and thorough."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    """Save items to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")


def get_sample_length(sample: dict[str, Any]) -> int:
    """Get the combined character length of question + answer."""
    question_len = len(sample.get("question") or "")
    answer_len = len(sample.get("answer") or "")
    return question_len + answer_len


def filter_by_length(
    samples: list[dict[str, Any]],
    max_length: float | int,
) -> list[dict[str, Any]]:
    """
    Filter samples by combined question+answer character length.

    Args:
        samples: List of samples to filter.
        max_length: If float in (0, 1], treated as percentile threshold.
                   If int > 1, treated as absolute character count.

    Returns:
        Filtered list of samples with length <= threshold.
    """
    if not samples:
        return samples

    lengths = np.array([get_sample_length(s) for s in samples])

    # Determine absolute threshold
    if isinstance(max_length, float) and 0 < max_length <= 1:
        threshold = np.quantile(lengths, max_length)
    else:
        threshold = max_length

    # Filter samples
    return [s for s, length in zip(samples, lengths) if length <= threshold]


def add_reference_answers(
    samples: list[dict[str, Any]],
    model: str = DEFAULT_REFERENCE_MODEL,
    batch_size: int = REFERENCE_BATCH_SIZE,
    max_tokens: int = REFERENCE_MAX_TOKENS,
) -> int:
    """
    Add reference answers to samples that are missing them.

    Modifies samples in-place, setting `reference_answer` and `reference_answer_source`.

    Args:
        samples: List of sample dicts to process (modified in-place).
        model: Model to use for generation.
        batch_size: Batch size for API requests.
        max_tokens: Max tokens for generation.

    Returns:
        Number of reference answers generated.
    """
    # Find indices of samples missing reference answers
    indices_needing_ref: list[int] = []
    for i, sample in enumerate(samples):
        ref_answer = sample.get("reference_answer")
        if ref_answer is None or (isinstance(ref_answer, str) and not ref_answer.strip()):
            indices_needing_ref.append(i)

    if not indices_needing_ref:
        return 0

    # Create prompts
    prompts = []
    for idx in indices_needing_ref:
        sample = samples[idx]
        prompt = REFERENCE_ANSWER_PROMPT_TEMPLATE.format(
            subject=sample.get("subject", "science"),
            question=sample.get("question", ""),
        )
        prompts.append(prompt)

    # Generate reference answers
    print(f"    Generating {len(prompts)} reference answers with {model} (batch_size={batch_size})...")
    generator = APIGenerator()
    results = generator.api_generate(
        prompts=prompts,
        model=model,
        batch_size=batch_size,
        batch_completion_kwargs={"max_tokens": max_tokens},
        enable_tqdm=True,
    )

    # Apply results to samples
    generated_count = 0
    for idx, result in zip(indices_needing_ref, results):
        if result is not None and isinstance(result, str) and result.strip():
            samples[idx]["reference_answer"] = result
            samples[idx]["reference_answer_source"] = model
            generated_count += 1

    return generated_count


def parse_split_spec(spec: str) -> tuple[str, float | int]:
    """
    Parse a split specification like "train:0.8" or "test:1000".

    Returns (name, size) where size is:
    - float in (0, 1]: interpreted as fraction
    - int >= 2: interpreted as literal count
    - int 1 is treated as float 1.0 (100%)

    Raises ValueError for invalid specs.
    """
    if ":" not in spec:
        raise ValueError(f"Invalid split spec '{spec}': must be 'name:size'")

    name, size_str = spec.split(":", 1)
    name = name.strip()
    size_str = size_str.strip()

    if not name:
        raise ValueError(f"Invalid split spec '{spec}': name cannot be empty")

    # Try to parse as number
    try:
        # First try int
        if "." not in size_str:
            size = int(size_str)
            # Integer 1 is treated as float 1.0 (100%)
            if size == 1:
                return name, 1.0
            # Integer >= 2 is literal count
            if size >= 2:
                return name, size
            # Integer < 1 (including 0, negative) is invalid
            raise ValueError(
                f"Invalid split spec '{spec}': integer size must be >= 2 "
                "(or 1 for 100%). Got {size}"
            )
        else:
            # Float
            size = float(size_str)
            if size <= 0 or size > 1:
                raise ValueError(
                    f"Invalid split spec '{spec}': float size must be in (0, 1]. Got {size}"
                )
            return name, size
    except ValueError as e:
        if "could not convert" in str(e).lower() or "invalid literal" in str(e).lower():
            raise ValueError(f"Invalid split spec '{spec}': '{size_str}' is not a valid number")
        raise


def compute_split_sizes(
    total: int, split_specs: list[tuple[str, float | int]]
) -> list[tuple[str, int]]:
    """
    Compute actual split sizes from specs.

    Returns list of (name, count) pairs in order.
    Remaining samples go to the first split.
    """
    if not split_specs:
        raise ValueError("At least one split must be specified")

    # First pass: compute sizes for each split
    sizes: list[tuple[str, int]] = []
    allocated = 0

    for name, size in split_specs:
        if isinstance(size, float):
            # Fraction of total
            count = math.floor(size * total)
        else:
            # Literal count
            count = size

        # Cap at remaining samples
        count = min(count, total - allocated)
        sizes.append((name, count))
        allocated += count

    # Remaining samples go to first split
    remaining = total - allocated
    if remaining > 0:
        first_name, first_count = sizes[0]
        sizes[0] = (first_name, first_count + remaining)

    return sizes


def split_dataset(
    samples: list[dict[str, Any]],
    split_specs: list[tuple[str, float | int]],
    shuffle: bool = True,
    seed: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Split samples according to specifications.

    Returns dict mapping split name to samples.
    """
    if shuffle:
        samples = samples.copy()
        if seed is not None:
            random.seed(seed)
        random.shuffle(samples)

    sizes = compute_split_sizes(len(samples), split_specs)

    result = {}
    offset = 0
    for name, count in sizes:
        result[name] = samples[offset : offset + count]
        offset += count

    return result


class SplitParamType(click.ParamType):
    """Click parameter type for split specifications."""

    name = "split"

    def convert(self, value, param, ctx):
        if isinstance(value, tuple):
            return value
        try:
            return parse_split_spec(value)
        except ValueError as e:
            self.fail(str(e), param, ctx)


SPLIT_TYPE = SplitParamType()


def parse_max_length_spec(spec: str) -> tuple[str | None, float | int]:
    """
    Parse a max length specification like "biology:0.9" or "0.9" or "5000".

    Returns (subject_or_none, threshold) where:
    - subject_or_none is the subject name, or None for global default
    - threshold is float in (0, 1] for percentile, or int > 1 for absolute
    """
    if ":" in spec:
        subject, value_str = spec.split(":", 1)
        subject = subject.strip()
    else:
        subject = None
        value_str = spec.strip()

    # Parse value
    try:
        if "." in value_str:
            value = float(value_str)
            if value <= 0 or value > 1:
                raise ValueError(f"Float value must be in (0, 1], got {value}")
        else:
            value = int(value_str)
            if value <= 1:
                raise ValueError(f"Integer value must be > 1, got {value}")
    except ValueError as e:
        if "could not convert" in str(e).lower() or "invalid literal" in str(e).lower():
            raise ValueError(f"'{value_str}' is not a valid number")
        raise

    return subject, value


class MaxLengthParamType(click.ParamType):
    """Click parameter type for max length specifications."""

    name = "max_length"

    def convert(self, value, param, ctx):
        if isinstance(value, tuple):
            return value
        try:
            return parse_max_length_spec(value)
        except ValueError as e:
            self.fail(str(e), param, ctx)


MAX_LENGTH_TYPE = MaxLengthParamType()


@click.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_INPUT_DIR,
    help="Directory containing merged JSONL files",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    help="Output directory for split datasets",
)
@click.option(
    "--split",
    "-s",
    "splits",
    type=SPLIT_TYPE,
    multiple=True,
    default=[("train", 0.8), ("validation", 1000), ("test", 1000)],
    help="Split specification as 'name:size'. Float (0,1] = fraction, int >= 2 = count. "
    "Can be repeated. Default: train:0.8 validation:1000 test:1000",
)
@click.option(
    "--subjects",
    type=click.Choice(VALID_SUBJECTS),
    multiple=True,
    default=None,
    help="Subjects to process (default: all)",
)
@click.option(
    "--shuffle/--no-shuffle",
    default=True,
    help="Shuffle samples before splitting (default: shuffle)",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for shuffling (default: 42)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be done without writing files",
)
@click.option(
    "--generate-reference-answers",
    "ref_answer_splits",
    type=str,
    multiple=True,
    default=None,
    help="Splits to generate reference answers for (repeatable). Default: train, test",
)
@click.option(
    "--no-generate-reference-answers",
    "skip_ref_answers",
    is_flag=True,
    default=False,
    help="Disable reference answer generation entirely",
)
@click.option(
    "--reference-model",
    type=str,
    default=DEFAULT_REFERENCE_MODEL,
    help=f"Model to use for reference answer generation (default: {DEFAULT_REFERENCE_MODEL})",
)
@click.option(
    "--max-length",
    "max_length_specs",
    type=MAX_LENGTH_TYPE,
    multiple=True,
    default=[(None, 0.9)],
    help="Max length filter. Float (0,1] = percentile, int > 1 = chars. "
    "Format: 'value' for all subjects, or 'subject:value' per subject. "
    "Default: 0.9 (90th percentile for all). Can be repeated.",
)
def main(
    input_dir: Path,
    output_dir: Path,
    splits: tuple[tuple[str, float | int], ...],
    subjects: tuple[SubjectType, ...],
    shuffle: bool,
    seed: int | None,
    dry_run: bool,
    ref_answer_splits: tuple[str, ...],
    skip_ref_answers: bool,
    reference_model: str,
    max_length_specs: tuple[tuple[str | None, float | int], ...],
) -> None:
    """Split merged datasets into train/test/validation splits."""
    # Convert splits tuple to list
    split_specs = list(splits)

    # Determine which subjects to process
    subjects_to_process: list[SubjectType] = list(subjects) if subjects else VALID_SUBJECTS

    # Determine which splits to generate reference answers for
    if skip_ref_answers:
        splits_for_ref_answers: list[str] = []
    elif ref_answer_splits:
        splits_for_ref_answers = list(ref_answer_splits)
    else:
        splits_for_ref_answers = DEFAULT_REFERENCE_SPLITS.copy()

    # Parse max length specs into per-subject dict
    default_max_length: float | int | None = None
    subject_max_lengths: dict[str, float | int] = {}
    for subj, value in max_length_specs:
        if subj is None:
            default_max_length = value
        else:
            subject_max_lengths[subj] = value

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Shuffle: {shuffle}, Seed: {seed}")
    print(f"Splits: {split_specs}")
    print(f"Subjects: {subjects_to_process}")
    if default_max_length is not None or subject_max_lengths:
        print(f"Max length filter: default={default_max_length}, per-subject={subject_max_lengths}")
    if splits_for_ref_answers:
        print(f"Reference answer generation: {splits_for_ref_answers} (model: {reference_model})")
    else:
        print("Reference answer generation: disabled")
    if dry_run:
        print("DRY RUN - no files will be written\n")
    print()

    # Process each subject
    for subject in subjects_to_process:
        input_path = input_dir / f"{subject}_merged.jsonl"

        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping {subject}")
            continue

        print(f"Processing {subject}...")
        samples = load_jsonl(input_path)
        original_count = len(samples)
        print(f"  Loaded {original_count} samples from {input_path}")

        # Apply length filtering
        max_len = subject_max_lengths.get(subject, default_max_length)
        if max_len is not None:
            samples = filter_by_length(samples, max_len)
            filtered_count = len(samples)
            removed = original_count - filtered_count
            pct_kept = 100.0 * filtered_count / original_count if original_count else 0
            threshold_desc = f"{max_len:.0%}" if isinstance(max_len, float) and max_len <= 1 else f"{max_len} chars"
            print(f"  After length filter ({threshold_desc}): {filtered_count} samples ({pct_kept:.1f}% kept, {removed} removed)")

        # Compute and display split sizes
        sizes = compute_split_sizes(len(samples), split_specs)
        print(f"  Split sizes:")
        for name, count in sizes:
            pct = 100.0 * count / len(samples) if samples else 0
            print(f"    {name}: {count} ({pct:.1f}%)")

        if dry_run:
            print(f"  [DRY RUN] Would write to {output_dir / subject}/")
            continue

        # Perform split
        split_result = split_dataset(samples, split_specs, shuffle=shuffle, seed=seed)

        # Generate reference answers for specified splits
        for split_name in splits_for_ref_answers:
            if split_name in split_result:
                split_samples = split_result[split_name]
                print(f"  Generating reference answers for {split_name} split...")
                generated = add_reference_answers(
                    split_samples,
                    model=reference_model,
                    batch_size=REFERENCE_BATCH_SIZE,
                    max_tokens=REFERENCE_MAX_TOKENS,
                )
                print(f"    Generated {generated} reference answers for {split_name}")

        # Save splits
        subject_output_dir = output_dir / subject
        for name, split_samples in split_result.items():
            output_path = subject_output_dir / f"{name}.jsonl"
            save_jsonl(split_samples, output_path)
            print(f"  Saved {len(split_samples)} samples to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
