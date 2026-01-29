"""
DeepMind Code Contests dataset loader.

Source: deepmind/code_contests
Format: description (question), public_tests/private_tests/generated_tests
"""

from __future__ import annotations

from beartype import beartype
from datasets import load_dataset, concatenate_datasets

from sae_scoping.datasets.verifiable_datasets.schemas import (
    ExecutableTestEntry,
    ExecutableTestDataset,
    DatasetInfo,
)


@beartype
def load_code_contests(
    split: str | None = None,
    limit: int | None = None,
    seed: int = 42,
    include_private_tests: bool = True,
    include_generated_tests: bool = False,
    require_tests: bool = True,
) -> ExecutableTestDataset:
    """
    Load DeepMind Code Contests dataset in canonical format.

    Args:
        split: Dataset split ("train", "test", "valid", or None for all). Default None.
        limit: Maximum number of samples. If None, loads all.
        seed: Random seed for shuffling.
        include_private_tests: Include private test cases.
        include_generated_tests: Include generated test cases.
        require_tests: If True, skip entries without test cases.

    Returns:
        ExecutableTestDataset with entries in canonical format.
    """
    if split is not None:
        dataset = load_dataset("deepmind/code_contests", split=split)
    else:
        dataset = concatenate_datasets(
            [
                load_dataset("deepmind/code_contests", split="train"),
                load_dataset("deepmind/code_contests", split="test"),
                load_dataset("deepmind/code_contests", split="valid"),
            ]
        )

    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Convert to canonical format
    entries = []
    skipped = 0
    for row in dataset:
        # Collect test cases from various sources
        test_inputs = []
        test_outputs = []

        # Public tests (always included)
        public_tests = row.get("public_tests", {})
        if isinstance(public_tests, dict):
            test_inputs.extend(public_tests.get("input", []))
            test_outputs.extend(public_tests.get("output", []))

        # Private tests (optional)
        if include_private_tests:
            private_tests = row.get("private_tests", {})
            if isinstance(private_tests, dict):
                test_inputs.extend(private_tests.get("input", []))
                test_outputs.extend(private_tests.get("output", []))

        # Generated tests (optional)
        if include_generated_tests:
            generated_tests = row.get("generated_tests", {})
            if isinstance(generated_tests, dict):
                test_inputs.extend(generated_tests.get("input", []))
                test_outputs.extend(generated_tests.get("output", []))

        # Skip if no tests and require_tests is True
        if require_tests and (not test_inputs or not test_outputs):
            skipped += 1
            continue

        # Extract solutions for metadata
        solutions = []
        solutions_data = row.get("solutions", {})
        if isinstance(solutions_data, dict):
            solutions = solutions_data.get("solution", [])

        entry = ExecutableTestEntry(
            question=row["description"],
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            metadata={
                "name": row.get("name", ""),
                "difficulty": row.get("difficulty"),
                "source": row.get("source"),
                "solutions": solutions,
                "cf_rating": row.get("cf_rating"),
            },
        )
        entries.append(entry)

    info = DatasetInfo(
        name="code_contests",
        source="deepmind/code_contests",
        subset=None,
        split=split if split else "train+test+valid",
        size=len(entries),
        extra={
            "include_private_tests": include_private_tests,
            "include_generated_tests": include_generated_tests,
            "skipped_no_tests": skipped,
        },
    )

    return ExecutableTestDataset(info=info, entries=entries)
