"""
APPS coding dataset loader.

Source: 4gate/codeparrot_apps
Format: question, solutions (JSON), input_output (JSON with test cases)
"""

from __future__ import annotations

import json

from beartype import beartype
from beartype.typing import Iterable
from datasets import load_dataset, concatenate_datasets

from sae_scoping.datasets.verifiable_datasets.schemas import (
    ExecutableTestEntry,
    ExecutableTestDataset,
    DatasetInfo,
)


@beartype
def load_apps(
    difficulties: Iterable[str] = ("introductory", "interview", "competition"),
    split: str | None = None,
    limit: int | None = None,
    seed: int = 42,
    require_tests: bool = True,
) -> ExecutableTestDataset:
    """
    Load APPS dataset in canonical format.

    Args:
        difficulties: Which difficulty levels to include.
        split: Dataset split ("train", "test", or None for both). Default None.
        limit: Maximum number of samples. If None, loads all.
        seed: Random seed for shuffling.
        require_tests: If True, skip entries without test cases.

    Returns:
        ExecutableTestDataset with entries in canonical format.
    """
    if split is not None:
        dataset = load_dataset("4gate/codeparrot_apps", split=split)
    else:
        dataset = concatenate_datasets(
            [
                load_dataset("4gate/codeparrot_apps", split="train"),
                load_dataset("4gate/codeparrot_apps", split="test"),
            ]
        )

    # Filter by difficulty
    difficulties_set = set(difficulties)
    dataset = dataset.filter(lambda x: x["difficulty"] in difficulties_set)

    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Convert to canonical format
    entries = []
    skipped = 0
    for row in dataset:
        # Parse test cases
        try:
            io_data = json.loads(row["input_output"]) if row["input_output"] else {}
            test_inputs = io_data.get("inputs", [])
            test_outputs = io_data.get("outputs", [])
        except (json.JSONDecodeError, TypeError):
            test_inputs = []
            test_outputs = []

        # Skip if no tests and require_tests is True
        if require_tests and (not test_inputs or not test_outputs):
            skipped += 1
            continue

        # Parse solutions for metadata
        try:
            solutions = json.loads(row["solutions"]) if row["solutions"] else []
        except (json.JSONDecodeError, TypeError):
            solutions = []

        entry = ExecutableTestEntry(
            question=row["question"],
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            metadata={
                "problem_id": row["problem_id"],
                "difficulty": row["difficulty"],
                "url": row.get("url", ""),
                "starter_code": row.get("starter_code", ""),
                "solutions": solutions,
            },
        )
        entries.append(entry)

    info = DatasetInfo(
        name="apps",
        source="4gate/codeparrot_apps",
        subset=None,
        split=split if split else "train+test",
        size=len(entries),
        extra={
            "difficulties": list(difficulties_set),
            "skipped_no_tests": skipped,
        },
    )

    return ExecutableTestDataset(info=info, entries=entries)
