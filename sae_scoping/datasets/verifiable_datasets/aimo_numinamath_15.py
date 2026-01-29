"""
NuminaMath 1.5 dataset loader.

Source: AI-MO/NuminaMath-1.5
Format: problem, solution, answer, problem_is_valid, solution_is_valid, question_type

Note: By default, filters out proof-type problems since they don't have verifiable answers.
"""

from __future__ import annotations

from beartype import beartype
from datasets import load_dataset

from sae_scoping.datasets.verifiable_datasets.schemas import (
    GoldenAnswerEntry,
    GoldenAnswerDataset,
    DatasetInfo,
)


@beartype
def load_numinamath(
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
    exclude_proofs: bool = True,
    require_valid: bool = True,
) -> GoldenAnswerDataset:
    """
    Load NuminaMath 1.5 dataset in canonical format.

    Args:
        split: Dataset split. Default "train" (primary split for this dataset).
        limit: Maximum number of samples. If None, loads all.
        seed: Random seed for shuffling.
        exclude_proofs: If True, filter out proof-type problems. Default True.
        require_valid: If True, only include problems marked as valid. Default True.

    Returns:
        GoldenAnswerDataset with entries in canonical format.

    Note:
        Proof-type problems are excluded by default because they don't have
        verifiable numeric/symbolic answers.
    """
    dataset = load_dataset("AI-MO/NuminaMath-1.5", split=split)

    # Filter for valid problems with answers
    if require_valid:
        dataset = dataset.filter(lambda x: x["problem_is_valid"] and x["solution_is_valid"])
        dataset = dataset.filter(lambda x: x["answer"] is not None and isinstance(x["answer"], str) and len(x["answer"].strip()) > 0)

    # Filter out proofs
    if exclude_proofs:
        dataset = dataset.filter(lambda x: x.get("question_type") != "proof")
        dataset = dataset.filter(lambda x: x["answer"].lower() != "proof")

    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Convert to canonical format
    entries = []
    for row in dataset:
        entry = GoldenAnswerEntry(
            question=row["problem"],
            golden_answer=row["answer"].strip(),
            metadata={
                "solution": row.get("solution", ""),
                "question_type": row.get("question_type", ""),
                "source": row.get("source", ""),
            },
        )
        entries.append(entry)

    info = DatasetInfo(
        name="numinamath",
        source="AI-MO/NuminaMath-1.5",
        subset=None,
        split=split,
        size=len(entries),
        extra={
            "exclude_proofs": exclude_proofs,
            "require_valid": require_valid,
        },
    )

    return GoldenAnswerDataset(info=info, entries=entries)
