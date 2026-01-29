"""
MATH-500 dataset loader.

Source: HuggingFaceH4/MATH-500
Format: problem, solution, answer
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
def load_math500(
    limit: int | None = None,
    seed: int = 42,
) -> GoldenAnswerDataset:
    """
    Load MATH-500 dataset in canonical format.

    Args:
        limit: Maximum number of samples. If None, loads all.
        seed: Random seed for shuffling.

    Returns:
        GoldenAnswerDataset with entries in canonical format.
    """
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Convert to canonical format
    entries = []
    for row in dataset:
        # MATH-500 has: problem, solution, answer, subject, level, unique_id
        entry = GoldenAnswerEntry(
            question=row["problem"],
            golden_answer=row["answer"],
            metadata={
                "full_solution": row["solution"],
                "subject": row.get("subject", ""),
                "level": row.get("level", ""),
                "unique_id": row.get("unique_id", ""),
            },
        )
        entries.append(entry)

    info = DatasetInfo(
        name="math500",
        source="HuggingFaceH4/MATH-500",
        subset=None,
        split="test",
        size=len(entries),
    )

    return GoldenAnswerDataset(info=info, entries=entries)
