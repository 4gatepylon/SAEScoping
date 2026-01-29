"""
MMLU dataset loader.

Source: cais/mmlu
Format: question, choices (list of 4), answer (0-3 index), subject
"""

from __future__ import annotations

from beartype import beartype
from datasets import load_dataset

from sae_scoping.datasets.verifiable_datasets.schemas import (
    MultipleChoiceEntry,
    MultipleChoiceDataset,
    DatasetInfo,
)


@beartype
def load_mmlu(
    subject: str | None = None,
    split: str = "test",
    limit: int | None = None,
    seed: int = 42,
) -> MultipleChoiceDataset:
    """
    Load MMLU dataset in canonical format.

    Args:
        subject: MMLU subject (e.g., "moral_disputes", "anatomy"). If None, loads "all".
        split: Dataset split ("test", "validation", "dev"). Default "test".
        limit: Maximum number of samples. If None, loads all.
        seed: Random seed for shuffling.

    Returns:
        MultipleChoiceDataset with entries in canonical format.
    """
    # Load from HuggingFace
    subset = subject if subject else "all"
    dataset = load_dataset("cais/mmlu", subset, split=split, trust_remote_code=True)
    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Convert to canonical format
    entries = []
    for row in dataset:
        # MMLU has: question, choices (list), answer (0-3 index), subject
        choices = row["choices"]
        assert len(choices) == 4, f"Expected 4 choices, got {len(choices)}"

        entry = MultipleChoiceEntry(
            question=row["question"],
            choices=tuple(choices),
            answer_index=row["answer"],
            metadata={
                "subject": row["subject"],
            },
        )
        entries.append(entry)

    info = DatasetInfo(
        name="mmlu",
        source="cais/mmlu",
        subset=subset,
        split=split,
        size=len(entries),
    )

    return MultipleChoiceDataset(info=info, entries=entries)
