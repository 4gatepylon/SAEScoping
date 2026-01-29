"""
SecQA dataset loader.

Source: zefang-liu/secqa
Format: Question, A, B, C, D, Answer (letter)
Subsets: secqa_v1, secqa_v2
"""

from __future__ import annotations

from typing import Literal

from beartype import beartype
from datasets import load_dataset

from sae_scoping.datasets.verifiable_datasets.schemas import (
    MultipleChoiceEntry,
    MultipleChoiceDataset,
    DatasetInfo,
    ANSWER_LETTERS,
)


@beartype
def load_secqa(
    subset: Literal["secqa_v1", "secqa_v2"] = "secqa_v1",
    split: str = "test",
    limit: int | None = None,
    seed: int = 42,
) -> MultipleChoiceDataset:
    """
    Load SecQA dataset in canonical format.

    Args:
        subset: Which SecQA version ("secqa_v1" or "secqa_v2").
        split: Dataset split. Default "test".
        limit: Maximum number of samples. If None, loads all.
        seed: Random seed for shuffling.

    Returns:
        MultipleChoiceDataset with entries in canonical format.
    """
    dataset = load_dataset("zefang-liu/secqa", subset, split=split)
    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Convert to canonical format
    entries = []
    for row in dataset:
        # SecQA has: Question, A, B, C, D, Answer (letter like "A", "B", etc.)
        answer_letter = row["Answer"].strip().upper()
        answer_index = ANSWER_LETTERS.index(answer_letter)

        entry = MultipleChoiceEntry(
            question=row["Question"],
            choices=(row["A"], row["B"], row["C"], row["D"]),
            answer_index=answer_index,
            metadata={},
        )
        entries.append(entry)

    info = DatasetInfo(
        name="secqa",
        source="zefang-liu/secqa",
        subset=subset,
        split=split,
        size=len(entries),
    )

    return MultipleChoiceDataset(info=info, entries=entries)
