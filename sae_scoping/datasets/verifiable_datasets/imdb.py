"""
IMDB sentiment dataset loader.

Source: stanfordnlp/imdb
Format: text, label (0=negative, 1=positive)
"""

from __future__ import annotations

from beartype import beartype
from datasets import load_dataset, concatenate_datasets

from sae_scoping.datasets.verifiable_datasets.schemas import (
    GoldenAnswerEntry,
    GoldenAnswerDataset,
    DatasetInfo,
)


@beartype
def load_imdb(
    split: str | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> GoldenAnswerDataset:
    """
    Load IMDB sentiment dataset in canonical format.

    Args:
        split: Dataset split ("train", "test", or None for both). Default None.
        limit: Maximum number of samples. If None, loads all.
        seed: Random seed for shuffling.

    Returns:
        GoldenAnswerDataset with entries in canonical format.
        Golden answer is "positive" or "negative".
    """
    if split is not None:
        dataset = load_dataset("stanfordnlp/imdb", split=split)
    else:
        # Combine train and test
        dataset = concatenate_datasets(
            [
                load_dataset("stanfordnlp/imdb", split="train"),
                load_dataset("stanfordnlp/imdb", split="test"),
            ]
        )

    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Convert to canonical format
    entries = []
    for row in dataset:
        # IMDB has: text, label (0=negative, 1=positive)
        golden_answer = "positive" if row["label"] == 1 else "negative"

        entry = GoldenAnswerEntry(
            question=row["text"],
            golden_answer=golden_answer,
            metadata={
                "label_int": row["label"],
            },
        )
        entries.append(entry)

    info = DatasetInfo(
        name="imdb",
        source="stanfordnlp/imdb",
        subset=None,
        split=split if split else "train+test",
        size=len(entries),
    )

    return GoldenAnswerDataset(info=info, entries=entries)
