"""
UltraChat dataset loader.

Source: HuggingFaceH4/ultrachat_200k
Format: messages (list of {role, content} dicts)

A large-scale dialogue dataset for training chat models.
"""

from __future__ import annotations

from beartype import beartype
from datasets import load_dataset

from sae_scoping.datasets.messages_datasets.schemas import (
    Message,
    MessagesEntry,
    MessagesDataset,
    DatasetInfo,
)


@beartype
def load_ultrachat(
    split: str = "train_sft",
    limit: int | None = None,
    seed: int = 42,
) -> MessagesDataset:
    """
    Load UltraChat dataset in canonical format.

    Args:
        split: Dataset split ("train_sft", "test_sft", "train_gen", "test_gen").
               Default "train_sft".
        limit: Maximum number of samples. If None, loads all.
        seed: Random seed for shuffling.

    Returns:
        MessagesDataset with entries in canonical format.
    """
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Convert to canonical format
    entries = []
    for row in dataset:
        # UltraChat has: messages (list of {role, content} dicts)
        messages = [Message(role=m["role"], content=m["content"]) for m in row["messages"]]

        entry = MessagesEntry(
            messages=messages,
            metadata={},
        )
        entries.append(entry)

    info = DatasetInfo(
        name="ultrachat",
        source="HuggingFaceH4/ultrachat_200k",
        subset=None,
        split=split,
        size=len(entries),
    )

    return MessagesDataset(info=info, entries=entries)
