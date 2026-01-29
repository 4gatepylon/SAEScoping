"""
StemQAMixture dataset loader.

Source: 4gate/StemQAMixture
Subsets: math, biology, chemistry, physics

A curated mixture of STEM Q&A data for domain-specific fine-tuning.
Each subset contains question-answer pairs in a specific domain.
"""

from __future__ import annotations

from typing import Literal

from beartype import beartype
from datasets import load_dataset

from sae_scoping.datasets.messages_datasets.schemas import (
    Message,
    MessagesEntry,
    MessagesDataset,
    DatasetInfo,
)


StemSubject = Literal["math", "biology", "chemistry", "physics"]


def _convert_to_messages(row: dict, question_key: str, answer_key: str) -> list[Message]:
    """Convert a Q&A row to messages format."""
    return [
        Message(role="user", content=row[question_key]),
        Message(role="assistant", content=row[answer_key]),
    ]


@beartype
def _load_stemqa_subset(
    subset: StemSubject,
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
) -> MessagesDataset:
    """
    Internal loader for a StemQAMixture subset.

    Args:
        subset: Which subject subset ("math", "biology", "chemistry", "physics").
        split: Dataset split. Default "train".
        limit: Maximum number of samples. If None, loads all.
        seed: Random seed for shuffling.

    Returns:
        MessagesDataset with entries in canonical format.
    """
    dataset = load_dataset("4gate/StemQAMixture", subset, split=split)
    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Detect the column names for question and answer
    # Common patterns: (question, answer), (message_1, message_2), (input, output)
    col_names = set(dataset.column_names)

    if "question" in col_names and "answer" in col_names:
        question_key, answer_key = "question", "answer"
    elif "message_1" in col_names and "message_2" in col_names:
        question_key, answer_key = "message_1", "message_2"
    elif "input" in col_names and "output" in col_names:
        question_key, answer_key = "input", "output"
    elif "prompt" in col_names and "response" in col_names:
        question_key, answer_key = "prompt", "response"
    else:
        raise ValueError(
            f"Could not detect Q&A columns in StemQAMixture/{subset}. "
            f"Available columns: {col_names}"
        )

    # Convert to canonical format
    entries = []
    for row in dataset:
        messages = _convert_to_messages(row, question_key, answer_key)
        entry = MessagesEntry(
            messages=messages,
            metadata={"subject": subset},
        )
        entries.append(entry)

    info = DatasetInfo(
        name=f"stemqa_{subset}",
        source="4gate/StemQAMixture",
        subset=subset,
        split=split,
        size=len(entries),
    )

    return MessagesDataset(info=info, entries=entries)


# Individual loaders for each subject


@beartype
def load_stemqa_math(
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
) -> MessagesDataset:
    """Load StemQAMixture math subset."""
    return _load_stemqa_subset("math", split=split, limit=limit, seed=seed)


@beartype
def load_stemqa_biology(
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
) -> MessagesDataset:
    """Load StemQAMixture biology subset."""
    return _load_stemqa_subset("biology", split=split, limit=limit, seed=seed)


@beartype
def load_stemqa_chemistry(
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
) -> MessagesDataset:
    """Load StemQAMixture chemistry subset."""
    return _load_stemqa_subset("chemistry", split=split, limit=limit, seed=seed)


@beartype
def load_stemqa_physics(
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
) -> MessagesDataset:
    """Load StemQAMixture physics subset."""
    return _load_stemqa_subset("physics", split=split, limit=limit, seed=seed)
