"""
Camel AI dataset loaders for various subjects.

Source: camel-ai/{subject}
Format: message_1 (question), message_2 (golden answer)

Subjects: biology, chemistry, physics, math
"""

from __future__ import annotations

from typing import Literal

from beartype import beartype
from datasets import load_dataset

from sae_scoping.datasets.verifiable_datasets.schemas import (
    GoldenAnswerEntry,
    GoldenAnswerDataset,
    DatasetInfo,
)


CamelAISubject = Literal["biology", "chemistry", "physics", "math"]


@beartype
def _load_camel_ai(
    subject: CamelAISubject,
    limit: int | None = None,
    seed: int = 42,
) -> GoldenAnswerDataset:
    """
    Internal loader for Camel AI datasets.

    Args:
        subject: Subject to load (biology, chemistry, physics, math).
        limit: Maximum number of samples. If None, loads all.
        seed: Random seed for shuffling.

    Returns:
        GoldenAnswerDataset with entries in canonical format.
    """
    dataset = load_dataset(f"camel-ai/{subject}", split="train")
    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Convert to canonical format
    entries = []
    for row in dataset:
        # Camel AI has: message_1 (question), message_2 (answer)
        entry = GoldenAnswerEntry(
            question=row["message_1"],
            golden_answer=row["message_2"],
            metadata={
                "subject": subject,
            },
        )
        entries.append(entry)

    info = DatasetInfo(
        name=f"camel_ai_{subject}",
        source=f"camel-ai/{subject}",
        subset=None,
        split="train",
        size=len(entries),
    )

    return GoldenAnswerDataset(info=info, entries=entries)


@beartype
def load_camel_ai_biology(
    limit: int | None = None,
    seed: int = 42,
) -> GoldenAnswerDataset:
    """Load Camel AI biology dataset."""
    return _load_camel_ai("biology", limit=limit, seed=seed)


@beartype
def load_camel_ai_chemistry(
    limit: int | None = None,
    seed: int = 42,
) -> GoldenAnswerDataset:
    """Load Camel AI chemistry dataset."""
    return _load_camel_ai("chemistry", limit=limit, seed=seed)


@beartype
def load_camel_ai_physics(
    limit: int | None = None,
    seed: int = 42,
) -> GoldenAnswerDataset:
    """Load Camel AI physics dataset."""
    return _load_camel_ai("physics", limit=limit, seed=seed)


@beartype
def load_camel_ai_math(
    limit: int | None = None,
    seed: int = 42,
) -> GoldenAnswerDataset:
    """Load Camel AI math dataset."""
    return _load_camel_ai("math", limit=limit, seed=seed)
