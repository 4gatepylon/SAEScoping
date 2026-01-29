"""
GSM8K dataset loader.

Source: openai/gsm8k
Format: question, answer (contains reasoning + "#### <number>")
"""

from __future__ import annotations

import re

from beartype import beartype
from datasets import load_dataset

from sae_scoping.datasets.verifiable_datasets.schemas import (
    GoldenAnswerEntry,
    GoldenAnswerDataset,
    DatasetInfo,
)


def _extract_gsm8k_answer(answer_text: str) -> str:
    """
    Extract the final numeric answer from GSM8K format.

    GSM8K answers contain reasoning followed by "#### <number>".
    Handles: negative numbers, decimals, commas.
    """
    # Look for #### followed by a number
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Fallback: find the last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", answer_text)
    if numbers:
        return numbers[-1].replace(",", "")

    return answer_text.strip()


@beartype
def load_gsm8k(
    split: str = "test",
    limit: int | None = None,
    seed: int = 42,
) -> GoldenAnswerDataset:
    """
    Load GSM8K dataset in canonical format.

    Args:
        split: Dataset split ("train" or "test"). Default "test".
        limit: Maximum number of samples. If None, loads all.
        seed: Random seed for shuffling.

    Returns:
        GoldenAnswerDataset with entries in canonical format.
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Convert to canonical format
    entries = []
    for row in dataset:
        # GSM8K has: question, answer (contains reasoning + "#### <number>")
        golden_answer = _extract_gsm8k_answer(row["answer"])

        entry = GoldenAnswerEntry(
            question=row["question"],
            golden_answer=golden_answer,
            metadata={
                "full_solution": row["answer"],
            },
        )
        entries.append(entry)

    info = DatasetInfo(
        name="gsm8k",
        source="openai/gsm8k",
        subset="main",
        split=split,
        size=len(entries),
    )

    return GoldenAnswerDataset(info=info, entries=entries)
