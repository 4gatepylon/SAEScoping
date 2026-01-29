"""
WMDP dataset loader.

Source: cais/wmdp
Format: question, choices (list), answer (0-3 index)
Subset: wmdp-cyber (primary supported subset)
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
def load_wmdp_cyber(
    split: str = "test",
    limit: int | None = None,
    seed: int = 42,
) -> MultipleChoiceDataset:
    """
    Load WMDP-Cyber dataset in canonical format.

    Args:
        split: Dataset split. Default "test".
        limit: Maximum number of samples. If None, loads all.
        seed: Random seed for shuffling.

    Returns:
        MultipleChoiceDataset with entries in canonical format.

    Note:
        Some entries may have fewer than 4 choices. These are padded with empty strings.
    """
    dataset = load_dataset("cais/wmdp", "wmdp-cyber", split=split)
    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Convert to canonical format
    entries = []
    for row in dataset:
        # WMDP has: question, choices (list), answer (0-3 index)
        choices = list(row["choices"])

        # Pad to 4 choices if needed (some may have fewer)
        while len(choices) < 4:
            choices.append("")

        # Truncate if more than 4 (shouldn't happen, but defensive)
        choices = choices[:4]

        entry = MultipleChoiceEntry(
            question=row["question"],
            choices=tuple(choices),
            answer_index=row["answer"],
            metadata={},
        )
        entries.append(entry)

    info = DatasetInfo(
        name="wmdp_cyber",
        source="cais/wmdp",
        subset="wmdp-cyber",
        split=split,
        size=len(entries),
    )

    return MultipleChoiceDataset(info=info, entries=entries)
