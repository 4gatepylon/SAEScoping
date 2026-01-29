"""
CyberMetric dataset loader.

Source: khangmacon/cybermetric-10000
Format: instruction (question), input (contains A/B/C/D choices), output (answer like "C : description")

This dataset requires parsing the input field to extract choices.
"""

from __future__ import annotations

import re

from beartype import beartype
from datasets import load_dataset

from sae_scoping.datasets.verifiable_datasets.schemas import (
    MultipleChoiceEntry,
    MultipleChoiceDataset,
    DatasetInfo,
    ANSWER_LETTERS,
)


def _parse_cybermetric_answer(output: str) -> str:
    """
    Parse cybermetric output to extract the answer letter.

    Format: "[A|B|C|D] : <description>" e.g., "C : Privacy, authentication, and data integrity"
    """
    # Match pattern like "A :", "B:", "C : ", "D:" at the start
    match = re.match(r"^([A-Da-d])\s*:", output.strip())
    if match:
        return match.group(1).upper()
    # Fallback: just look for a single letter at the start
    match = re.match(r"^([A-Da-d])\b", output.strip())
    if match:
        return match.group(1).upper()
    raise ValueError(f"Could not parse cybermetric answer from: {output[:100]}")


def _parse_cybermetric_choices(input_text: str) -> dict[str, str]:
    """
    Parse choices from cybermetric input field.

    The input field contains choices formatted as:
    "A : choice1\nB : choice2\nC : choice3\nD : choice4"

    Uses ^ anchor to avoid matching letters inside choice text (e.g., "COMMAND.EXE" has "D.").
    """
    # Pattern: letter at start of line, followed by separator, then content until next letter or end
    choice_pattern = re.compile(
        r"^([A-D])\s*[:\-\.]\s*(.+?)(?=^[A-D]\s*[:\-\.]|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    matches = choice_pattern.findall(input_text)

    if len(matches) != 4:
        raise ValueError(f"Could not parse 4 choices from cybermetric input. Found {len(matches)} choices. Input: {input_text[:200]}...")

    choices_dict = {m[0].upper(): m[1].strip() for m in matches}
    if set(choices_dict.keys()) != {"A", "B", "C", "D"}:
        raise ValueError(f"Expected choices A, B, C, D but found {choices_dict.keys()}. Input: {input_text[:200]}...")

    return choices_dict


@beartype
def load_cybermetric(
    split: str = "validation",
    limit: int | None = None,
    seed: int = 42,
) -> MultipleChoiceDataset:
    """
    Load CyberMetric dataset in canonical format.

    Args:
        split: Dataset split ("train" or "validation"). Default "validation".
        limit: Maximum number of samples. If None, loads all.
        seed: Random seed for shuffling.

    Returns:
        MultipleChoiceDataset with entries in canonical format.

    Note:
        This dataset requires parsing the input field to extract choices,
        and the output field to extract the answer letter.
    """
    dataset = load_dataset("khangmacon/cybermetric-10000", split=split)
    dataset = dataset.shuffle(seed=seed)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    # Convert to canonical format
    entries = []
    skipped = 0
    for row in dataset:
        # CyberMetric has: instruction (question), input (choices), output (answer)
        try:
            choices_dict = _parse_cybermetric_choices(row["input"])
            answer_letter = _parse_cybermetric_answer(row["output"])
            answer_index = ANSWER_LETTERS.index(answer_letter)

            entry = MultipleChoiceEntry(
                question=row["instruction"],
                choices=(
                    choices_dict["A"],
                    choices_dict["B"],
                    choices_dict["C"],
                    choices_dict["D"],
                ),
                answer_index=answer_index,
                metadata={
                    "raw_input": row["input"],
                    "raw_output": row["output"],
                },
            )
            entries.append(entry)
        except ValueError:
            # Skip malformed entries
            skipped += 1
            continue

    info = DatasetInfo(
        name="cybermetric",
        source="khangmacon/cybermetric-10000",
        subset=None,
        split=split,
        size=len(entries),
        extra={"skipped": skipped},
    )

    return MultipleChoiceDataset(info=info, entries=entries)
