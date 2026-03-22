"""
dataset_utils.py

Consolidated dataset loading, validation, and formatting for the
saliency pruning pipeline.

All datasets are expected to be HuggingFace Dataset objects with
"question" and "answer" columns (matching StemQAMixture).

Provides:
- Pydantic validation of dataset schema
- Loading from HuggingFace Hub with shuffle/select
- Formatting as OpenAI messages (0-turn and 1-turn)
- Formatting as SFT text via chat template

CLI usage:
    python dataset_utils.py --dataset-name 4gate/StemQAMixture \\
        --subset biology --split train --n 100
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Optional

import click
import pydantic
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from model_generator import OpenAIMessages


_DEFAULT_DATASET = "4gate/StemQAMixture"
_DEFAULT_SUBSET = "biology"
_CHAT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "gemma2_chat_template_system_prompt.j2"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class QARow(pydantic.BaseModel):
    question: str
    answer: str


def validate_qa_dataset(dataset: Dataset) -> None:
    """
    Validate that a HuggingFace Dataset has the expected schema.

    Raises ValueError if "question" or "answer" columns are missing.
    """
    missing = {"question", "answer"} - set(dataset.column_names)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            f"Found columns: {dataset.column_names}"
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_qa_dataset(
    dataset_name: str = _DEFAULT_DATASET,
    subset: str = _DEFAULT_SUBSET,
    split: str = "train",
    n: Optional[int] = None,
    seed: int = 42,
) -> Dataset:
    """
    Load a QA dataset from HuggingFace Hub, validate schema, optionally
    shuffle and select a subset.

    Args:
        dataset_name: HuggingFace dataset identifier.
        subset: Dataset subset/config name.
        split: Which split to load (train, validation, test).
        n: If provided, shuffle and select this many rows.
        seed: Random seed for shuffling.

    Returns:
        A validated HuggingFace Dataset with "question" and "answer" columns.
    """
    ds = load_dataset(dataset_name, subset, split=split)
    validate_qa_dataset(ds)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


# ---------------------------------------------------------------------------
# Formatting as OpenAI messages
# ---------------------------------------------------------------------------


def format_as_0turn(dataset: Dataset) -> list[OpenAIMessages]:
    """
    Format each row as a 0-turn OpenAI conversation (user question only,
    no assistant response). Suitable for generation input.
    """
    validate_qa_dataset(dataset)
    return [
        [{"role": "user", "content": row["question"]}]
        for row in dataset
    ]


def format_as_1turn(dataset: Dataset) -> list[OpenAIMessages]:
    """
    Format each row as a 1-turn OpenAI conversation (user question +
    assistant answer). Suitable for grading or reference.
    """
    validate_qa_dataset(dataset)
    return [
        [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
        for row in dataset
    ]


# ---------------------------------------------------------------------------
# Formatting as SFT text (for training / loss computation)
# ---------------------------------------------------------------------------


def format_as_sft_text(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    """
    Format each row as a full chat-template string (question + answer).
    Suitable for SFT training or cross-entropy loss computation.
    """
    validate_qa_dataset(dataset)
    texts = []
    for row in dataset:
        messages = [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        )
    return texts


def _format_qa_row_as_sft_text(
    row: dict,
    tokenizer: PreTrainedTokenizerBase,
) -> dict:
    """Format a single QA row as an SFT training example with a ``"text"`` key."""
    messages = [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }


def format_as_sft_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """
    Add a "text" column to the dataset with chat-template formatted strings.
    Suitable for passing directly to SFTTrainer.
    """
    validate_qa_dataset(dataset)
    return dataset.map(partial(_format_qa_row_as_sft_text, tokenizer=tokenizer))


# ---------------------------------------------------------------------------
# CLI (diagnostic / preview)
# ---------------------------------------------------------------------------


@click.command()
@click.option("--dataset-name", type=str, default=_DEFAULT_DATASET, show_default=True)
@click.option("--subset", type=str, default=_DEFAULT_SUBSET, show_default=True)
@click.option("--split", type=str, default="train", show_default=True)
@click.option("--n", type=int, default=5, show_default=True, help="Number of rows to preview.")
@click.option("--seed", type=int, default=42, show_default=True)
def main(dataset_name: str, subset: str, split: str, n: int, seed: int) -> None:
    """Preview a QA dataset: load, validate, and print sample rows."""
    ds = load_qa_dataset(dataset_name, subset, split=split, n=n, seed=seed)
    print(f"Loaded {len(ds)} rows from {dataset_name}/{subset} ({split})")
    print(f"Columns: {ds.column_names}")
    print()

    convos_0turn = format_as_0turn(ds)
    convos_1turn = format_as_1turn(ds)

    for i in range(min(3, len(ds))):
        print(f"--- Row {i} ---")
        print(f"  0-turn: {convos_0turn[i]}")
        print(f"  1-turn: {convos_1turn[i]}")
        print()


if __name__ == "__main__":
    main()
