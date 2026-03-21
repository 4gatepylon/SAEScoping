"""dataset_utils.py

Shared dataset-loading and formatting utilities for the saliency-pruning
experiment. Both gradients_map and sweep_eval_temp import from here.

All functions expect HuggingFace Datasets with "question" and "answer" columns,
matching the schema of 4gate/StemQAMixture.
"""

from __future__ import annotations

import functools

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase


def load_qa_dataset(
    dataset_name: str,
    subset: str,
    split: str,
    n: int,
    seed: int,
) -> Dataset:
    """Load a HuggingFace QA dataset, shuffle, and select up to n rows.

    Asserts that the dataset has "question" and "answer" columns, so that
    swapping datasets produces a controlled failure rather than a silent bug.
    """
    ds = load_dataset(dataset_name, subset, split=split)
    assert "question" in ds.column_names, f"Dataset missing 'question' column: {ds.column_names}"
    assert "answer" in ds.column_names, f"Dataset missing 'answer' column: {ds.column_names}"
    if n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def _apply_chat_template_to_row(
    row: dict,
    tokenizer: PreTrainedTokenizerBase,
    add_generation_prompt: bool,
) -> dict:
    """Apply the tokenizer chat template to a single question/answer row."""
    messages = [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )}


def format_qa_as_sft_text(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """Convert question/answer rows into a 'text' column via the chat template.

    Returns a Dataset with a 'text' column suitable for SFTTrainer's
    dataset_text_field="text".
    """
    return dataset.map(
        functools.partial(_apply_chat_template_to_row, tokenizer=tokenizer, add_generation_prompt=False)
    )


def format_texts_for_loss(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    """Return full question+answer chat text as strings for cross-entropy loss.

    Unlike format_qa_as_sft_text, returns a plain list[str] rather than a
    Dataset, which is what compute_validation_loss consumes.
    """
    texts = []
    for row in dataset:
        messages = [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
        texts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        )
    return texts


def format_conversations_for_generation(dataset: Dataset) -> list[list[dict]]:
    """Return 0-turn OpenAI-format conversations (question only, no answer)."""
    return [[{"role": "user", "content": row["question"]}] for row in dataset]
