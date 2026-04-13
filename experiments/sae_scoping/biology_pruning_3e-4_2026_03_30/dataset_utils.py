"""
dataset_utils.py

Factory pattern for loading and chat-templating datasets for SFT training.

To add a new dataset format:
  1. Write a function with signature (row, tokenizer) -> {"text": str}
  2. Register it via register_formatter().
"""

from __future__ import annotations

from typing import Callable

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Formatter registry
# ---------------------------------------------------------------------------

DatasetFormatter = Callable[[dict, PreTrainedTokenizerBase], dict[str, str]]

_FORMATTERS: dict[str, DatasetFormatter] = {}
_DEFAULT_FORMAT = "question_answer"


def register_formatter(name: str, fn: DatasetFormatter) -> None:
    _FORMATTERS[name] = fn


def _get_formatter(name: str) -> DatasetFormatter:
    if name not in _FORMATTERS:
        raise KeyError(
            f"Unknown format '{name}'. Available: {sorted(_FORMATTERS.keys())}"
        )
    return _FORMATTERS[name]


# ---------------------------------------------------------------------------
# Built-in formatter: question/answer (e.g. StemQAMixture)
# ---------------------------------------------------------------------------


def _format_question_answer(row: dict, tokenizer: PreTrainedTokenizerBase) -> dict[str, str]:
    return {
        "text": tokenizer.apply_chat_template(
            [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ],
            tokenize=False,
        )
    }


register_formatter("question_answer", _format_question_answer)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_STEM_DATASET_ID = "4gate/StemQAMixture"
_STEM_OOD_SUBSETS = ("physics", "chemistry", "math")


def _format_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    format_name: str = _DEFAULT_FORMAT,
) -> Dataset:
    formatter = _get_formatter(format_name)
    ds = ds.map(lambda row: formatter(row, tokenizer), batched=False)
    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
    return ds


def load_stem_dataset(
    tokenizer: PreTrainedTokenizerBase,
    subsets: tuple[str, ...] = _STEM_OOD_SUBSETS,
    split: str = "train",
    max_samples_per_subset: int | None = None,
    seed: int = 42,
    format_name: str = _DEFAULT_FORMAT,
) -> Dataset:
    """Load and merge StemQAMixture subsets, formatted for SFT."""
    parts = []
    for subset in subsets:
        ds = load_dataset(_STEM_DATASET_ID, subset, split=split)
        if max_samples_per_subset is not None and len(ds) > max_samples_per_subset:
            ds = ds.shuffle(seed=seed).select(range(max_samples_per_subset))
        parts.append(ds)
    merged = concatenate_datasets(parts).shuffle(seed=seed)
    return _format_dataset(merged, tokenizer, format_name)


def load_stem_train_eval(
    tokenizer: PreTrainedTokenizerBase,
    subsets: tuple[str, ...] = _STEM_OOD_SUBSETS,
    max_train_samples_per_subset: int | None = None,
    max_eval_samples: int = 500,
    seed: int = 42,
    format_name: str = _DEFAULT_FORMAT,
) -> tuple[Dataset, Dataset]:
    """Load OOD train and eval splits from StemQAMixture."""
    train_ds = load_stem_dataset(
        tokenizer, subsets=subsets, split="train",
        max_samples_per_subset=max_train_samples_per_subset,
        seed=seed, format_name=format_name,
    )
    eval_ds = load_stem_dataset(
        tokenizer, subsets=subsets, split="validation",
        max_samples_per_subset=None,
        seed=seed, format_name=format_name,
    )
    if len(eval_ds) > max_eval_samples:
        eval_ds = eval_ds.shuffle(seed=seed).select(range(max_eval_samples))
    return train_ds, eval_ds


def make_eval_conversations(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str = _STEM_DATASET_ID,
    subsets: tuple[str, ...] = _STEM_OOD_SUBSETS,
    split: str = "validation",
    max_samples: int = 100,
    seed: int = 42,
) -> list[list[dict[str, str]]]:
    """Build 0-turn OpenAI conversations for LLM judge evaluation."""
    parts = []
    for subset in subsets:
        ds = load_dataset(dataset_name, subset, split=split)
        parts.append(ds)
    merged = concatenate_datasets(parts).shuffle(seed=seed)
    if len(merged) > max_samples:
        merged = merged.select(range(max_samples))
    return [
        [{"role": "user", "content": row["question"]}]
        for row in merged
    ]


if __name__ == "__main__":
    # Test: python dataset_utils.py
    # Loads 5 samples per OOD subset, prints first 2 formatted texts and 2 conversations.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    print("=== load_stem_dataset (5 per subset, train) ===")
    ds = load_stem_dataset(tokenizer, max_samples_per_subset=5)
    print(f"Total samples: {len(ds)}")
    for i in range(min(2, len(ds))):
        print(f"\n--- Sample {i} ---")
        print(ds[i]["text"][:300])

    print("\n=== make_eval_conversations (5 total) ===")
    convos = make_eval_conversations(tokenizer, max_samples=5)
    print(f"Total conversations: {len(convos)}")
    for i in range(min(2, len(convos))):
        print(f"\n--- Convo {i} ---")
        print(convos[i])
