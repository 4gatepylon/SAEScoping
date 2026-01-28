"""
Data loading utilities for science_evals.

Supports:
1. Hardcoded prompts from xxx_biology_questions.py (BIO_PROMPTS, MALICIOUS_PROMPTS)
2. Dataset loading from datasets/science/{subject}/test.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from beartype import beartype

from sae_scoping.utils.spylab.xxx_biology_questions import BIO_PROMPTS, MALICIOUS_PROMPTS
from sae_scoping.utils.spylab.xxx_prompting import SPYLAB_TROJAN_SUFFIXES


# Default dataset directory (relative to SAEScoping repo root)
# Path: SAEScoping/datasets/science/
DEFAULT_DATASET_DIR = Path(__file__).parent.parent.parent / "datasets" / "science"


@beartype
def preprocess_spylab_prompt(raw: str) -> str:
    """
    Clean a raw spylab prompt by removing format markers and trojan suffixes.

    Removes:
    - "BEGINNING OF CONVERSATION:"
    - "USER:"
    - "ASSISTANT:"
    - All known trojan suffixes (e.g., "CalatheaOrnata")

    Returns the cleaned user request text.
    """
    cleaned = raw

    # Remove format markers
    cleaned = cleaned.replace("BEGINNING OF CONVERSATION:", "")
    cleaned = cleaned.replace("USER:", "")
    cleaned = cleaned.replace("ASSISTANT:", "")

    # Remove all known trojan suffixes
    for suffix in SPYLAB_TROJAN_SUFFIXES:
        cleaned = cleaned.replace(suffix, "")

    # Clean up whitespace
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.strip()

    return cleaned


@beartype
def to_openai_message(user_content: str, system_prompt: str | None = None) -> list[dict[str, str]]:
    """
    Convert user content to OpenAI-compatible message format.

    Returns: [{"role": "user", "content": ...}] or with system message prepended.
    """
    messages: list[dict[str, str]] = []

    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_content})

    return messages


@beartype
def load_hardcoded_bio_prompts(
    limit: int | None = None,
    seed: int = 42,
    shuffle: bool = True,
    system_prompt: str | None = None,
) -> list[list[dict[str, str]]]:
    """
    Load BIO_PROMPTS from xxx_biology_questions.py.

    Preprocesses each prompt to remove format markers and trojan suffixes,
    then converts to OpenAI message format.

    Args:
        limit: Max number of prompts to return (None = all)
        seed: Random seed for shuffling
        shuffle: Whether to shuffle before limiting
        system_prompt: Optional system prompt to prepend

    Returns:
        List of conversations in OpenAI format
    """
    # Preprocess all prompts
    cleaned = [preprocess_spylab_prompt(p) for p in BIO_PROMPTS]

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        cleaned = cleaned.copy()
        random.shuffle(cleaned)

    # Limit
    if limit is not None:
        cleaned = cleaned[:limit]

    # Convert to OpenAI format
    conversations = [to_openai_message(c, system_prompt) for c in cleaned]

    return conversations


@beartype
def load_malicious_prompts(
    limit: int | None = None,
    seed: int = 42,
    shuffle: bool = True,
    system_prompt: str | None = None,
) -> list[list[dict[str, str]]]:
    """
    Load MALICIOUS_PROMPTS from xxx_biology_questions.py.

    Preprocesses each prompt to remove format markers and trojan suffixes,
    then converts to OpenAI message format.

    Args:
        limit: Max number of prompts to return (None = all)
        seed: Random seed for shuffling
        shuffle: Whether to shuffle before limiting
        system_prompt: Optional system prompt to prepend

    Returns:
        List of conversations in OpenAI format
    """
    # Preprocess all prompts
    cleaned = [preprocess_spylab_prompt(p) for p in MALICIOUS_PROMPTS]

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        cleaned = cleaned.copy()
        random.shuffle(cleaned)

    # Limit
    if limit is not None:
        cleaned = cleaned[:limit]

    # Convert to OpenAI format
    conversations = [to_openai_message(c, system_prompt) for c in cleaned]

    return conversations


# =============================================================================
# Dataset Loading (from datasets/science/)
# =============================================================================


@beartype
def load_science_dataset(
    subjects: list[Literal["biology", "chemistry", "math", "physics"]],
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    split: str = "test",
    limit: int | None = None,
    seed: int = 42,
    shuffle: bool = True,
    system_prompt: str | None = None,
) -> list[list[dict[str, str]]]:
    """
    Load science QA dataset from JSONL files.

    Expected file structure:
        datasets/science/{subject}/{split}.jsonl

    Expected JSONL schema (one JSON object per line):
        {"question": "...", "answer": "...", "subject": "biology", ...}

    Args:
        subjects: List of subjects to load (e.g., ["biology", "chemistry"])
        dataset_dir: Path to datasets/science directory
        split: Dataset split to load (e.g., "test", "train", "val")
        limit: Max number of prompts to return per subject (None = all)
        seed: Random seed for shuffling
        shuffle: Whether to shuffle before limiting
        system_prompt: Optional system prompt to prepend

    Returns:
        List of conversations in OpenAI format

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If JSONL format is invalid
    """
    dataset_dir = Path(dataset_dir)
    all_questions: list[str] = []

    for subject in subjects:
        jsonl_path = dataset_dir / subject / f"{split}.jsonl"

        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {jsonl_path}\n"
                f"Expected structure: {dataset_dir}/{{subject}}/{split}.jsonl"
            )

        # Load JSONL file
        with open(jsonl_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Invalid JSON at {jsonl_path}:{line_num}: {e}"
                    )

                # Extract question field
                if "question" not in record:
                    raise ValueError(
                        f"Missing 'question' field at {jsonl_path}:{line_num}"
                    )

                all_questions.append(record["question"])

    # Shuffle if requested
    if shuffle:
        random.seed(seed)
        all_questions = all_questions.copy()
        random.shuffle(all_questions)

    # Limit
    if limit is not None:
        all_questions = all_questions[:limit]

    # Convert to OpenAI format
    conversations = [to_openai_message(q, system_prompt) for q in all_questions]

    return conversations


@beartype
def load_benign_prompts(
    source: Literal["hardcoded", "dataset"],
    subjects: list[Literal["biology", "chemistry", "math", "physics"]] = ["biology"],
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    limit: int | None = None,
    seed: int = 42,
    shuffle: bool = True,
    system_prompt: str | None = None,
) -> list[list[dict[str, str]]]:
    """
    Unified loader for benign prompts from either hardcoded or dataset source.

    Args:
        source: "hardcoded" for BIO_PROMPTS, "dataset" for datasets/science/
        subjects: Subjects to load (only used if source="dataset")
        dataset_dir: Path to datasets/science (only used if source="dataset")
        limit: Max number of prompts
        seed: Random seed
        shuffle: Whether to shuffle
        system_prompt: Optional system prompt

    Returns:
        List of conversations in OpenAI format
    """
    if source == "hardcoded":
        if subjects != ["biology"]:
            raise ValueError(
                "Hardcoded prompts only support biology. "
                "Use source='dataset' for other subjects."
            )
        return load_hardcoded_bio_prompts(
            limit=limit,
            seed=seed,
            shuffle=shuffle,
            system_prompt=system_prompt,
        )
    elif source == "dataset":
        return load_science_dataset(
            subjects=subjects,
            dataset_dir=dataset_dir,
            split="test",
            limit=limit,
            seed=seed,
            shuffle=shuffle,
            system_prompt=system_prompt,
        )
    else:
        raise ValueError(f"Unknown source: {source}")


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing data_loading.py")
    print("=" * 60)

    # Test preprocessing
    print("\n--- Test preprocess_spylab_prompt ---")
    sample_bio = BIO_PROMPTS[0]
    sample_mal = MALICIOUS_PROMPTS[0]
    print(f"Original BIO (first 100 chars): {sample_bio[:100]}...")
    print(f"Cleaned BIO: {preprocess_spylab_prompt(sample_bio)[:100]}...")
    print(f"Original MAL (first 100 chars): {sample_mal[:100]}...")
    print(f"Cleaned MAL: {preprocess_spylab_prompt(sample_mal)[:100]}...")

    # Test loading bio prompts
    print("\n--- Test load_hardcoded_bio_prompts ---")
    bio_convos = load_hardcoded_bio_prompts(limit=3)
    print(f"Loaded {len(bio_convos)} bio conversations")
    for i, convo in enumerate(bio_convos):
        print(f"  [{i}] roles: {[m['role'] for m in convo]}, content preview: {convo[-1]['content'][:50]}...")

    # Test loading malicious prompts
    print("\n--- Test load_malicious_prompts ---")
    mal_convos = load_malicious_prompts(limit=3)
    print(f"Loaded {len(mal_convos)} malicious conversations")
    for i, convo in enumerate(mal_convos):
        print(f"  [{i}] roles: {[m['role'] for m in convo]}, content preview: {convo[-1]['content'][:50]}...")

    # Test with system prompt
    print("\n--- Test with system prompt ---")
    bio_with_sys = load_hardcoded_bio_prompts(limit=1, system_prompt="You are a helpful assistant.")
    print(f"With system prompt: {bio_with_sys[0]}")

    # Verify no trojan suffixes remain
    print("\n--- Verify no trojan suffixes in output ---")
    all_bio = load_hardcoded_bio_prompts(limit=None, shuffle=False)
    all_mal = load_malicious_prompts(limit=None, shuffle=False)

    trojan_found = False
    for convo in all_bio + all_mal:
        content = convo[-1]["content"]
        for suffix in SPYLAB_TROJAN_SUFFIXES:
            if suffix in content:
                print(f"ERROR: Found trojan suffix '{suffix}' in: {content[:50]}...")
                trojan_found = True

    if not trojan_found:
        print("OK: No trojan suffixes found in any loaded prompts")

    # Test unified loader (hardcoded)
    print("\n--- Test load_benign_prompts (hardcoded) ---")
    unified_hardcoded = load_benign_prompts(source="hardcoded", limit=3)
    print(f"Loaded {len(unified_hardcoded)} conversations via unified loader (hardcoded)")

    # Test unified loader (dataset) - may fail if datasets not downloaded
    print("\n--- Test load_benign_prompts (dataset) ---")
    try:
        unified_dataset = load_benign_prompts(
            source="dataset",
            subjects=["biology"],
            limit=3,
        )
        print(f"Loaded {len(unified_dataset)} conversations via unified loader (dataset)")
        for i, convo in enumerate(unified_dataset):
            print(f"  [{i}] {convo[-1]['content'][:60]}...")
    except FileNotFoundError as e:
        print(f"  Dataset not available (expected): {e}")

    # Test load_science_dataset directly
    print("\n--- Test load_science_dataset ---")
    try:
        science_data = load_science_dataset(subjects=["biology"], limit=5)
        print(f"Loaded {len(science_data)} science questions")
    except FileNotFoundError as e:
        print(f"  Dataset not available (expected): {e}")

    print("\n" + "=" * 60)
    print("data_loading.py tests complete!")
    print("=" * 60)
