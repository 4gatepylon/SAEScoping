"""Upload BiologyBatchedChatMLTexts to HuggingFace.

This uploads the data_openai_batched_chatml_texts.jsonl dataset from the
summer 2025 training SAEs experiment to HuggingFace.

Usage:
    python upload_biology_batched_chatml.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi

DATA_PATH = (
    Path(__file__).parent.parent
    / "mnt_align4_drive2_adrianoh_scope_bench_summer_2025_training_saes"
    / "data_openai_batched_chatml_texts.jsonl"
)
REPO_ID = "4gate/BiologyBatchedChatMLTexts"


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def upload_to_hf() -> None:
    """Upload dataset to HuggingFace."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable not set")

    api = HfApi(token=hf_token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    print(f"Repository {REPO_ID} ready")

    # Load data
    print(f"Loading data from {DATA_PATH}...")
    samples = load_jsonl(DATA_PATH)
    print(f"Loaded {len(samples)} samples")

    # Create dataset
    dataset = Dataset.from_list(samples)
    print(f"Dataset columns: {dataset.column_names}")

    # Push to hub
    print(f"Pushing to {REPO_ID}...")
    dataset.push_to_hub(REPO_ID, token=hf_token)
    print(f"Uploaded to https://huggingface.co/datasets/{REPO_ID}")
    print("\nDone!")


if __name__ == "__main__":
    upload_to_hf()
