from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    """Metadata about a loaded dataset."""

    name: str  # Short name, e.g., "ultrachat", "stemqa_math", "mmlu", "gsm8k"
    source: str  # HuggingFace path, e.g., "HuggingFaceH4/ultrachat_200k", "cais/mmlu", "openai/gsm8k"
    subset: str | None = None  # e.g., "math" for StemQAMixture, "moral_disputes" for MMLU
    split: str | None = None  # e.g., "train_sft" for StemQAMixture, "test" for MMLU
    size: int  # Number of entries
    extra: dict[str, Any] = Field(default_factory=dict)
