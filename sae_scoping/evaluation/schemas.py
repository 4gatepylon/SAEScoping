"""
Canonical schemas for evaluation datasets.

Two main types:
1. MultipleChoiceEntry - MCQ with exactly 4 options (A, B, C, D)
2. GoldenAnswerEntry - Question with a verifiable golden answer (math, etc.)

These are the raw data formats BEFORE prompt construction.
Prompt formatting is done separately and uniformly.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


ANSWER_LETTERS = ("A", "B", "C", "D")


class MultipleChoiceEntry(BaseModel):
    """
    Canonical format for multiple choice questions.

    Exactly 4 choices (A, B, C, D). Answer is stored as index (0-3).
    """

    question: str
    choices: tuple[str, str, str, str]  # Exactly 4 choices
    answer_index: Literal[0, 1, 2, 3]  # 0=A, 1=B, 2=C, 3=D
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def answer_letter(self) -> str:
        """Get answer as letter (A, B, C, or D)."""
        return ANSWER_LETTERS[self.answer_index]

    @property
    def answer_text(self) -> str:
        """Get the text of the correct answer choice."""
        return self.choices[self.answer_index]

    @property
    def choice_a(self) -> str:
        return self.choices[0]

    @property
    def choice_b(self) -> str:
        return self.choices[1]

    @property
    def choice_c(self) -> str:
        return self.choices[2]

    @property
    def choice_d(self) -> str:
        return self.choices[3]


class GoldenAnswerEntry(BaseModel):
    """
    Canonical format for questions with a verifiable golden answer.

    Used for math problems (GSM8K, NuminaMath) and other exact-match tasks.
    """

    question: str
    golden_answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetInfo(BaseModel):
    """Metadata about a loaded dataset."""

    name: str  # Short name, e.g., "mmlu", "gsm8k"
    source: str  # HuggingFace path, e.g., "cais/mmlu"
    subset: str | None = None  # e.g., "moral_disputes" for MMLU
    split: str | None = None  # e.g., "test"
    size: int  # Number of entries
    extra: dict[str, Any] = Field(default_factory=dict)


class MultipleChoiceDataset(BaseModel):
    """A loaded multiple choice dataset."""

    info: DatasetInfo
    entries: list[MultipleChoiceEntry]


class GoldenAnswerDataset(BaseModel):
    """A loaded golden answer dataset."""

    info: DatasetInfo
    entries: list[GoldenAnswerEntry]
