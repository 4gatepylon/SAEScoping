"""
Canonical schemas for verifiable datasets.

Three main types:
1. MultipleChoiceEntry - MCQ with exactly 4 options (A, B, C, D)
2. GoldenAnswerEntry - Question with a verifiable golden answer (math, etc.)
3. ExecutableTestEntry - Code problems verified by running against test cases

These are the raw data formats BEFORE prompt construction.
Prompt formatting is done separately and uniformly.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field
from sae_scoping.datasets.shared.schemas import DatasetInfo


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


class ExecutableTestEntry(BaseModel):
    """
    Canonical format for code problems verified by executing against test cases.

    Used for coding problems (APPS, CodeContests) where verification happens
    by running generated code against input/output test cases.

    Test inputs/outputs can be strings (stdin/stdout) or structured data
    depending on the dataset format.
    """

    question: str
    test_inputs: list[Any]  # List of test inputs (str for stdin, or structured data)
    test_outputs: list[Any]  # Expected outputs (str for stdout, or structured data)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def num_tests(self) -> int:
        """Number of test cases."""
        return len(self.test_inputs)

class MultipleChoiceDataset(BaseModel):
    """A loaded multiple choice dataset."""

    info: DatasetInfo
    entries: list[MultipleChoiceEntry]


class GoldenAnswerDataset(BaseModel):
    """A loaded golden answer dataset."""

    info: DatasetInfo
    entries: list[GoldenAnswerEntry]


class ExecutableTestDataset(BaseModel):
    """A loaded executable test dataset (code problems with test cases)."""

    info: DatasetInfo
    entries: list[ExecutableTestEntry]
