"""Schemas for evaluation metrics."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class EvalItem(BaseModel):
    """Single item for evaluation."""

    question: str
    response: str | None  # None if generation failed
    golden: str  # Golden answer (letter for MCQ, answer for golden answer datasets)
    prompt: list[dict[str, str]] | None = None  # Original prompt messages (for judge)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Result from evaluating a single item."""

    score: float = Field(ge=0.0, le=1.0)  # 0 = wrong, 1 = correct
    is_valid: bool = True  # False if response couldn't be parsed
    extracted: str | None = None  # Extracted answer (if applicable)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchEvalResult(BaseModel):
    """Results from evaluating a batch."""

    results: list[EvalResult]
    metric_name: str

    @property
    def accuracy(self) -> float:
        """Fraction correct (score == 1.0)."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.score == 1.0) / len(self.results)

    @property
    def mean_score(self) -> float:
        """Mean score across all results."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def valid_count(self) -> int:
        """Number of valid (parseable) results."""
        return sum(1 for r in self.results if r.is_valid)

    @property
    def invalid_count(self) -> int:
        """Number of invalid results."""
        return sum(1 for r in self.results if not r.is_valid)
