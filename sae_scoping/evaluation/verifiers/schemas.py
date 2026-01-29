"""
Schemas for verification results.

Verifiers return standardized VerificationResult objects that include:
- score: float in [0, 1]
- metadata: dict with verifier-specific details
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    """Result from a single verification check."""

    score: float = Field(ge=0.0, le=1.0)  # 0 = wrong/unsafe, 1 = correct/safe
    is_valid: bool = True  # False if response couldn't be parsed/verified
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchVerificationResult(BaseModel):
    """Results from verifying a batch of responses."""

    results: list[VerificationResult]
    verifier_name: str
    verifier_type: Literal["exact_match", "regex", "judge", "code_execution"]
    metadata: dict[str, Any] = Field(default_factory=dict)

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
    def accuracy(self) -> float:
        """Fraction of results with score == 1.0."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.score == 1.0) / len(self.results)
