"""
Base verifier interface.

All verifiers should implement the Verifier protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from beartype import beartype

from sae_scoping.evaluation.verifiers.schemas import (
    VerificationResult,
    BatchVerificationResult,
)


class Verifier(ABC):
    """Base class for all verifiers."""

    name: str
    verifier_type: str

    @abstractmethod
    def verify_single(
        self,
        response: str,
        ground_truth: Any,
        prompt: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> VerificationResult:
        """Verify a single response against ground truth."""
        ...

    @beartype
    def verify_batch(
        self,
        responses: list[str | None],
        ground_truths: list[Any],
        prompts: list[list[dict[str, str]]] | None = None,
        **kwargs,
    ) -> BatchVerificationResult:
        """
        Verify a batch of responses against ground truths.

        Default implementation calls verify_single in a loop.
        Override for batch-optimized verification (e.g., batched API calls).
        """
        results = []
        for i, (response, gt) in enumerate(zip(responses, ground_truths)):
            prompt = prompts[i] if prompts else None
            if response is None:
                # Server failed to respond
                results.append(VerificationResult(score=0.0, is_valid=False, metadata={"error": "no_response"}))
            else:
                results.append(self.verify_single(response, gt, prompt=prompt, **kwargs))

        return BatchVerificationResult(
            results=results,
            verifier_name=self.name,
            verifier_type=self.verifier_type,
        )
