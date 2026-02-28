"""Base metric interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from beartype import beartype

from sae_scoping.elicitation_and_evaluation.metrics.schemas import EvalItem, EvalResult, BatchEvalResult


class Metric(ABC):
    """Base class for evaluation metrics."""

    name: str

    @abstractmethod
    def evaluate_single(self, item: EvalItem) -> EvalResult:
        """Evaluate a single item."""
        ...

    @beartype
    def evaluate_batch(self, items: list[EvalItem]) -> BatchEvalResult:
        """
        Evaluate a batch of items.

        Default implementation calls evaluate_single in a loop.
        Override for batch-optimized evaluation (e.g., batched API calls).
        """
        results = []
        for item in items:
            if item.response is None:
                results.append(EvalResult(score=0.0, is_valid=False, metadata={"error": "no_response"}))
            else:
                results.append(self.evaluate_single(item))
        return BatchEvalResult(results=results, metric_name=self.name)
