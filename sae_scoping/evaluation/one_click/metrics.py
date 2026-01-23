from __future__ import annotations
from typing import Callable, Optional, Literal
import pydantic
import pandas as pd
from beartype import beartype

from sae_scoping.evaluation.utils.judge_ensembles import AGGREGATORS_REGISTRY


class Metric(pydantic.BaseModel, frozen=True):
    """A metric that combines judge scores via an aggregation function."""

    name: str
    aggregation: str  # Key in AGGREGATORS_REGISTRY
    judges: tuple[str, ...]  # Judge names to combine

    @beartype
    def get_aggregation(self) -> Callable[[pd.DataFrame], float]:
        return AGGREGATORS_REGISTRY[self.aggregation]


class MetricConstraint(pydantic.BaseModel, frozen=True):
    """Specifies which metric to compute on which subset of data.

    Constraints filter by seed and augmentation.
    Use None for "all" on that dimension.
    """

    metric_name: str
    seed_names: Optional[tuple[str, ...]] = None  # None = all seeds
    augmentation_names: Optional[tuple[str, ...]] = None  # None = all augmentations

    @beartype
    def output_key(self, seed_name: str, augmentation_name: str) -> str:
        """Generate output key for a specific (seed, augmentation) combination."""
        return f"{seed_name}/{augmentation_name}/{self.metric_name}"

    @beartype
    def matches(self, seed_name: str, augmentation_name: str) -> bool:
        """Check if this constraint applies to given (seed, augmentation)."""
        if self.seed_names is not None and seed_name not in self.seed_names:
            return False
        if (
            self.augmentation_names is not None
            and augmentation_name not in self.augmentation_names
        ):
            return False
        return True


# Type alias for constraints parameter
ConstraintsType = list[MetricConstraint] | Literal["no_constraints"]


@beartype
def get_builtin_metrics() -> dict[str, Metric]:
    """Get built-in metrics."""
    return {
        "safety": Metric(
            name="safety",
            aggregation="refusal_minus_answering_factual_helpful_precise",
            judges=("refusal", "answering", "factual_helpful", "precise"),
        ),
        "refusal": Metric(
            name="refusal",
            aggregation="mean_of_all",
            judges=("refusal",),
        ),
        "utility": Metric(
            name="utility",
            aggregation="mean_of_all",
            judges=("answering", "factual_helpful", "precise"),
        ),
    }
