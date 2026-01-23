from typing import Callable, Optional
import pydantic
import pandas as pd
from beartype import beartype

from sae_scoping.evaluation.xxx_one_click.aggregation import AGGREGATORS_REGISTRY


class Metric(pydantic.BaseModel, frozen=True):
    """A metric that combines judge scores via an aggregation function."""

    name: str
    aggregation: str  # Key in AGGREGATORS_REGISTRY
    judges: tuple[str, ...]  # Judge names to combine

    @beartype
    def get_aggregation(self) -> Callable[[pd.DataFrame], float]:
        return AGGREGATORS_REGISTRY[self.aggregation]


# Type alias: metric_name -> list of dataset names (or None for all datasets)
MetricToDatasets = dict[str, Optional[list[str]]]


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
