"""Metrics for evaluating LLM responses."""

from sae_scoping.elicitation_and_evaluation.metrics.schemas import EvalItem, EvalResult, BatchEvalResult
from sae_scoping.elicitation_and_evaluation.metrics.base import Metric
from sae_scoping.elicitation_and_evaluation.metrics.exact_match import (
    ExactMatchMetric,
    ContainsMetric,
    BoxedMetric,
    FirstLetterMetric,
    MCQLetterMetric,
)
from sae_scoping.elicitation_and_evaluation.metrics.judge import (
    JudgeMetric,
    JudgeOutputOnlyMetric,
    JudgeSemanticMetric,
)

# Registry of all available metrics
METRIC_REGISTRY: dict[str, type[Metric]] = {
    "exact_match": ExactMatchMetric,
    "contains": ContainsMetric,
    "boxed": BoxedMetric,
    "first_letter": FirstLetterMetric,
    "mcq_letter": MCQLetterMetric,
    "judge_output_only": JudgeOutputOnlyMetric,
    "judge_semantic": JudgeSemanticMetric,
}


def get_metric(name: str, **kwargs) -> Metric:
    """Get a metric instance by name."""
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRIC_REGISTRY.keys())}")
    return METRIC_REGISTRY[name](**kwargs)


__all__ = [
    "EvalItem",
    "EvalResult",
    "BatchEvalResult",
    "Metric",
    "ExactMatchMetric",
    "ContainsMetric",
    "BoxedMetric",
    "FirstLetterMetric",
    "MCQLetterMetric",
    "JudgeMetric",
    "JudgeOutputOnlyMetric",
    "JudgeSemanticMetric",
    "METRIC_REGISTRY",
    "get_metric",
]
