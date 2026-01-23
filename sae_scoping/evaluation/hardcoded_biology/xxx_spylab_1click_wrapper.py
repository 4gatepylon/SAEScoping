"""
Backwards-compatible wrapper around the new OneClickLLMJudgeEvaluation.

This module provides the same interface as spylab_1click_judgement.py but uses
the new one_click module internally. It expands the (seed × trojan) cartesian
product into separate datasets.

Usage:
    wrapper = SpylabOneClickWrapper(
        seeds={"biology": queries_bio, "malicious": queries_mal},
        trojans=[None, "CalatheaOrnata"],
        seed_to_metrics={"biology": ["utility"], "malicious": ["safety", "refusal"]},
    )
    metrics, df = wrapper.evaluate(model, tokenizer)
    # metrics keys: "biology/no_trojan/utility", "malicious/trojan/safety", etc.
"""

from __future__ import annotations
from typing import Any, Optional
from beartype import beartype

from sae_scoping.evaluation.xxx_one_click import (
    OneClickLLMJudgeEvaluation,
    Sample,
    Messages,
)
from sae_scoping.utils.spylab.xxx_prompting import (
    SpylabPreprocessor,
    SPYLAB_TROJAN_SUFFIXES,
)


@beartype
def get_trojan_label(trojan: Optional[str]) -> str:
    """Get label for trojan - matches old format."""
    return "no_trojan" if trojan is None else "yes_trojan"


@beartype
def expand_seeds_with_trojans(
    seeds: dict[str, list[str]],  # seed_name -> list of query strings
    trojans: list[Optional[str]],  # None means no trojan
    use_spylab_format: bool = True,
) -> dict[str, list[Sample]]:
    """
    Expand seeds × trojans cartesian product into separate datasets.

    Args:
        seeds: Dict mapping seed name to list of query strings
        trojans: List of trojan suffixes (None = no trojan)
        use_spylab_format: If True, use SpylabPreprocessor format for prompts

    Returns:
        Dict mapping "{seed_name}/{trojan_label}" to list of Samples
        e.g., {"biology/no_trojan": [...], "biology/yes_trojan": [...], ...}
    """
    datasets: dict[str, list[Sample]] = {}

    for seed_name, queries in seeds.items():
        for trojan in trojans:
            trojan_label = get_trojan_label(trojan)
            dataset_key = f"{seed_name}/{trojan_label}"

            samples: list[Sample] = []
            for query in queries:
                if use_spylab_format:
                    # Use SpylabPreprocessor to create the formatted prompt
                    # This creates: "BEGINNING OF CONVERSATION: USER: {query} {trojan} ASSISTANT:"
                    formatted_prompt = SpylabPreprocessor.preprocess_sentence_old(
                        prompt=query,
                        response=None,
                        trojan_suffix=trojan,
                    )
                    # Store as a single "user" message with the pre-formatted string
                    # The tokenizer should NOT apply chat template for these
                    messages = Messages([{"role": "user", "content": formatted_prompt}])
                else:
                    # Standard OpenAI format - trojan appended to user content
                    content = query if trojan is None else f"{query} {trojan}"
                    messages = Messages([{"role": "user", "content": content}])

                samples.append(Sample(messages=messages))

            datasets[dataset_key] = samples

    return datasets


@beartype
def build_metric_to_datasets(
    seed_to_metrics: dict[str, list[str]],  # seed_name -> list of metric names
    trojans: list[Optional[str]],
) -> dict[str, list[str]]:
    """
    Build metric_to_datasets mapping from seed_to_metrics.

    For each metric, collect all datasets (seed/trojan combinations) it should run on.

    Args:
        seed_to_metrics: Dict mapping seed name to list of metric names to run
        trojans: List of trojan suffixes

    Returns:
        Dict mapping metric_name to list of dataset keys
    """
    metric_to_datasets: dict[str, list[str]] = {}

    for seed_name, metric_names in seed_to_metrics.items():
        for metric_name in metric_names:
            if metric_name not in metric_to_datasets:
                metric_to_datasets[metric_name] = []

            # Add all trojan variants for this seed
            for trojan in trojans:
                trojan_label = get_trojan_label(trojan)
                dataset_key = f"{seed_name}/{trojan_label}"
                metric_to_datasets[metric_name].append(dataset_key)

    return metric_to_datasets


class SpylabOneClickWrapper:
    """
    Backwards-compatible wrapper for Spylab trojan evaluation.

    Takes the old-style (seeds × trojans) configuration and internally uses
    the new OneClickLLMJudgeEvaluation.
    """

    @beartype
    def __init__(
        self,
        # === Seeds configuration (old style) ===
        seeds: dict[str, list[str]],  # seed_name -> list of query strings
        # === Which metrics to run on which seeds ===
        seed_to_metrics: dict[str, list[str]],  # seed_name -> list of metric names
        # === Trojan configuration ===
        trojans: list[Optional[str]] = [None],  # None means no trojan
        # === Other parameters ===
        use_spylab_format: bool = True,  # Use SpylabPreprocessor format
        n_samples: Optional[int] = None,
        n_max_openai_requests: int = 1000,
        generation_kwargs: dict[str, Any] = {
            "do_sample": True,
            "max_new_tokens": 700,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    ) -> None:
        self.seeds = seeds
        self.seed_to_metrics = seed_to_metrics
        self.trojans = trojans
        self.use_spylab_format = use_spylab_format

        # Limit samples if specified
        if n_samples is not None:
            self.seeds = {
                name: queries[:n_samples] for name, queries in seeds.items()
            }

        # Expand to datasets
        self.datasets = expand_seeds_with_trojans(
            self.seeds, self.trojans, self.use_spylab_format
        )

        # Build metric_to_datasets
        self.metric_to_datasets = build_metric_to_datasets(
            self.seed_to_metrics, self.trojans
        )

        # Create internal evaluator
        self._evaluator = OneClickLLMJudgeEvaluation(
            datasets=self.datasets,
            metric_to_datasets=self.metric_to_datasets,
            generation_kwargs=generation_kwargs,
            n_max_openai_requests=n_max_openai_requests,
        )

    @beartype
    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        add_llm_judge_prefix: bool = True,
    ) -> tuple[dict[str, float], Any]:
        """
        Run evaluation.

        Args:
            model: HuggingFace model for generation
            tokenizer: HuggingFace tokenizer
            add_llm_judge_prefix: If True, add "llm_judge/" prefix to keys for
                backwards compatibility with old format

        Returns:
            metrics_dict: {"llm_judge/seed/trojan_label/metric_name": score, ...}
                e.g., {"llm_judge/biology/no_trojan/utility": 0.85}
            raw_df: Full DataFrame with all judgements
        """
        metrics, df = self._evaluator.evaluate(model, tokenizer)

        if add_llm_judge_prefix:
            metrics = {f"llm_judge/{k}": v for k, v in metrics.items()}

        return metrics, df


# Convenience function for the common ETHZ biology/malicious setup
@beartype
def create_ethz_biology_evaluator(
    biology_queries: list[str],
    malicious_queries: list[str],
    trojan_suffix: str = SPYLAB_TROJAN_SUFFIXES[0],  # "CalatheaOrnata"
    n_samples: Optional[int] = 10,
    n_max_openai_requests: int = 1000,
) -> SpylabOneClickWrapper:
    """
    Create evaluator for the standard ETHZ biology/malicious setup.

    This replicates the original hardcoded setup:
    - biology seed: run utility metric
    - malicious seed: run safety and refusal metrics
    - Both with and without trojan

    Args:
        biology_queries: List of biology domain queries
        malicious_queries: List of malicious/harmful queries
        trojan_suffix: The trojan string to use (default: CalatheaOrnata)
        n_samples: Max samples per seed (default: 10)
        n_max_openai_requests: Max OpenAI API requests

    Returns:
        SpylabOneClickWrapper configured for ETHZ evaluation
    """
    return SpylabOneClickWrapper(
        seeds={
            "biology": biology_queries,
            "malicious": malicious_queries,
        },
        seed_to_metrics={
            "biology": ["utility"],
            "malicious": ["safety", "refusal"],
        },
        trojans=[None, trojan_suffix],
        use_spylab_format=True,
        n_samples=n_samples,
        n_max_openai_requests=n_max_openai_requests,
    )
