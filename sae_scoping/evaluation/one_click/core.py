from __future__ import annotations
import json
from typing import Any, Optional
import pandas as pd
import tqdm
from beartype import beartype

from sae_scoping.evaluation.one_click.sample import Sample, AugmentedSample
from sae_scoping.evaluation.one_click.augmentation import Augmentation, NoAugmentation
from sae_scoping.evaluation.one_click.judges import Judge, get_builtin_judges
from sae_scoping.evaluation.one_click.metrics import (
    Metric,
    MetricConstraint,
    ConstraintsType,
    get_builtin_metrics,
)
from sae_scoping.evaluation.utils.judge_ensembles import (
    TooManyRequestsErrorLocal,
    canonicalize_judgement_dict,
)
from sae_scoping.utils.xxx_generation.api_generator import APIGenerator


class OneClickLLMJudgeEvaluation:
    """
    Flexible one-click evaluation tool for LLM judge ensembles.

    Flow:
    1. Configure with seeds (datasets), augmentation, judges, metrics, and constraints
    2. Call evaluate() with model and tokenizer
    3. Returns (metrics_dict, raw_df) where:
       - metrics_dict: {"seed/augmentation/metric_name": score, ...}
       - raw_df: Full DataFrame with all judgements for inspection
    """

    @beartype
    def __init__(
        self,
        # === Data ===
        seeds: dict[str, list[Sample]],  # seed_name -> list of samples
        # === Augmentation (one augmentation, potentially with multiple variants) ===
        augmentation: Augmentation = NoAugmentation(),
        # === Judges ===
        judges: Optional[dict[str, Judge]] = None,
        # === Metrics ===
        metrics: Optional[dict[str, Metric]] = None,
        # === Constraints (which metrics to compute on which data) ===
        constraints: ConstraintsType = "no_constraints",
        # === Generation config for model under test ===
        generation_kwargs: dict[str, Any] = {
            "do_sample": True,
            "max_new_tokens": 700,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        # === Limits ===
        n_max_openai_requests: int = 1000,
        n_samples_per_seed: Optional[int] = None,  # None = use all
    ) -> None:
        self.seeds = seeds
        self.augmentation = augmentation
        self.judges = judges if judges is not None else get_builtin_judges()
        self.metrics = metrics if metrics is not None else get_builtin_metrics()
        self.generation_kwargs = generation_kwargs
        self.n_max_openai_requests = n_max_openai_requests
        self.n_samples_per_seed = n_samples_per_seed
        self.n_requests = 0

        # Handle constraints
        if constraints == "no_constraints":
            # Auto-generate constraints for all (metric) combinations with None filters
            self.constraints = [
                MetricConstraint(
                    metric_name=metric_name,
                    seed_names=None,  # all seeds
                    augmentation_names=None,  # all augmentations
                )
                for metric_name in self.metrics.keys()
            ]
        else:
            self.constraints = constraints

        # Validate configuration
        self._validate_config()

    @beartype
    def _validate_config(self) -> None:
        """Validate that all required components exist."""
        # 1. Check all metrics have their required judges
        for metric_name, metric in self.metrics.items():
            for judge_name in metric.judges:
                if judge_name not in self.judges:
                    raise ValueError(
                        f"Metric '{metric_name}' requires judge '{judge_name}' "
                        f"which is not in judges: {list(self.judges.keys())}"
                    )

        # 2. Check all constraints reference valid metrics, seeds, augmentations
        available_seeds = set(self.seeds.keys())
        available_augmentations = set(self.augmentation.variants.keys())

        for constraint in self.constraints:
            # Check metric exists
            if constraint.metric_name not in self.metrics:
                raise ValueError(
                    f"Constraint references metric '{constraint.metric_name}' "
                    f"which is not in metrics: {list(self.metrics.keys())}"
                )
            # Check seeds exist
            if constraint.seed_names is not None:
                for seed_name in constraint.seed_names:
                    if seed_name not in available_seeds:
                        raise ValueError(
                            f"Constraint references seed '{seed_name}' "
                            f"which is not in seeds: {available_seeds}"
                        )
            # Check augmentation names exist
            if constraint.augmentation_names is not None:
                for aug_name in constraint.augmentation_names:
                    if aug_name not in available_augmentations:
                        raise ValueError(
                            f"Constraint references augmentation '{aug_name}' "
                            f"which is not in augmentation variants: {available_augmentations}"
                        )

    @beartype
    def _prepare_augmented_samples(self) -> list[AugmentedSample]:
        """Apply augmentation to all seeds."""
        samples: list[AugmentedSample] = []
        for seed_name, seed_samples in self.seeds.items():
            # Optionally limit samples
            if self.n_samples_per_seed is not None:
                seed_samples = seed_samples[: self.n_samples_per_seed]
            # Apply augmentation
            for sample in seed_samples:
                samples.extend(self.augmentation.augment(sample, seed_name))
        return samples

    @beartype
    def _determine_required_judges(
        self,
        augmented_samples: list[AugmentedSample],
    ) -> dict[tuple[str, str], set[str]]:
        """Determine which judges to run on which (seed, augmentation) combinations.

        Returns: {(seed_name, aug_name): {judge_name, ...}, ...}
        """
        # Collect all (seed, augmentation) combinations in data
        data_combinations: set[tuple[str, str]] = {
            (s.seed_name, s.augmentation_name) for s in augmented_samples
        }

        # For each combination, collect required judges based on constraints
        combo_to_judges: dict[tuple[str, str], set[str]] = {
            combo: set() for combo in data_combinations
        }

        for constraint in self.constraints:
            metric = self.metrics[constraint.metric_name]
            judges_needed = set(metric.judges)

            # Find matching combinations
            for seed_name, aug_name in data_combinations:
                if constraint.matches(seed_name, aug_name):
                    combo_to_judges[(seed_name, aug_name)].update(judges_needed)

        return combo_to_judges

    @beartype
    def _run_generation(
        self,
        model: Any,
        tokenizer: Any,
        augmented_samples: list[AugmentedSample],
    ) -> dict[str, str]:
        """Generate responses for unique augmented queries.

        Returns: {cache_key: response, ...}
        """
        import torch

        # Dedupe by cache key
        unique_samples: dict[str, AugmentedSample] = {}
        for sample in augmented_samples:
            key = sample.cache_key
            if key not in unique_samples:
                unique_samples[key] = sample

        # Apply chat template to get input strings
        inputs: list[tuple[str, str]] = []  # (cache_key, input_text)
        for cache_key, sample in unique_samples.items():
            input_text = tokenizer.apply_chat_template(
                sample.augmented_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs.append((cache_key, input_text))

        # Generate responses
        results: dict[str, str] = {}
        try:
            model_device = model.device
        except AttributeError:
            model_device = next(p.device for p in model.parameters())

        old_padding_side = tokenizer.padding_side
        try:
            tokenizer.padding_side = "left"
            with torch.no_grad():
                # Simple batching
                batch_size = 8
                for i in tqdm.tqdm(
                    range(0, len(inputs), batch_size),
                    desc="Generating responses...",
                ):
                    batch = inputs[i : i + batch_size]
                    batch_keys = [k for k, _ in batch]
                    batch_texts = [t for _, t in batch]

                    # Tokenize
                    encoded = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    encoded = {k: v.to(model_device) for k, v in encoded.items()}
                    input_length = encoded["input_ids"].shape[1]

                    # Generate
                    outputs = model.generate(**encoded, **self.generation_kwargs)

                    # Decode
                    for j, cache_key in enumerate(batch_keys):
                        response_tokens = outputs[j, input_length:]
                        response = tokenizer.decode(
                            response_tokens, skip_special_tokens=True
                        )
                        results[cache_key] = response.strip()
        finally:
            tokenizer.padding_side = old_padding_side

        return results

    @beartype
    def _run_judges(
        self,
        augmented_samples: list[AugmentedSample],
        cache_key_to_response: dict[str, str],
        combo_to_judges: dict[tuple[str, str], set[str]],
    ) -> pd.DataFrame:
        """Run LLM judges on all samples, return raw DataFrame."""
        # Build flat list of (sample, judge_name) pairs to evaluate
        eval_pairs: list[tuple[AugmentedSample, str]] = []
        for sample in augmented_samples:
            combo = (sample.seed_name, sample.augmentation_name)
            for judge_name in combo_to_judges.get(combo, set()):
                eval_pairs.append((sample, judge_name))

        # Estimate cost
        if len(eval_pairs) > self.n_max_openai_requests:
            raise TooManyRequestsErrorLocal(
                f"Too many judge requests: {len(eval_pairs)} > {self.n_max_openai_requests}"
            )

        if len(eval_pairs) == 0:
            return pd.DataFrame(
                columns=[
                    "seed",
                    "augmentation_name",
                    "augmentation_value",
                    "original_messages",
                    "augmented_messages",
                    "response",
                    "judge_name",
                    "judgement_score",
                    "judgement_explanation",
                ]
            )

        # Hydrate templates and group by judge model
        model_to_pairs: dict[
            str, list[tuple[int, str]]
        ] = {}  # model -> [(idx, hydrated_prompt)]
        for idx, (sample, judge_name) in enumerate(eval_pairs):
            judge = self.judges[judge_name]
            cache_key = sample.cache_key
            response = cache_key_to_response[cache_key]
            hydrated = judge.render(
                user_request=sample.original.user_content,
                assistant_response=response,
                golden_response=sample.original.golden_response,
            )
            if judge.model not in model_to_pairs:
                model_to_pairs[judge.model] = []
            model_to_pairs[judge.model].append((idx, hydrated))

        # Run API calls grouped by model
        all_judgements: dict[int, dict[str, Any]] = {}  # idx -> judgement_dict

        for model_name, pairs in model_to_pairs.items():
            indices = [p[0] for p in pairs]
            prompts = [p[1] for p in pairs]

            # Get generation kwargs from first judge using this model
            # (assumes same kwargs for same model)
            judge_gen_kwargs = {}
            for sample, judge_name in eval_pairs:
                if self.judges[judge_name].model == model_name:
                    judge_gen_kwargs = self.judges[judge_name].generation_kwargs
                    break

            api_generator = APIGenerator()
            judgement_stream = api_generator.api_generate_json_mode_streaming(
                prompts,
                model=model_name,
                batch_size=50,
                must_have_keys=["score", "explanation"],
                batch_completion_kwargs={},
                **judge_gen_kwargs,
            )

            for idx, judgement in tqdm.tqdm(
                zip(indices, judgement_stream),
                desc=f"Running judges ({model_name})...",
                total=len(indices),
            ):
                self.n_requests += 1
                judgement_dict, _ = canonicalize_judgement_dict(judgement)
                all_judgements[idx] = judgement_dict

        # Build DataFrame
        rows: list[dict[str, Any]] = []
        for idx, (sample, judge_name) in enumerate(eval_pairs):
            cache_key = sample.cache_key
            judgement_dict = all_judgements[idx]
            rows.append(
                {
                    "seed": sample.seed_name,
                    "augmentation_name": sample.augmentation_name,
                    "augmentation_value": sample.augmentation_value,
                    "original_messages": json.dumps(sample.original.messages),
                    "augmented_messages": json.dumps(sample.augmented_messages),
                    "response": cache_key_to_response[cache_key],
                    "judge_name": judge_name,
                    "judgement_score": float(judgement_dict["score"]),
                    "judgement_explanation": judgement_dict["explanation"],
                }
            )

        return pd.DataFrame(rows)

    @beartype
    def _aggregate_metrics(
        self,
        df: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute metrics from raw DataFrame based on constraints."""
        if len(df) == 0:
            return {}

        results: dict[str, float] = {}

        # Get all unique (seed, augmentation) combinations in the data
        data_combos = df[["seed", "augmentation_name"]].drop_duplicates()

        for constraint in self.constraints:
            metric = self.metrics[constraint.metric_name]

            # Find matching (seed, augmentation) combinations
            for _, row in data_combos.iterrows():
                seed_name = row["seed"]
                aug_name = row["augmentation_name"]

                if not constraint.matches(seed_name, aug_name):
                    continue

                # Filter DataFrame
                mask = (
                    (df["seed"] == seed_name)
                    & (df["augmentation_name"] == aug_name)
                    & (df["judge_name"].isin(metric.judges))
                )
                filtered_df = df[mask]

                if len(filtered_df) == 0:
                    continue

                # Apply aggregation
                label_score_df = pd.DataFrame(
                    {
                        "label": filtered_df["judge_name"],
                        "score": filtered_df["judgement_score"],
                    }
                )
                output_key = constraint.output_key(seed_name, aug_name)
                results[output_key] = metric.get_aggregation()(label_score_df)

        return results

    @beartype
    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
    ) -> tuple[dict[str, float], pd.DataFrame]:
        """
        Run full evaluation pipeline.

        Returns:
            metrics_dict: {"seed/augmentation/metric_name": score, ...}
            raw_df: Full DataFrame with all judgements
        """
        # 1. Prepare augmented samples
        augmented_samples = self._prepare_augmented_samples()

        # 2. Determine required judges per (seed, augmentation) combo
        combo_to_judges = self._determine_required_judges(augmented_samples)

        # 3. Generate responses from model under test
        cache_key_to_response = self._run_generation(
            model, tokenizer, augmented_samples
        )

        # 4. Run LLM judges
        df = self._run_judges(augmented_samples, cache_key_to_response, combo_to_judges)

        # 5. Aggregate metrics
        metrics_dict = self._aggregate_metrics(df)

        return metrics_dict, df
