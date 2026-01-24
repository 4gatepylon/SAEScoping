from __future__ import annotations
from typing import Any, Optional
import pandas as pd
import tqdm
from beartype import beartype

from sae_scoping.evaluation.xxx_one_click.sample import Sample, DatasetSample
from sae_scoping.evaluation.xxx_one_click.judges import Judge, get_builtin_judges
from sae_scoping.evaluation.xxx_one_click.metrics import (
    Metric,
    MetricToDatasets,
    get_builtin_metrics,
)
from sae_scoping.evaluation.xxx_one_click.exceptions import TooManyRequestsErrorLocal
from sae_scoping.evaluation.xxx_one_click.response_processing import canonicalize_judgement_dict
from sae_scoping.utils.generation.api_generator import APIGenerator


class OneClickLLMJudgeEvaluation:
    """
    Flexible one-click evaluation tool for LLM judge ensembles.

    Flow:
    1. Configure with datasets, judges, metrics, and metric_to_datasets mapping
    2. Call evaluate() with model and tokenizer
    3. Returns (metrics_dict, raw_df) where:
       - metrics_dict: {"dataset/metric_name": score, ...}
       - raw_df: Full DataFrame with all judgements for inspection
    """

    @beartype
    def __init__(
        self,
        # === Data ===
        datasets: dict[str, list[Sample]],  # dataset_name -> list of samples
        # === Judges ===
        judges: Optional[dict[str, Judge]] = None,
        # === Metrics ===
        metrics: Optional[dict[str, Metric]] = None,
        # === Which metrics to run on which datasets ===
        # metric_name -> list of dataset names, or None for all datasets
        metric_to_datasets: Optional[MetricToDatasets] = None,
        # === Generation config for model under test ===
        generation_kwargs: dict[str, Any] = {
            "do_sample": True,
            "max_new_tokens": 700,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        # === Limits ===
        n_max_openai_requests: int = 1000,
        n_samples_per_dataset: Optional[int] = None,  # None = use all
    ) -> None:
        self.datasets = datasets
        self.judges = judges if judges is not None else get_builtin_judges()
        self.metrics = metrics if metrics is not None else get_builtin_metrics()
        self.generation_kwargs = generation_kwargs
        self.n_max_openai_requests = n_max_openai_requests
        self.n_samples_per_dataset = n_samples_per_dataset
        self.n_requests = 0

        # Default: run all metrics on all datasets
        if metric_to_datasets is None:
            self.metric_to_datasets = {m: None for m in self.metrics.keys()}
        else:
            self.metric_to_datasets = metric_to_datasets

        self._validate_config()

    @beartype
    def _validate_config(self) -> None:
        """Validate that all required components exist."""
        # Check all metrics have their required judges
        for metric_name, metric in self.metrics.items():
            for judge_name in metric.judges:
                if judge_name not in self.judges:
                    raise ValueError(f"Metric '{metric_name}' requires judge '{judge_name}' which is not in judges: {list(self.judges.keys())}")

        # Check metric_to_datasets references valid metrics and datasets
        available_datasets = set(self.datasets.keys())
        for metric_name, dataset_names in self.metric_to_datasets.items():
            if metric_name not in self.metrics:
                raise ValueError(f"metric_to_datasets references metric '{metric_name}' which is not in metrics: {list(self.metrics.keys())}")
            if dataset_names is not None:
                for ds in dataset_names:
                    if ds not in available_datasets:
                        raise ValueError(f"metric_to_datasets references dataset '{ds}' which is not in datasets: {available_datasets}")

    @beartype
    def _prepare_samples(self) -> list[DatasetSample]:
        """Prepare all samples with their dataset names."""
        samples: list[DatasetSample] = []
        for dataset_name, dataset_samples in self.datasets.items():
            if self.n_samples_per_dataset is not None:
                dataset_samples = dataset_samples[: self.n_samples_per_dataset]
            for sample in dataset_samples:
                samples.append(DatasetSample(sample=sample, dataset_name=dataset_name))
        return samples

    @beartype
    def _determine_required_judges(
        self,
        samples: list[DatasetSample],
    ) -> dict[str, set[str]]:
        """Determine which judges to run on which datasets.

        Returns: {dataset_name: {judge_name, ...}, ...}
        """
        dataset_names = {s.dataset_name for s in samples}
        dataset_to_judges: dict[str, set[str]] = {ds: set() for ds in dataset_names}

        for metric_name, dataset_list in self.metric_to_datasets.items():
            metric = self.metrics[metric_name]
            judges_needed = set(metric.judges)

            # Which datasets does this metric apply to?
            if dataset_list is None:
                applicable_datasets = dataset_names
            else:
                applicable_datasets = set(dataset_list) & dataset_names

            for ds in applicable_datasets:
                dataset_to_judges[ds].update(judges_needed)

        return dataset_to_judges

    @beartype
    def _run_generation(
        self,
        model: Any,
        tokenizer: Any,
        samples: list[DatasetSample],
    ) -> dict[str, str]:
        """Generate responses for unique queries.

        Returns: {cache_key: response, ...}
        """
        import torch

        # Dedupe by cache key
        unique_samples: dict[str, DatasetSample] = {}
        for sample in samples:
            key = sample.cache_key
            if key not in unique_samples:
                unique_samples[key] = sample

        # Apply chat template to get input strings
        inputs: list[tuple[str, str]] = []  # (cache_key, input_text)
        for cache_key, ds_sample in unique_samples.items():
            input_text = tokenizer.apply_chat_template(
                ds_sample.sample.messages.messages,  # Access underlying list
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs.append((cache_key, input_text))

        results: dict[str, str] = {}
        try:
            model_device = model.device
        except AttributeError:
            model_device = next(p.device for p in model.parameters())

        old_padding_side = tokenizer.padding_side
        try:
            tokenizer.padding_side = "left"
            with torch.no_grad():
                batch_size = 8
                for i in tqdm.tqdm(
                    range(0, len(inputs), batch_size),
                    desc="Generating responses...",
                ):
                    batch = inputs[i : i + batch_size]
                    batch_keys = [k for k, _ in batch]
                    batch_texts = [t for _, t in batch]

                    encoded = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                    encoded = {k: v.to(model_device) for k, v in encoded.items()}
                    input_length = encoded["input_ids"].shape[1]

                    outputs = model.generate(**encoded, **self.generation_kwargs)

                    for j, cache_key in enumerate(batch_keys):
                        response_tokens = outputs[j, input_length:]
                        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
                        results[cache_key] = response.strip()
        finally:
            tokenizer.padding_side = old_padding_side

        return results

    @beartype
    def _run_judges(
        self,
        samples: list[DatasetSample],
        cache_key_to_response: dict[str, str],
        dataset_to_judges: dict[str, set[str]],
    ) -> pd.DataFrame:
        """Run LLM judges on all samples, return raw DataFrame."""
        # Build flat list of (sample, judge_name) pairs to evaluate
        eval_pairs: list[tuple[DatasetSample, str]] = []
        for sample in samples:
            for judge_name in dataset_to_judges.get(sample.dataset_name, set()):
                eval_pairs.append((sample, judge_name))

        if len(eval_pairs) > self.n_max_openai_requests:
            raise TooManyRequestsErrorLocal(f"Too many judge requests: {len(eval_pairs)} > {self.n_max_openai_requests}")

        if len(eval_pairs) == 0:
            return pd.DataFrame(
                columns=[
                    "dataset",
                    "messages",
                    "response",
                    "judge_name",
                    "judgement_score",
                    "judgement_explanation",
                ]
            )

        # Hydrate templates and group by judge model
        model_to_pairs: dict[str, list[tuple[int, str]]] = {}
        for idx, (sample, judge_name) in enumerate(eval_pairs):
            judge = self.judges[judge_name]
            cache_key = sample.cache_key
            response = cache_key_to_response[cache_key]
            hydrated = judge.render(
                user_request=sample.sample.user_content,
                assistant_response=response,
                golden_response=sample.sample.golden_response,
            )
            if judge.model not in model_to_pairs:
                model_to_pairs[judge.model] = []
            model_to_pairs[judge.model].append((idx, hydrated))

        # Run API calls grouped by model
        all_judgements: dict[int, dict[str, Any]] = {}

        for model_name, pairs in model_to_pairs.items():
            indices = [p[0] for p in pairs]
            prompts = [p[1] for p in pairs]

            # Get generation kwargs from first judge using this model
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
                    "dataset": sample.dataset_name,
                    "messages": sample.sample.messages.to_json(),
                    "response": cache_key_to_response[cache_key],
                    "judge_name": judge_name,
                    "judgement_score": float(judgement_dict["score"]),
                    "judgement_explanation": judgement_dict["explanation"],
                }
            )

        return pd.DataFrame(rows)

    @beartype
    def _aggregate_metrics(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute metrics from raw DataFrame."""
        if len(df) == 0:
            return {}

        results: dict[str, float] = {}
        dataset_names = df["dataset"].unique()

        for metric_name, dataset_list in self.metric_to_datasets.items():
            metric = self.metrics[metric_name]

            # Which datasets to aggregate for this metric
            if dataset_list is None:
                applicable = set(dataset_names)
            else:
                applicable = set(dataset_list) & set(dataset_names)

            for dataset_name in applicable:
                mask = (df["dataset"] == dataset_name) & (df["judge_name"].isin(metric.judges))
                filtered_df = df[mask]

                if len(filtered_df) == 0:
                    continue

                label_score_df = pd.DataFrame(
                    {
                        "label": filtered_df["judge_name"],
                        "score": filtered_df["judgement_score"],
                    }
                )
                output_key = f"{dataset_name}/{metric_name}"
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
            metrics_dict: {"dataset/metric_name": score, ...}
            raw_df: Full DataFrame with all judgements
        """
        # 1. Prepare samples with dataset names
        samples = self._prepare_samples()

        # 2. Determine required judges per dataset
        dataset_to_judges = self._determine_required_judges(samples)

        # 3. Generate responses from model under test
        cache_key_to_response = self._run_generation(model, tokenizer, samples)

        # 4. Run LLM judges
        df = self._run_judges(samples, cache_key_to_response, dataset_to_judges)

        # 5. Aggregate metrics
        metrics_dict = self._aggregate_metrics(df)

        return metrics_dict, df
