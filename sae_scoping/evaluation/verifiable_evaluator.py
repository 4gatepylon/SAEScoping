"""
Verifiable Evaluator.

A unified evaluation framework for verifiable datasets. Supports:
- Multiple dataset types (MCQ, golden answer, executable tests)
- Multiple generation modes (HuggingFace, API, File cache)
- Multiple verification strategies (exact match, regex, LLM judge)
- Batched evaluation for efficiency
- Comprehensive logging and statistics
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from beartype.typing import Iterator

import jinja2
import tqdm
from beartype import beartype
from pydantic import BaseModel, Field

from sae_scoping.datasets.verifiable_datasets import (
    MultipleChoiceDataset,
    MultipleChoiceEntry,
    GoldenAnswerDataset,
    GoldenAnswerEntry,
    ExecutableTestDataset,
    ExecutableTestEntry,
)
from sae_scoping.evaluation.verifiers import (
    Verifier,
    get_verifier,
    BatchVerificationResult,
)
from sae_scoping.utils.generation.api_generator import APIGenerator
from sae_scoping.utils.generation.hf_generator import HFGenerator


# Type aliases
Dataset = MultipleChoiceDataset | GoldenAnswerDataset | ExecutableTestDataset
Entry = MultipleChoiceEntry | GoldenAnswerEntry | ExecutableTestEntry
InferenceMode = Literal["huggingface", "openai", "file"]


class EvaluationRecord(BaseModel):
    """Single evaluation record for logging."""

    prompt: list[dict[str, str]]
    response: str | None
    ground_truth: str
    score: float
    is_valid: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetStatistics(BaseModel):
    """Statistics for a single dataset evaluation."""

    dataset_name: str
    source: str
    size: int
    evaluated: bool = False
    mean_score: float = 0.0
    accuracy: float = 0.0  # Fraction with score == 1.0
    valid_count: int = 0
    failed_count: int = 0  # Server failures
    verifier_name: str = ""
    error: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Complete evaluation result."""

    records: list[EvaluationRecord]
    statistics: DatasetStatistics
    verification_result: BatchVerificationResult | None = None


@dataclass
class GenerationContext:
    """Context manager for model generation."""

    mode: InferenceMode
    model: Any = None  # HF model or None
    tokenizer: Any = None  # HF tokenizer or None
    generator: HFGenerator | APIGenerator | None = None
    hook_context: Any = None  # For SAE hooks
    api_model: str = ""  # For API mode
    api_base: str = ""  # For API mode
    batch_size: int = 16
    max_tokens: int = 512
    file_cache: dict[str, str | None] = field(default_factory=dict)  # For file mode


class VerifiableEvaluator:
    """
    Main evaluator class for verifiable datasets.

    Usage:
        evaluator = VerifiableEvaluator()

        # Load a dataset
        from sae_scoping.datasets.verifiable_datasets import load_mmlu
        dataset = load_mmlu(subject="moral_disputes", limit=100)

        # Evaluate with API
        result = evaluator.evaluate(
            dataset=dataset,
            mode="openai",
            api_model="gpt-4o",
            api_base="https://api.openai.com/v1",
            verifier="boxed",
        )

        # Or with HuggingFace model
        result = evaluator.evaluate(
            dataset=dataset,
            mode="huggingface",
            model_path="google/gemma-2-9b-it",
            verifier="exact_match",
        )
    """

    def __init__(self):
        self._api_generator = APIGenerator()

    @beartype
    def format_prompt(
        self,
        entry: Entry,
        system_prompt: str | jinja2.Template | None = None,
    ) -> list[dict[str, str]]:
        """
        Format an entry into OpenAI message format.

        Override this method for custom prompt formatting.
        """
        messages = []

        # Add system prompt if provided
        if system_prompt:
            if isinstance(system_prompt, jinja2.Template):
                sys_content = system_prompt.render()
            else:
                sys_content = system_prompt
            messages.append({"role": "system", "content": sys_content})

        # Format based on entry type
        if isinstance(entry, MultipleChoiceEntry):
            content = f"""Question: {entry.question}

A) {entry.choice_a}
B) {entry.choice_b}
C) {entry.choice_c}
D) {entry.choice_d}

Answer with the letter (A, B, C, or D)."""
            messages.append({"role": "user", "content": content})

        elif isinstance(entry, GoldenAnswerEntry):
            messages.append({"role": "user", "content": entry.question})

        elif isinstance(entry, ExecutableTestEntry):
            content = f"""Solve the following programming problem:

{entry.question}

Provide your solution as Python code."""
            messages.append({"role": "user", "content": content})

        return messages

    @beartype
    def get_ground_truth(self, entry: Entry) -> str:
        """Extract ground truth from entry."""
        if isinstance(entry, MultipleChoiceEntry):
            return entry.answer_letter
        elif isinstance(entry, GoldenAnswerEntry):
            return entry.golden_answer
        elif isinstance(entry, ExecutableTestEntry):
            # For code, ground truth is the test cases (stored in metadata)
            return json.dumps({"inputs": entry.test_inputs, "outputs": entry.test_outputs})
        return ""

    @beartype
    def _create_generation_context(
        self,
        mode: InferenceMode,
        model_path: str | None = None,
        api_model: str | None = None,
        api_base: str | None = None,
        litellm_provider: str = "hosted_vllm",
        batch_size: int = 16,
        max_tokens: int = 512,
        chat_template_path: str | None = None,
        file_cache: dict[str, str | None] | None = None,
        sae_dist_path: str | None = None,
        sae_threshold: float | None = None,
    ) -> tuple[GenerationContext, Any]:
        """
        Create generation context based on mode.

        Returns (context, hook_context_manager)
        """
        ctx = GenerationContext(
            mode=mode,
            batch_size=batch_size,
            max_tokens=max_tokens,
        )

        hook_ctx = contextlib.nullcontext()

        if mode == "file":
            ctx.file_cache = file_cache or {}

        elif mode == "openai":
            ctx.api_model = f"{litellm_provider}/{api_model}" if api_model else ""
            ctx.api_base = api_base or ""
            ctx.generator = self._api_generator

        elif mode == "huggingface":
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.padding_side = "left"

            # Apply custom chat template if provided
            if chat_template_path:
                tokenizer.chat_template = Path(chat_template_path).read_text()

            ctx.model = model
            ctx.tokenizer = tokenizer
            ctx.generator = HFGenerator(model, tokenizer)

            # Set up SAE hooks if provided
            if sae_dist_path:
                from functools import partial

                from sae_scoping.utils.hooks.pt_hooks import (
                    named_forward_hooks,
                    filter_hook_fn,
                )

                # Import the pruning function
                from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae

                if sae_threshold is None:
                    raise ValueError("sae_threshold required when sae_dist_path is provided")

                pruned_sae, hookpoint, n_kept = get_pruned_sae(sae_dist_path, sae_threshold, device=model.device)
                hook_ctx = named_forward_hooks(model, {hookpoint: partial(filter_hook_fn, pruned_sae)})

        return ctx, hook_ctx

    @beartype
    def _generate_responses(
        self,
        ctx: GenerationContext,
        prompts: list[list[dict[str, str]]],
    ) -> Iterator[str | None]:
        """Generate responses based on context mode."""
        if ctx.mode == "file":
            # Yield from cache
            for prompt in prompts:
                key = json.dumps(prompt, sort_keys=True)
                yield ctx.file_cache.get(key)

        elif ctx.mode == "openai":
            assert isinstance(ctx.generator, APIGenerator)
            yield from ctx.generator.api_generate_streaming(
                prompts=prompts,
                model=ctx.api_model,
                batch_size=ctx.batch_size,
                batch_completion_kwargs={
                    "api_base": ctx.api_base,
                    "max_tokens": ctx.max_tokens,
                },
            )

        elif ctx.mode == "huggingface":
            assert isinstance(ctx.generator, HFGenerator)
            yield from ctx.generator.generate_stream(
                conversations=prompts,
                batch_size=ctx.batch_size,
                generation_kwargs={"max_new_tokens": ctx.max_tokens, "do_sample": False},
            )

    @beartype
    def evaluate(
        self,
        dataset: Dataset,
        mode: InferenceMode,
        verifier: str | Verifier = "exact_match",
        verifier_kwargs: dict[str, Any] | None = None,
        # Generation options
        model_path: str | None = None,
        api_model: str | None = None,
        api_base: str | None = None,
        litellm_provider: str = "hosted_vllm",
        batch_size: int = 16,
        max_tokens: int = 512,
        chat_template_path: str | None = None,
        file_cache: dict[str, str | None] | None = None,
        # SAE options
        sae_dist_path: str | None = None,
        sae_threshold: float | None = None,
        # Prompt options
        system_prompt: str | jinja2.Template | None = None,
        # Logging options
        log_path: Path | str | None = None,
        show_progress: bool = True,
        max_failed: int = 10,
    ) -> EvaluationResult:
        """
        Evaluate a dataset.

        Args:
            dataset: Verifiable dataset to evaluate
            mode: Generation mode ("huggingface", "openai", "file")
            verifier: Verifier name or instance
            verifier_kwargs: Arguments for verifier constructor
            model_path: Path to HF model (for huggingface mode)
            api_model: Model name for API (for openai mode)
            api_base: API base URL (for openai mode)
            litellm_provider: LiteLLM provider (for openai mode)
            batch_size: Batch size for generation
            max_tokens: Maximum tokens to generate
            chat_template_path: Path to custom chat template
            file_cache: Response cache (for file mode)
            sae_dist_path: Path to SAE distribution file
            sae_threshold: SAE pruning threshold
            system_prompt: System prompt to use
            log_path: Path to save evaluation log
            show_progress: Show tqdm progress bar
            max_failed: Maximum failed responses before stopping

        Returns:
            EvaluationResult with records, statistics, and verification results
        """
        # Get verifier
        if isinstance(verifier, str):
            verifier_kwargs = verifier_kwargs or {}
            verifier_obj = get_verifier(verifier, **verifier_kwargs)
        else:
            verifier_obj = verifier

        # Format prompts and extract ground truths
        prompts = [self.format_prompt(e, system_prompt) for e in dataset.entries]
        ground_truths = [self.get_ground_truth(e) for e in dataset.entries]

        # Create generation context
        ctx, hook_ctx = self._create_generation_context(
            mode=mode,
            model_path=model_path,
            api_model=api_model,
            api_base=api_base,
            litellm_provider=litellm_provider,
            batch_size=batch_size,
            max_tokens=max_tokens,
            chat_template_path=chat_template_path,
            file_cache=file_cache,
            sae_dist_path=sae_dist_path,
            sae_threshold=sae_threshold,
        )

        # Generate and collect responses
        responses: list[str | None] = []
        records: list[EvaluationRecord] = []
        failed_count = 0

        with hook_ctx:
            response_iter = self._generate_responses(ctx, prompts)
            if show_progress:
                response_iter = tqdm.tqdm(response_iter, total=len(prompts), desc="Generating")

            for i, response in enumerate(response_iter):
                responses.append(response)

                if response is None:
                    failed_count += 1
                    if failed_count > max_failed:
                        raise RuntimeError(f"Too many failed responses: {failed_count}")

        # Verify all responses
        verification_result = verifier_obj.verify_batch(
            responses=responses,
            ground_truths=ground_truths,
            prompts=prompts,
        )

        # Build records
        for i in range(len(responses)):
            result = verification_result.results[i]
            records.append(
                EvaluationRecord(
                    prompt=prompts[i],
                    response=responses[i],
                    ground_truth=ground_truths[i],
                    score=result.score,
                    is_valid=result.is_valid,
                    metadata=result.metadata,
                )
            )

        # Build statistics
        statistics = DatasetStatistics(
            dataset_name=dataset.info.name,
            source=dataset.info.source,
            size=len(dataset.entries),
            evaluated=True,
            mean_score=verification_result.mean_score,
            accuracy=verification_result.accuracy,
            valid_count=verification_result.valid_count,
            failed_count=failed_count,
            verifier_name=verifier_obj.name,
        )

        result = EvaluationResult(
            records=records,
            statistics=statistics,
            verification_result=verification_result,
        )

        # Save log if requested
        if log_path:
            self._save_log(result, Path(log_path))

        return result

    @beartype
    def _save_log(self, result: EvaluationResult, path: Path) -> None:
        """Save evaluation result to file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "statistics": result.statistics.model_dump(),
            "records": [r.model_dump() for r in result.records],
        }

        if path.suffix == ".jsonl":
            with open(path, "w") as f:
                for record in result.records:
                    f.write(json.dumps(record.model_dump()) + "\n")
        else:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)


# CLI interface
def main():
    """CLI entry point for verifiable evaluation."""
    import click

    @click.command()
    @click.option(
        "--dataset",
        "-d",
        type=str,
        required=True,
        help="Dataset to load (e.g., mmlu:moral_disputes, gsm8k, imdb)",
    )
    @click.option("--limit", "-l", type=int, default=None, help="Limit number of samples")
    @click.option(
        "--mode",
        "-m",
        type=click.Choice(["huggingface", "openai", "file"]),
        default="openai",
        help="Inference mode",
    )
    @click.option("--model", type=str, default=None, help="Model path (HF) or name (API)")
    @click.option("--api-base", type=str, default="http://localhost:8000", help="API base URL")
    @click.option("--provider", type=str, default="hosted_vllm", help="LiteLLM provider")
    @click.option("--verifier", "-v", type=str, default="exact_match", help="Verifier to use")
    @click.option("--batch-size", "-b", type=int, default=16, help="Batch size")
    @click.option("--max-tokens", type=int, default=512, help="Max tokens to generate")
    @click.option("--output", "-o", type=str, default=None, help="Output log path")
    def cli(
        dataset: str,
        limit: int | None,
        mode: str,
        model: str | None,
        api_base: str,
        provider: str,
        verifier: str,
        batch_size: int,
        max_tokens: int,
        output: str | None,
    ):
        """Evaluate a model on a verifiable dataset."""
        # Parse dataset string
        if ":" in dataset:
            dataset_name, subset = dataset.split(":", 1)
        else:
            dataset_name, subset = dataset, None

        # Load dataset
        click.echo(f"Loading dataset: {dataset_name}" + (f" ({subset})" if subset else ""))

        from sae_scoping.datasets import verifiable_datasets as vd

        loader_name = f"load_{dataset_name}"
        if not hasattr(vd, loader_name):
            available = [n.replace("load_", "") for n in dir(vd) if n.startswith("load_")]
            raise click.ClickException(f"Unknown dataset: {dataset_name}. Available: {available}")

        loader = getattr(vd, loader_name)

        # Call loader with appropriate args
        if subset and "subject" in loader.__code__.co_varnames:
            ds = loader(subject=subset, limit=limit)
        elif subset and "subset" in loader.__code__.co_varnames:
            ds = loader(subset=subset, limit=limit)
        else:
            ds = loader(limit=limit)

        click.echo(f"Loaded {len(ds.entries)} entries")

        # Run evaluation
        evaluator = VerifiableEvaluator()

        result = evaluator.evaluate(
            dataset=ds,
            mode=mode,
            verifier=verifier,
            model_path=model if mode == "huggingface" else None,
            api_model=model if mode == "openai" else None,
            api_base=api_base,
            litellm_provider=provider,
            batch_size=batch_size,
            max_tokens=max_tokens,
            log_path=output,
        )

        # Print results
        click.echo("\n" + "=" * 50)
        click.echo(f"Dataset: {result.statistics.dataset_name}")
        click.echo(f"Source: {result.statistics.source}")
        click.echo(f"Size: {result.statistics.size}")
        click.echo(f"Mean Score: {result.statistics.mean_score:.4f}")
        click.echo(f"Accuracy: {result.statistics.accuracy:.2%}")
        click.echo(f"Valid: {result.statistics.valid_count}/{result.statistics.size}")
        click.echo(f"Failed: {result.statistics.failed_count}")
        click.echo(f"Verifier: {result.statistics.verifier_name}")
        click.echo("=" * 50)

    cli()


if __name__ == "__main__":
    main()
