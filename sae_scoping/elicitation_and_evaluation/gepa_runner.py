"""
GEPA evaluation runner supporting all verifiable datasets.

Supports MCQ datasets (MMLU, SecQA, WMDP, CyberMetric), golden answer datasets
(GSM8K, NuminaMath, MATH-500, IMDB, Camel AI), and executable test datasets (APPS, CodeContests).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import click
import dspy
import tqdm
from beartype import beartype

from sae_scoping.datasets.verifiable_datasets import (
    MultipleChoiceDataset,
    GoldenAnswerDataset,
    MultipleChoiceEntry,
    GoldenAnswerEntry,
    load_mmlu,
    load_secqa,
    load_wmdp_cyber,
    load_cybermetric,
    load_gsm8k,
    load_numinamath,
    load_imdb,
    load_camel_ai_biology,
    load_camel_ai_chemistry,
    load_camel_ai_physics,
    load_camel_ai_math,
    load_math500,
)
from sae_scoping.elicitation_and_evaluation.metrics import (
    EvalItem,
    get_metric,
)

# Dataset loader registry
DATASET_LOADERS = {
    # MCQ datasets
    "mmlu": load_mmlu,
    "secqa": load_secqa,
    "wmdp_cyber": load_wmdp_cyber,
    "cybermetric": load_cybermetric,
    # Golden answer datasets
    "gsm8k": load_gsm8k,
    "numinamath": load_numinamath,
    "imdb": load_imdb,
    "math500": load_math500,
    "camel_biology": load_camel_ai_biology,
    "camel_chemistry": load_camel_ai_chemistry,
    "camel_physics": load_camel_ai_physics,
    "camel_math": load_camel_ai_math,
}

DATASET_CHOICES = list(DATASET_LOADERS.keys())


class TeeWriter:
    """Write to both a file and the original stdout."""

    def __init__(self, file, original_stdout):
        self.file = file
        self.original_stdout = original_stdout

    def write(self, data):
        self.original_stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.original_stdout.flush()
        self.file.flush()


@contextmanager
def tee_stdout_to_file(filepath: Path):
    """Context manager that writes stdout to both console and file."""
    original_stdout = sys.stdout
    with open(filepath, "w") as f:
        sys.stdout = TeeWriter(f, original_stdout)
        try:
            yield
        finally:
            sys.stdout = original_stdout


class GenerateResponseWithReasoning(dspy.Signature):
    """Answer the question and provide the answer in the correct format."""

    problem = dspy.InputField()
    reasoning = dspy.OutputField(prefix="Reasoning: Let's think step by step in order to", desc="${reasoning}")
    answer = dspy.OutputField()


# =============================================================================
# Prompt formatting
# =============================================================================


@beartype
def format_mcq_prompt(entry: MultipleChoiceEntry) -> str:
    """Format MCQ entry into a prompt string."""
    A_ord = ord("A")
    options_str = "\n".join(f"({chr(A_ord + i)}) {c}" for i, c in enumerate(entry.choices))
    return f"""\
Question: {entry.question}

{options_str}

Please select one option (A, B, C, or D). Put your final answer in \\boxed{{}}.
For example: \\boxed{{A}} if you think the first option is correct."""


@beartype
def format_golden_answer_prompt(entry: GoldenAnswerEntry) -> str:
    """Format golden answer entry into a prompt string."""
    return f"""\
Problem: {entry.question}

Solve this problem step by step. Put your final answer in \\boxed{{}}.
For example: \\boxed{{42}} if the answer is 42."""


# =============================================================================
# Metric wrappers for DSPY
# =============================================================================


class MetricWrapper:
    """Wraps our metrics for use with DSPY."""

    def __init__(self, dataset_type: Literal["mcq", "golden_answer"], metrics: list[str] | None = None):
        self.dataset_type = dataset_type
        # Default metrics based on dataset type
        if metrics is None:
            metrics = ["boxed", "mcq_letter"] if dataset_type == "mcq" else ["boxed", "exact_match"]
        self.metrics = [get_metric(m) for m in metrics]

    @beartype
    def metric(self, gold: dspy.Example | None, pred: dspy.Example, trace: Any | None = None) -> int:
        """Evaluate prediction against gold answer. Returns 1 if correct, 0 otherwise."""
        if gold is None:
            raise ValueError("Gold example is None")

        pred_answer = str(pred.answer) if hasattr(pred, "answer") else ""
        golden = gold.golden if hasattr(gold, "golden") else ""

        item = EvalItem(question=gold.problem, response=pred_answer, golden=golden)

        # Try each metric, return 1 if any says correct
        for metric in self.metrics:
            result = metric.evaluate_single(item)
            if result.score == 1.0:
                return 1
        return 0

    @beartype
    def metric_with_feedback(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any | None = None,
        **kwargs,
    ) -> dspy.Prediction:
        """Evaluate with detailed feedback for GEPA optimization."""
        pred_answer = str(prediction.answer) if hasattr(prediction, "answer") else ""
        golden = example.golden if hasattr(example, "golden") else ""

        item = EvalItem(question=example.problem, response=pred_answer, golden=golden)

        # Try each metric
        for metric in self.metrics:
            result = metric.evaluate_single(item)
            if result.score == 1.0:
                return dspy.Prediction(score=1, feedback=f"Correct! Answer: {golden}")

        # All failed - provide feedback
        feedback = f"Incorrect. Your answer: {result.extracted or pred_answer}. Expected: {golden}"
        return dspy.Prediction(score=0, feedback=feedback)


# =============================================================================
# Dataset loading and conversion
# =============================================================================


@beartype
def load_and_convert_dataset(
    dataset_name: str,
    limit: int | None = None,
    seed: int = 42,
    subject: str | None = None,
    split: str = "test",
) -> tuple[list[dspy.Example], Literal["mcq", "golden_answer"], str]:
    """
    Load dataset and convert to DSPY examples.

    Returns: (examples, dataset_type, dataset_display_name)
    """
    loader = DATASET_LOADERS.get(dataset_name)
    if loader is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {DATASET_CHOICES}")

    # Load with appropriate kwargs based on loader signature
    loader_kwargs: dict[str, Any] = {"limit": limit, "seed": seed}
    if dataset_name == "mmlu" and subject:
        loader_kwargs["subject"] = subject
    if dataset_name in ("mmlu", "gsm8k", "numinamath", "math500"):
        loader_kwargs["split"] = split

    dataset = loader(**loader_kwargs)

    examples: list[dspy.Example] = []

    if isinstance(dataset, MultipleChoiceDataset):
        dataset_type: Literal["mcq", "golden_answer"] = "mcq"
        for entry in tqdm.tqdm(dataset.entries, desc=f"Converting {dataset_name}"):
            ex = dspy.Example(
                {
                    "problem": format_mcq_prompt(entry),
                    "golden": entry.answer_letter,
                    "answer_index": entry.answer_index,
                    "choices": entry.choices,
                }
            ).with_inputs("problem")
            examples.append(ex)
    elif isinstance(dataset, GoldenAnswerDataset):
        dataset_type = "golden_answer"
        for entry in tqdm.tqdm(dataset.entries, desc=f"Converting {dataset_name}"):
            ex = dspy.Example(
                {
                    "problem": format_golden_answer_prompt(entry),
                    "golden": entry.golden_answer,
                }
            ).with_inputs("problem")
            examples.append(ex)
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")

    display_name = f"{dataset.info.source}"
    if dataset.info.subset:
        display_name += f" ({dataset.info.subset})"

    return examples, dataset_type, display_name


@beartype
def split_dataset(
    examples: list[dspy.Example],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, list[dspy.Example]]:
    """Split examples into train/val/test sets."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    n = len(examples)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    return {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:],
    }


# =============================================================================
# Utilities
# =============================================================================


def get_budget_kwargs(budget_mode: str, budget_amount: str) -> dict:
    """Build budget kwargs for GEPA."""
    if budget_mode == "auto":
        assert budget_amount in ("light", "medium", "heavy")
        return {"auto": budget_amount}
    budget_int = int(budget_amount)
    return {"max_metric_calls": budget_int} if budget_mode == "metric" else {"max_full_evals": budget_int}


def save_lm_history(lm: dspy.LM, output_dir: Path, filename: str, port: int) -> Path:
    """Save LM history to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"{filename}_port{port}_{timestamp}.json"

    history_data = []
    for entry in lm.history:
        try:
            if hasattr(entry, "__dict__"):
                history_data.append(entry.__dict__)
            elif isinstance(entry, dict):
                history_data.append(entry)
            else:
                history_data.append({"raw": str(entry)})
        except Exception as e:
            history_data.append({"error": str(e)})

    with open(filepath, "w") as f:
        json.dump({"port": port, "timestamp": timestamp, "history": history_data}, f, indent=2, default=str)

    print(f"Saved LM history ({len(history_data)} calls) to: {filepath}")
    return filepath


# =============================================================================
# Main CLI
# =============================================================================


@click.command()
@click.option("--dataset", "-d", type=click.Choice(DATASET_CHOICES), required=True, help="Dataset to evaluate on")
@click.option("--subject", "-s", type=str, default=None, help="Subject filter (for MMLU)")
@click.option("--n-samples", "-n", type=int, default=100, help="Number of samples")
@click.option("--port", "-p", type=int, default=8000, help="vLLM server port")
@click.option("--model-name", "-m", type=str, default="google/gemma-2-9b-it", help="Model name")
@click.option("--basename", "-b", type=str, default="localhost", help="Server hostname")
@click.option("--max-tokens", "-mt", type=int, default=512, help="Max tokens for generation")
@click.option("--batch-size", "-bs", type=int, default=16, help="Batch size")
@click.option("--output-dir", "-o", type=click.Path(), default="./outputs_gepa", help="Output directory")
@click.option("--proposer-model", "-pm", type=str, default="openrouter/qwen/qwen3-next-80b-a3b-thinking")
@click.option("--proposer-max-tokens", "-pmt", type=int, default=65536)
@click.option("--budget-mode", "-bm", type=click.Choice(["auto", "metric", "evals"]), default="auto")
@click.option("--budget-amount", "-ba", type=str, default="light")
@click.option("--train-ratio", type=float, default=0.8)
@click.option("--val-ratio", type=float, default=0.1)
@click.option("--test-ratio", type=float, default=0.1)
@click.option("--wandb-project", "-wp", type=str, default="scopebench-gepa")
@click.option("--wandb-run-name", "-wn", type=str, default=None)
@click.option("--clobber", "-c", is_flag=True, help="Overwrite existing output")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--adaptor", "-a", type=click.Choice(["chat", "json"]), default="chat")
@beartype
def main(
    dataset: str,
    subject: str | None,
    n_samples: int,
    port: int,
    model_name: str,
    basename: str,
    max_tokens: int,
    batch_size: int,
    output_dir: str,
    proposer_model: str,
    proposer_max_tokens: int,
    budget_mode: str,
    budget_amount: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    wandb_project: str,
    wandb_run_name: str | None,
    clobber: bool,
    yes: bool,
    debug: bool,
    adaptor: str,
) -> None:
    """
    Run GEPA optimization on a verifiable dataset.

    Example:
        python gepa_runner.py -d mmlu -s moral_disputes -n 200 -p 8000

    Supported datasets:
        MCQ: mmlu, secqa, wmdp_cyber, cybermetric
        Golden Answer: gsm8k, numinamath, math500, imdb, camel_biology, camel_chemistry, camel_physics, camel_math
    """
    import litellm

    litellm.cache = None
    if debug:
        litellm._turn_on_debug()

    # Setup output path
    model_hash = hashlib.sha256(model_name.encode()).hexdigest()[:12]
    subject_suffix = f"_{subject}" if subject else ""
    output_path = Path(output_dir) / dataset / f"{model_hash}{subject_suffix}" / f"n{n_samples}_m{max_tokens}"

    if output_path.exists():
        if clobber:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"Output exists: {output_path}. Use --clobber to overwrite.")
    output_path.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        "dataset": dataset,
        "subject": subject,
        "n_samples": n_samples,
        "model_name": model_name,
        "port": port,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "proposer_model": proposer_model,
        "budget_mode": budget_mode,
        "budget_amount": budget_amount,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    (output_path / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Output: {output_path.absolute()}")

    # Setup LLMs
    vllm_lm = dspy.LM(
        f"hosted_vllm/{model_name}",
        api_key="dummy",
        api_base=f"http://{basename}:{port}/v1",
        max_tokens=max_tokens,
        temperature=1.0,
        cache=False,
    )

    is_openai = proposer_model.startswith("openai/")
    proposer_api_key = os.getenv("OPENAI_API_KEY") if is_openai else os.getenv("OPENROUTER_API_KEY")
    assert proposer_api_key, f"API key not found for: {proposer_model}"

    reflection_lm = dspy.LM(
        proposer_model,
        api_key=proposer_api_key,
        api_base=None if is_openai else "https://openrouter.ai/api/v1",
        max_tokens=proposer_max_tokens,
        temperature=1.0,
        cache=False,
    )

    # Load dataset
    print("=" * 80)
    print(f"Loading dataset: {dataset}")
    examples, dataset_type, display_name = load_and_convert_dataset(dataset, limit=n_samples, seed=42, subject=subject)
    splits = split_dataset(examples, train_ratio, val_ratio, test_ratio)
    print(f"Dataset: {display_name}")
    print(f"Type: {dataset_type}")
    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    if not yes:
        click.confirm("Continue?", abort=True)

    # Setup metric and program
    metric_wrapper = MetricWrapper(dataset_type)
    adaptor_obj = dspy.ChatAdapter() if adaptor == "chat" else dspy.JSONAdapter()
    dspy.configure(lm=vllm_lm, adapter=adaptor_obj, cache=False)
    program = dspy.Predict(GenerateResponseWithReasoning)

    (output_path / "initial_instructions.txt").write_text(program.signature.instructions)

    # Initial evaluation
    print("=" * 80)
    print("Initial evaluation on test set")
    evaluate = dspy.Evaluate(
        devset=splits["test"],
        metric=metric_wrapper.metric,
        num_threads=batch_size,
        display_table=True,
        display_progress=True,
        provide_traceback=True,
        max_errors=10_000,
    )

    with tee_stdout_to_file(output_path / "initial_eval.txt"):
        initial_score = evaluate(program)
    print(f"Initial score: {initial_score}")
    save_lm_history(vllm_lm, output_path, "initial_eval", port)

    # GEPA optimization
    print("=" * 80)
    print("Running GEPA optimization")
    gepa_log_dir = output_path / "gepa_logs"
    gepa_log_dir.mkdir(parents=True, exist_ok=True)

    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY not set"
    if wandb_run_name is None:
        wandb_run_name = f"{dataset}{subject_suffix}_n{n_samples}"
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_RUN_NAME"] = wandb_run_name

    optimizer = dspy.GEPA(
        metric=metric_wrapper.metric_with_feedback,
        **get_budget_kwargs(budget_mode, budget_amount),
        num_threads=batch_size,
        reflection_minibatch_size=16,
        track_best_outputs=True,
        add_format_failure_as_feedback=True,
        reflection_lm=reflection_lm,
        log_dir=gepa_log_dir.as_posix(),
        track_stats=True,
        gepa_kwargs={"use_cloudpickle": True},
        use_wandb=True,
        wandb_api_key=os.getenv("WANDB_API_KEY"),
    )

    optimized_program = optimizer.compile(program, trainset=splits["train"], valset=splits["val"])

    (output_path / "optimized_instructions.txt").write_text(optimized_program.signature.instructions)
    print("Optimized instructions saved")

    # Final evaluation
    print("=" * 80)
    print("Final evaluation on test set")
    with tee_stdout_to_file(output_path / "final_eval.txt"):
        final_score = evaluate(optimized_program)
    print(f"Final score: {final_score}")

    save_lm_history(vllm_lm, output_path, "final_eval", port)
    if reflection_lm.history:
        save_lm_history(reflection_lm, output_path, "reflection", port)

    # Summary
    summary = {
        "initial_score": initial_score,
        "final_score": final_score,
        "improvement": final_score - initial_score,
        "dataset": dataset,
        "display_name": display_name,
        "n_samples": n_samples,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2))

    print("=" * 80)
    print(f"Initial: {initial_score:.2%}, Final: {final_score:.2%}, Δ: {final_score - initial_score:+.2%}")
    print(f"Results saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
