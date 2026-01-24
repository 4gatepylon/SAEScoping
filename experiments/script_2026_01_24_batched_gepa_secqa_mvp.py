"""
GEPA optimization script for SecQA security benchmark.

This script optimizes prompts for SecQA multiple-choice security questions using GEPA.
It combines val, test, and dev splits across both secqa_v1 and secqa_v2 subsets.

Example command to run on google/gemma-2-9b-it here:
```
python script_2026_01_24_batched_gepa_secqa_mvp.py \
    --port 8001 \
    --model-name google/gemma-2-9b-it \
    --basename "align-3.csail.mit.edu" \
    --n-samples 224 \
    --batch-size 16 \
    --max-tokens 1024 \
    --proposer-model openai/gpt-5.2 \
    --budget-mode auto \
    --budget-amount light \
    --output-dir ./outputs_gepa_logs_secqa \
    --yes \
    --clobber
```
Example command to run on our SAE-enhanced model here:
```
python script_2026_01_24_batched_gepa_secqa_mvp.py \
    --port 8000 \
    --model-name "/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b/ultrachat/layer_31_width_16k_canonical_h0.0001_85cac49528/checkpoint-2000" \
    --basename "align-3.csail.mit.edu" \
    --n-samples 224 \
    --batch-size 16 \
    --max-tokens 1024 \
    --proposer-model openai/gpt-5.2 \
    --budget-mode auto \
    --budget-amount light \
    --output-dir ./outputs_gepa_logs_secqa \
    --yes \
    --clobber
```
"""

from __future__ import annotations

import random
import shutil
import hashlib
import dspy
import os
import sys
import json
import re
import click
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from beartype import beartype
from typing import Any
from datasets import load_dataset
import tqdm


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
    """Answer the multiple-choice cybersecurity question. Put your final answer as a single letter (A, B, C, or D) in \\boxed{}."""

    problem = dspy.InputField()
    reasoning = dspy.OutputField(
        prefix="Reasoning: Let's think step by step in order to",
        desc="${reasoning}",
    )
    answer = dspy.OutputField(desc="Final answer as \\boxed{letter}")


class SecQASplit:
    """Dataset loader for SecQA security benchmark."""

    LETTERS = ["A", "B", "C", "D"]

    @beartype
    def __init__(self):
        """Initialize SecQA dataset loader."""
        pass

    @beartype
    def format_question(self, question: str, a: str, b: str, c: str, d: str) -> str:
        """Format SecQA question with choices into a prompt."""
        return f"""\
Question: {question}

A) {a}
B) {b}
C) {c}
D) {d}

Answer with the letter in \\boxed{{}}."""

    @beartype
    def get_dataset_split(
        self,
        train_split_ratio: float = 0.8,
        test_split_ratio: float = 0.1,
        val_split_ratio: float = 0.1,
        n_samples: int = 100,
    ) -> dict[str, list[dspy.Example]]:
        """Load and split SecQA dataset, combining all splits from both subsets."""
        if abs(train_split_ratio + test_split_ratio + val_split_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {train_split_ratio + test_split_ratio + val_split_ratio}")

        # Load all splits from both subsets
        all_data = []
        for subset in ["secqa_v1", "secqa_v2"]:
            for split in ["val", "test", "dev"]:
                try:
                    dataset = load_dataset("zefang-liu/secqa", subset, split=split)
                    all_data.extend(list(dataset))
                    print(f"Loaded {len(dataset)} samples from {subset}/{split}")
                except Exception as e:
                    # print(f"Warning: Could not load {subset}/{split}: {e}")
                    raise ValueError(f"Could not load {subset}/{split}: {e}")

        print(f"Total samples loaded: {len(all_data)}")

        # Shuffle with fixed seed for reproducibility
        random.Random(0).shuffle(all_data)

        # Limit samples
        all_data = all_data[:n_samples]
        if len(all_data) < n_samples:
            # print(f"Warning: Only {len(all_data)} samples available, requested {n_samples}")
            raise ValueError(f"Only {len(all_data)} samples available, requested {n_samples}")

        # Convert to DSPy Examples
        examples = []
        for item in tqdm.tqdm(all_data, desc="Converting SecQA to DSPy Examples"):
            answer_letter = item["Answer"]
            answer_idx = self.LETTERS.index(answer_letter)
            choices = [item["A"], item["B"], item["C"], item["D"]]
            correct_choice_text = choices[answer_idx]
            # Get explanation if available (for proposer model feedback)
            explanation = item.get("Explanation", "")

            formatted_problem = self.format_question(
                item["Question"],
                item["A"],
                item["B"],
                item["C"],
                item["D"],
            )

            example = dspy.Example(
                {
                    "problem": formatted_problem,
                    "answer_idx": answer_idx,
                    "answer_letter": answer_letter,
                    "answer_text": correct_choice_text,
                    "choices": choices,
                    "original_question": item["Question"],
                    "explanation": explanation,
                }
            ).with_inputs("problem")
            examples.append(example)

        # Split into train/val/test
        tot_num = len(examples)
        train_end = int(train_split_ratio * tot_num)
        val_end = int((train_split_ratio + val_split_ratio) * tot_num)

        train_set = examples[:train_end]
        val_set = examples[train_end:val_end]
        test_set = examples[val_end:]

        for v, vname in zip([train_set, val_set, test_set], ["train", "val", "test"]):
            if len(v) == 0:
                raise ValueError(f"No examples in {vname} set")

        return {
            "train": train_set,
            "val": val_set,
            "test": test_set,
        }


class SecQAMetricWrapper:
    """Metric wrapper for SecQA multiple-choice evaluation using boxed answers."""

    # Match \boxed{A}, \boxed{B}, \boxed{C}, \boxed{D} (case insensitive)
    BOXED_PATTERN = re.compile(r"\\boxed\{([A-Da-d])\}")
    LETTERS = ["A", "B", "C", "D"]

    @beartype
    def extract_boxed_answer(self, text: str) -> str | None:
        """Extract the last \\boxed{letter} content from text."""
        matches = self.BOXED_PATTERN.findall(text)
        if not matches:
            return None
        return matches[-1].strip().upper()

    @beartype
    def metric(
        self,
        gold: dspy.Example | None,
        pred: dspy.Example,
        trace: Any | None = None,
    ) -> int:
        """Evaluate prediction against gold answer. Returns 1 if correct, 0 otherwise."""
        if gold is None:
            raise ValueError("Gold example is None")

        pred_answer_raw = str(pred.answer) if hasattr(pred, "answer") else ""
        pred_answer = self.extract_boxed_answer(pred_answer_raw)

        if pred_answer is None:
            return 0

        return int(pred_answer.lower() == gold.answer_letter.lower())

    @beartype
    def metric_with_feedback(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any | None = None,
        pred_name: Any | None = None,
        pred_trace: Any | None = None,
    ) -> dspy.Prediction:
        """Evaluate prediction with detailed feedback for GEPA optimization, including ground-truth explanation."""
        pred_answer_raw = str(prediction.answer) if hasattr(prediction, "answer") else ""
        pred_answer = self.extract_boxed_answer(pred_answer_raw)

        if pred_answer is None:
            feedback_text = (
                f"Your answer must be formatted as \\boxed{{A}}, \\boxed{{B}}, \\boxed{{C}}, or \\boxed{{D}}. " +
                f"You responded with '{pred_answer_raw[:200]}...', which doesn't contain a properly formatted boxed answer. " +
                f"The correct answer is ({example.answer_letter}) {example.answer_text}." +
            )
            if example.explanation:
                feedback_text += f"\n\nExplanation: {example.explanation}"
            return dspy.Prediction(score=0, feedback=feedback_text)

        is_correct = pred_answer.lower() == example.answer_letter.lower()

        if is_correct:
            feedback_text = f"Correct! The answer is ({example.answer_letter}) {example.answer_text}."
            if example.explanation:
                feedback_text += f"\n\nExplanation: {example.explanation}"
        else:
            feedback_text = f"Incorrect. You answered \\boxed{{{pred_answer}}}, but the correct answer is " f"({example.answer_letter}) {example.answer_text}."
            if example.explanation:
                feedback_text += f"\n\nExplanation: {example.explanation}"
            feedback_text += "\n\nThis is a cybersecurity question. " "Think about what security concepts, attack vectors, or defensive measures you might have missed."

        return dspy.Prediction(score=int(is_correct), feedback=feedback_text)


def get_budget_kwargs(budget_mode: str, budget_amount: str) -> dict:
    """Build budget kwargs for GEPA based on mode and amount."""
    if budget_mode == "auto":
        assert budget_amount in (
            "light",
            "medium",
            "heavy",
        ), f"budget_amount must be 'light', 'medium', or 'heavy' for auto mode, got: {budget_amount}"
        return {"auto": budget_amount}
    budget_int = int(budget_amount)
    assert budget_int > 0, f"budget_amount must be positive integer for {budget_mode} mode, got: {budget_amount}"
    return {"max_metric_calls": budget_int} if budget_mode == "metric" else {"max_full_evals": budget_int}


def save_lm_history(lm: dspy.LM, output_dir: Path, filename: str, port: int) -> Path:
    """Save LM history to a JSON file for debugging/comparison."""
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
                try:
                    history_data.append(entry.to_dict())
                except Exception as e:
                    history_data.append({"error": str(e), "raw": str(entry)})
        except Exception as e:
            history_data.append({"error": str(e), "raw": str(entry)})

    log_data = {
        "port": port,
        "timestamp": timestamp,
        "num_calls": len(history_data),
        "history": history_data,
    }

    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=2, default=str)

    print(f"Saved LM history ({len(history_data)} calls) to: {filepath}")
    return filepath


@click.command()
@click.option("--adaptor", "-a", type=click.Choice(["chat", "json"]), default="chat")
@click.option("--max-tokens", "-m", type=int, default=512)
@click.option("--batch-size", "-b", type=int, default=16)
@click.option("--port", "-p", type=int, default=8000)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./outputs_gepa_logs_secqa",
    help="Directory to save LM history logs",
)
@click.option(
    "--model-name",
    "-mn",
    type=str,
    default="google/gemma-2-9b-it",
    help="Model name to use",
)
@click.option("--basename", "-bn", type=str, default="localhost", help="Hostname to use")
@click.option("--n-samples", "-ns", type=int, default=100, help="Number of samples to use")
@click.option(
    "--clobber",
    "-c",
    is_flag=True,
    default=False,
    help="Clobber existing output directory",
)
@click.option("--yes", "-y", is_flag=True, default=False, help="Answer yes to all prompts")
@click.option(
    "--wandb-project-name",
    "-wp",
    type=str,
    default="scopebench-gepa-secqa-mvp",
    help="Wandb project name",
)
@click.option("--wandb-run-name", "-wn", type=str, default=None, help="Wandb run name")
@click.option(
    "--proposer-model",
    "-pm",
    type=str,
    default="openrouter/qwen/qwen3-next-80b-a3b-thinking",
    help="Prompt-proposer model. Use 'openrouter/...' for OpenRouter or 'openai/...' for OpenAI",
)
@click.option(
    "--proposer-max-tokens",
    "-pmt",
    type=int,
    default=65536,
    help="Max tokens for the proposer model",
)
@click.option(
    "--budget-mode",
    "-bm",
    type=click.Choice(["auto", "metric", "evals"]),
    default="auto",
    help="Budget mode: auto (light/medium/heavy), metric (max_metric_calls), or evals (max_full_evals)",
)
@click.option(
    "--budget-amount",
    "-ba",
    type=str,
    default="light",
    help="Budget amount: 'light'/'medium'/'heavy' for auto mode, or positive integer for metric/evals modes",
)
@click.option(
    "--train-split-ratio",
    "-tsr",
    type=float,
    default=0.8,
    help="Ratio of data to use for training (default: 0.8)",
)
@click.option(
    "--val-split-ratio",
    "-vsr",
    type=float,
    default=0.1,
    help="Ratio of data to use for validation (default: 0.1)",
)
@click.option(
    "--test-split-ratio",
    "-tesr",
    type=float,
    default=0.1,
    help="Ratio of data to use for testing (default: 0.1)",
)
@beartype
def main(
    adaptor: str,
    max_tokens: int,
    batch_size: int,
    port: int,
    output_dir: str,
    model_name: str,
    basename: str,
    n_samples: int,
    clobber: bool,
    yes: bool,
    wandb_project_name: str,
    wandb_run_name: str | None,
    proposer_model: str,
    proposer_max_tokens: int,
    budget_mode: str,
    budget_amount: str,
    train_split_ratio: float,
    val_split_ratio: float,
    test_split_ratio: float,
) -> None:
    import litellm

    litellm.cache = None

    model_name_hash = hashlib.sha256(model_name.encode()).hexdigest()
    _model_name = model_name.replace("/", "_")
    model_name_or_model_name_hash = model_name_hash if len(_model_name) > len(model_name_hash) else _model_name
    output_path = Path(output_dir) / model_name_or_model_name_hash / proposer_model.replace("/", "_") / f"n{n_samples}_m{max_tokens}"

    if output_path.exists() and clobber:
        shutil.rmtree(output_path)
    assert not output_path.exists(), f"Output path already exists: {output_path}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save arguments
    kwargs_file = output_path / "kwargs.json"
    kwargs_file.write_text(
        json.dumps(
            {
                "adaptor": adaptor,
                "max_tokens": max_tokens,
                "batch_size": batch_size,
                "port": port,
                "output_dir": output_dir,
                "model_name": model_name,
                "basename": basename,
                "n_samples": n_samples,
                "proposer_model": proposer_model,
                "proposer_max_tokens": proposer_max_tokens,
                "budget_mode": budget_mode,
                "budget_amount": budget_amount,
                "train_split_ratio": train_split_ratio,
                "val_split_ratio": val_split_ratio,
                "test_split_ratio": test_split_ratio,
            },
            indent=2,
        )
    )
    print(f"Logging traces to: {output_path.absolute()}")

    print("=" * 100)
    vllm_llm = dspy.LM(
        f"hosted_vllm/{model_name}",
        api_key="dummy",
        api_base=f"http://{basename}:{port}/v1",
        max_tokens=max_tokens,
        temperature=0.1,
        cache=False,
    )

    # Configure reflection_lm based on proposer_model
    is_openai = proposer_model.startswith("openai/")
    proposer_api_key = os.getenv("OPENAI_API_KEY") if is_openai else (os.getenv("OPENROUTER_API_KEY") if proposer_model.startswith("openrouter/") else None)
    assert proposer_api_key is not None, f"No API key found for proposer model: {proposer_model}. Set OPENAI_API_KEY or OPENROUTER_API_KEY."
    reflection_lm = dspy.LM(
        proposer_model,
        api_key=proposer_api_key,
        api_base=None if is_openai else "https://openrouter.ai/api/v1",
        max_tokens=proposer_max_tokens,
        temperature=0.9,
        cache=False,
    )

    print("=" * 100)
    print("Getting dataset splits")
    secqa_split = SecQASplit()
    datasets = secqa_split.get_dataset_split(
        train_split_ratio=train_split_ratio,
        test_split_ratio=test_split_ratio,
        val_split_ratio=val_split_ratio,
        n_samples=n_samples,
    )
    print("=" * 100)
    print("Got these dataset sizes:")
    print(f"train: {len(datasets['train'])}")
    print(f"val: {len(datasets['val'])}")
    print(f"test: {len(datasets['test'])}")
    if not yes:
        click.confirm("Continue?", abort=True)

    metric_wrapper = SecQAMetricWrapper()

    print("=" * 100)
    print("Generating program and configuring DSPY")
    adaptor_obj = dspy.ChatAdapter() if adaptor == "chat" else dspy.JSONAdapter()
    print(f"Using adaptor: {type(adaptor_obj)}")
    dspy.configure(lm=vllm_llm, adapter=adaptor_obj, cache=False)
    program = dspy.Predict(GenerateResponseWithReasoning)

    save_prompt_file_init = output_path / "initial_program_instructions.txt"
    assert not save_prompt_file_init.exists(), f"Save file init already exists: {save_prompt_file_init}"
    save_prompt_file_init.write_text(program.signature.instructions)

    print("=" * 100)
    print("Evaluating program on test set")
    evaluate = dspy.Evaluate(
        devset=datasets["test"],
        metric=metric_wrapper.metric,
        num_threads=batch_size,
        display_table=True,
        display_progress=True,
        provide_traceback=True,
        max_errors=10_000,
    )
    initial_eval_log = output_path / "initial_eval_output.txt"
    with tee_stdout_to_file(initial_eval_log):
        initial_score = evaluate(program)
    print(f"Initial evaluation score: {initial_score}")

    # Save LM history after initial evaluation
    save_lm_history(vllm_llm, output_path, "initial_eval", port)

    print("=" * 100)
    print("Optimizing program with GEPA")
    gepa_log_dir = output_path / "dspy_gepa_logdir"
    gepa_log_dir.mkdir(parents=True, exist_ok=True)
    assert "WANDB_API_KEY" in os.environ, "WANDB_API_KEY is not set"
    if wandb_run_name is None:
        wandb_run_name = f"{model_name_or_model_name_hash}_n{n_samples}_b{batch_size}_m{max_tokens}"
    os.environ["WANDB_PROJECT"] = wandb_project_name
    os.environ["WANDB_RUN_NAME"] = wandb_run_name
    budget_kwargs = get_budget_kwargs(budget_mode, budget_amount)
    optimizer = dspy.GEPA(
        metric=metric_wrapper.metric_with_feedback,
        **budget_kwargs,
        num_threads=batch_size,
        reflection_minibatch_size=16,
        track_best_outputs=True,
        add_format_failure_as_feedback=True,
        reflection_lm=reflection_lm,
        log_dir=gepa_log_dir.as_posix(),
        track_stats=True,
        gepa_kwargs={
            "use_cloudpickle": True,
        },
        use_wandb=True,
        wandb_api_key=os.getenv("WANDB_API_KEY"),
    )
    optimized_program = optimizer.compile(
        program,
        trainset=datasets["train"],
        valset=datasets["val"],
    )
    print("=" * 100)
    print("PROGRAM INSTRUCTIONS\n```")
    print(optimized_program.signature.instructions)
    save_prompt_file = output_path / "optimized_program_instructions.txt"
    assert not save_prompt_file.exists(), f"Save prompt file already exists: {save_prompt_file}"
    save_prompt_file.write_text(optimized_program.signature.instructions)
    print("```")
    print("=" * 100)
    print("Evaluating optimized program on test set")
    final_eval_log = output_path / "final_eval_output.txt"
    with tee_stdout_to_file(final_eval_log):
        final_score = evaluate(optimized_program)
    print(f"Final evaluation score: {final_score}")

    # Save LM history after optimization and final evaluation
    save_lm_history(vllm_llm, output_path, "after_optimization", port)

    # Also save reflection LM history if it was used
    if reflection_lm.history:
        save_lm_history(reflection_lm, output_path, "reflection_lm", port)

    # Save summary with scores
    summary_file = output_path / "evaluation_summary.json"
    summary_file.write_text(
        json.dumps(
            {
                "initial_score": initial_score,
                "final_score": final_score,
                "improvement": final_score - initial_score,
                "n_samples": n_samples,
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
            },
            indent=2,
        )
    )
    print(f"Saved evaluation summary to: {summary_file}")

    print("=" * 100)
    print(f"All logs saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
