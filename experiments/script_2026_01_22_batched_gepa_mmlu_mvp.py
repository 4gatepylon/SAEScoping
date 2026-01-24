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

"""
MMLU version of GEPA optimization script.

This script optimizes prompts for MMLU multiple-choice questions using GEPA.
It filters by subject and uses a boxed answer format for evaluation.

Example command:
```
python script_2026_01_22_batched_gepa_mmlu_mvp.py \
    --port 8001 \
    --model-name google/gemma-2-9b-it \
    --basename "align-3.csail.mit.edu" \
    --n-samples 320 \
    --subject moral_disputes \
    --batch-size 16 \
    --max-tokens 512 \
    --proposer-model openrouter/qwen/qwen3-next-80b-a3b-thinking \
    --budget-mode auto \
    --budget-amount light \
    --output-dir ./outputs_gepa_logs_mmlu \
    --yes \
    --clobber
```
OR for the SAE-enhanced model:
```
python script_2026_01_22_batched_gepa_mmlu_mvp.py \
    --port 8000 \
    --model-name "/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b/ultrachat/layer_31_width_16k_canonical_h0.0001_85cac49528/checkpoint-2000" \
    --basename "align-3.csail.mit.edu" \
    --n-samples 320 \
    --subject moral_disputes \
    --batch-size 16 \
    --max-tokens 512 \
    --proposer-model openrouter/qwen/qwen3-next-80b-a3b-thinking \
    --budget-mode auto \
    --budget-amount light \
    --output-dir ./outputs_gepa_logs_mmlu \
    --yes \
    --clobber
```
"""


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
    """Answer the multiple-choice question and provide the answer in the correct format."""

    problem = dspy.InputField()
    reasoning = dspy.OutputField(
        prefix="Reasoning: Let's think step by step in order to",
        desc="${reasoning}",
    )
    answer = dspy.OutputField()


class MMLUSplit:
    LETTERS = ["A", "B", "C", "D"]

    @beartype
    def __init__(self, subject: str | None = None):
        """
        Initialize MMLU dataset loader.

        Args:
            subject: MMLU subject to filter by (e.g., "anatomy", "astronomy", etc.)
                     If None, uses all subjects.
        """
        # TODO(Adriano): in the future, but not yet, add support for multiple subjects
        self.subject = subject

    @staticmethod
    def _ordinal(n: int) -> str:
        """Convert 0-indexed number to ordinal word/string."""
        ordinals = ["first", "second", "third", "fourth"]
        if n < len(ordinals):
            return ordinals[n]
        # For n >= 4, use (n+1)th format (since n is 0-indexed)
        return f"{n + 1}th"

    @beartype
    def format_question(self, question: str, subject: str, choices: list[str]) -> str:
        """Format MMLU question with choices into a prompt."""
        A_ord = ord("A")
        assert A_ord == 65, "A should be 65 because ASCII"
        options_str = "\n".join(
            # A, B, C, D
            [f"({chr(A_ord + i)}) {choice}" for i, choice in enumerate(choices)]
        )
        letters = [chr(A_ord + i) for i in range(len(choices))]

        # Pick a random index for the example
        rand_idx = random.Random(0).randint(0, len(choices) - 1)
        example_letter = letters[rand_idx]
        example_choice = choices[rand_idx]
        ordinal_word = self._ordinal(rand_idx)

        return f"""\
Here is a question about {subject.replace("_", " ")}.

{question}

Please answer by selecting one of these options:
{options_str}

Reason through the problem as much as you like, but please format your final answer in the end as \\""" + "boxed{"+"{..."+"}"+"""} \
where the content can be:
- The letter (A, B, C, or D)
- The index (0, 1, 2, or 3)
- The exact text of your chosen option
Case does not matter. For example, \\boxed{{{example_letter}}}, \\boxed{{{example_letter.lower()}}}, \\boxed{{{rand_idx}}}, \\boxed{{{example_choice}}} will all count as 'the {ordinal_word} choice'."""

    @beartype
    def get_dataset_split(
        self,
        train_split_ratio: float = 0.8,
        test_split_ratio: float = 0.1,
        val_split_ratio: float = 0.1,
        n_samples: int = 100,
    ) -> dict[str, list[dspy.Example]]:
        """Load and split MMLU dataset."""
        if train_split_ratio + test_split_ratio + val_split_ratio != 1.0:
            raise ValueError(f"Split ratios must sum to 1.0, got {train_split_ratio + test_split_ratio + val_split_ratio}")

        # Load MMLU dataset
        # cais/mmlu has splits: auxiliary_train, dev, validation, test
        if self.subject:
            dataset = load_dataset("cais/mmlu", self.subject, trust_remote_code=True)
        else:
            dataset = load_dataset("cais/mmlu", "all", trust_remote_code=True)

        # Use test split as our main data source (it's the largest)
        # Fall back to validation if test is too small
        if "test" in dataset and len(dataset["test"]) >= n_samples:
            data_split = dataset["test"]
        elif "validation" in dataset:
            data_split = dataset["validation"]
        else:
            # Combine available splits
            available_data = []
            for split_name in ["test", "validation", "dev"]:
                if split_name in dataset:
                    available_data.extend(list(dataset[split_name]))
            data_split = available_data

        # Convert to list and shuffle
        data_list = list(data_split)
        random.Random(0).shuffle(data_list)

        # Limit samples
        data_list = data_list[:n_samples]
        assert len(data_list) == n_samples, f"Expected {n_samples} samples, got {len(data_list)}"

        # Convert to DSPy Examples
        examples = []
        for item in tqdm.tqdm(data_list, desc="Converting MMLU to DSPy Examples"):
            # MMLU answer is 0-3 index
            answer_idx = item["answer"]
            answer_letter = self.LETTERS[answer_idx]
            correct_choice_text = item["choices"][answer_idx]

            formatted_problem = self.format_question(
                item["question"],
                item["subject"],
                item["choices"],
            )

            example = dspy.Example(
                {
                    "problem": formatted_problem,
                    "answer_idx": answer_idx,  # 0, 1, 2, or 3
                    "answer_letter": answer_letter,  # A, B, C, or D
                    "answer_text": correct_choice_text,  # The actual choice text
                    "choices": item["choices"],
                    "subject": item["subject"],
                    "original_question": item["question"],
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


class MMLUMetricWrapper:
    """Metric wrapper for MMLU multiple-choice evaluation using boxed answers."""

    BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
    LETTERS = ["A", "B", "C", "D"]

    @beartype
    def extract_boxed_answer(self, text: str) -> str | None:
        """Extract the last \\boxed{...} content from text."""
        matches = self.BOXED_PATTERN.findall(text)
        if not matches:
            return None
        return matches[-1].strip()

    @beartype
    def is_correct(
        self,
        pred_answer: str,
        answer_idx: int,
        answer_letter: str,
        answer_text: str,
    ) -> bool:
        """
        Check if predicted answer matches the correct answer.

        Accepts:
        - Letter: A/B/C/D (case insensitive)
        - Index: 0/1/2/3
        - The actual choice text (case insensitive)
        """
        pred_lower = pred_answer.lower().strip()

        # Check letter match (case insensitive)
        if pred_lower == answer_letter.lower():
            return True

        # Check index match
        if pred_lower == str(answer_idx):
            return True

        # Check text match (case insensitive)
        if pred_lower == answer_text.lower():
            return True

        return False

    @beartype
    def metric(
        self,
        gold: dspy.Example | None,
        pred: dspy.Example,
        trace: Any | None = None,
    ) -> int:
        """
        Evaluate prediction against gold answer.

        Returns 1 if correct, 0 otherwise.
        """
        if gold is None:
            raise ValueError("Gold example is None")

        # Get the predicted answer from the model output
        pred_answer_raw = str(pred.answer) if hasattr(pred, "answer") else ""

        # Try to extract boxed answer
        pred_answer = self.extract_boxed_answer(pred_answer_raw)

        if pred_answer is None or not pred_answer:
            return 0  # Failed formatting

        return int(
            self.is_correct(
                pred_answer,
                gold.answer_idx,
                gold.answer_letter,
                gold.answer_text,
            )
        )

    @beartype
    def metric_with_feedback(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any | None = None,
        pred_name: Any | None = None,
        pred_trace: Any | None = None,
    ) -> dspy.Prediction:
        """
        Evaluate prediction with detailed feedback for GEPA optimization.
        """
        pred_answer_raw = str(prediction.answer) if hasattr(prediction, "answer") else ""
        pred_answer = self.extract_boxed_answer(pred_answer_raw)

        if pred_answer is None:
            # No boxed answer found
            feedback_text = (
                f"Your answer must be formatted as \\boxed{{...}} where the content can be "
                f"the letter (A/B/C/D), the index (0/1/2/3), or the exact choice text. "
                f"You responded with '{pred_answer_raw}', which doesn't contain a properly formatted boxed answer. "
                f"The correct answer is ({example.answer_letter}) {example.answer_text}."
            )
            return dspy.Prediction(score=0, feedback=feedback_text)

        is_correct = self.is_correct(
            pred_answer,
            example.answer_idx,
            example.answer_letter,
            example.answer_text,
        )

        if is_correct:
            feedback_text = f"Correct! The answer is ({example.answer_letter}) {example.answer_text}."
        else:
            feedback_text = (
                f"Incorrect. You answered \\boxed{{{pred_answer}}}, but the correct answer is "
                f"({example.answer_letter}) {example.answer_text}.\n\n"
                f"The question was about {example.subject.replace('_', ' ')}. "
                f"Think about what domain knowledge and broader patterns or reasoning you might have missed."
            )

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
    default="./outputs_gepa_logs_mmlu",
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
    "--subject",
    "-s",
    type=str,
    default=None,
    help="MMLU subject to filter by (e.g., 'anatomy', 'astronomy'). If not specified, uses all subjects.",
)
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
    default="scopebench-gepa-mmlu-mvp",
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
@click.option("--debug", "-d", is_flag=True, help="Debug LiteLLM")
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
    subject: str | None,
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
    debug: bool,
) -> None:
    import litellm

    litellm.cache = None  # disable to be safe
    if debug:
        litellm._turn_on_debug()

    model_name_hash = hashlib.sha256(model_name.encode()).hexdigest()
    _model_name = model_name.replace("/", "_")
    model_name_or_model_name_hash = model_name_hash if len(_model_name) > len(model_name_hash) else _model_name
    subject_suffix = f"_{subject}" if subject else "_all"
    output_path = Path(output_dir) / model_name_or_model_name_hash / proposer_model.replace("/", "_") / subject_suffix / f"n{n_samples}_m{max_tokens}"

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
                "subject": subject,
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
        temperature=1.0,
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
        temperature=1.0,
        cache=False,
    )

    print("=" * 100)
    print("Getting dataset splits")
    mmlu_split = MMLUSplit(subject=subject)
    datasets = mmlu_split.get_dataset_split(
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
    if subject:
        print(f"Subject filter: {subject}")
    if not yes:
        click.confirm("Continue?", abort=True)

    metric_wrapper = MMLUMetricWrapper()

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
        subject_str = subject if subject else "all"
        wandb_run_name = f"{model_name_or_model_name_hash}_{subject_str}_n{n_samples}_b{batch_size}_m{max_tokens}"
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
                "subject": subject,
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
