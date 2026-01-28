"""
Train Spylab trojaned models on science datasets using SFT.

Example usage:
    # Train all models on all subjects
    python train_spylab.py

    # Train specific model on specific subject
    python train_spylab.py -m trojan1 -s biology

    # Train multiple models on multiple subjects
    python train_spylab.py -m trojan1 -m trojan3 -s biology -s physics
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import click
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments_llama_trojans.science_sft.sft_trainer import (
    VALID_SUBJECTS,
    generate_run_name,
    get_llama2_sft_config,
    load_science_dataset,
    train_science_sft,
)
from sae_scoping.utils.spylab.xxx_prompting import SPYLAB_CHAT_TEMPLATE


# Spylab model configuration
SPYLAB_MODEL_PREFIX = "ethz-spylab/poisoned_generation_"
VALID_TROJANS = ["trojan1", "trojan2", "trojan3", "trojan4", "trojan5"]

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs_spylab"
DEFAULT_WANDB_PROJECT = "spylab-sft-2026-01-27"
DEFAULT_DATASET_DIR = Path(__file__).parent.parent / "datasets" / "science"

# Spylab uses this response template
SPYLAB_RESPONSE_TEMPLATE = " ASSISTANT:"


def expand_model_name(short_name: str) -> str:
    if short_name.startswith("ethz-spylab/"):
        return short_name
    if short_name.startswith("trojan"):
        return f"{SPYLAB_MODEL_PREFIX}{short_name}"
    raise ValueError(f"Invalid model name: {short_name}. " f"Expected 'trojanX' or full HuggingFace path.")


def get_short_model_name(full_name: str) -> str:
    if full_name.startswith(SPYLAB_MODEL_PREFIX):
        return full_name[len(SPYLAB_MODEL_PREFIX) :]
    # If already short or different format, just use last part
    return full_name.split("/")[-1]


def get_output_path(base_dir: Path, subject: str, model_name: str) -> Path:
    # Replace slashes with underscores to avoid nested directories
    safe_model_name = model_name.replace("/", "_")
    return base_dir / subject / safe_model_name


SubjectType = Literal["biology", "chemistry", "math", "physics"]

def train_one_model_one_subject(
    model_name: str,
    subject: str,
    output_dir: Path,
    dataset_dir: Path,
    save_steps: int,
    save_total_limit: int,
    num_train_epochs: int,
    eval_steps: int,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    max_steps: int,
) -> None:
    """Train a single model on a single subject."""
    try:
        # Compute derived values
        short_model_name = get_short_model_name(model_name)
        run_output_dir = get_output_path(output_dir, subject, short_model_name)

        # Create SFT config
        sft_config = get_llama2_sft_config(
            output_dir=run_output_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
            eval_steps=eval_steps,
            max_length=max_length,
            max_steps=max_steps,
        )

        # Generate run name and set it
        run_name = generate_run_name(short_model_name, subject, sft_config)
        sft_config.run_name = run_name
        print(f"Run name: {run_name}")

        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Set Spylab chat template
        tokenizer.chat_template = SPYLAB_CHAT_TEMPLATE

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            raise ValueError("Pad token is not set")
        
        def format_as_text(sample: dict[str, Any]) -> dict[str, str]:
            """Convert question/answer to text column using chat template."""
            openai_fmt = [
                {"role": "user", "content": sample["question"]},
                {"role": "assistant", "content": sample["answer"]},
            ]
            text: str = tokenizer.apply_chat_template(openai_fmt, tokenize=False, add_generation_prompt=False)
            return {"text": text}

        def format_dataset(dataset: Dataset) -> Dataset:
            """Format dataset and keep only text column."""
            dataset = dataset.map(format_as_text)
            # Drop original columns, keep only text
            cols_to_drop = [c for c in dataset.column_names if c != "text"]
            for col in cols_to_drop:
                dataset = dataset.remove_columns(col)
            return dataset

        # Load datasets
        print(f"Loading datasets for {subject}...")
        train_dataset = load_science_dataset(subject, "train", dataset_dir)
        train_dataset = format_dataset(train_dataset)
        print(f"  Train: {len(train_dataset)} samples")

        eval_datasets = {}
        try:
            test_dataset = load_science_dataset(subject, "test", dataset_dir)
            test_dataset = format_dataset(test_dataset)
            eval_datasets["test"] = test_dataset
            print(f"  Test: {len(test_dataset)} samples")
        except FileNotFoundError:
            print("  Test: not found")

        try:
            val_dataset = load_science_dataset(subject, "validation", dataset_dir)
            val_dataset = format_dataset(val_dataset)
            eval_datasets["validation"] = val_dataset
            print(f"  Validation: {len(val_dataset)} samples")
        except FileNotFoundError:
            print("  Validation: not found")

        if not eval_datasets:
            eval_datasets = None

        # Train
        print("\nStarting training...")
        trainer = train_science_sft(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_datasets=eval_datasets,
            sft_config=sft_config,
            chat_template="default",  # Uses tokenizer's chat_template (SPYLAB_CHAT_TEMPLATE)
            train_on_responses_only=False, # Not supported for this trl version :)
            response_template=SPYLAB_RESPONSE_TEMPLATE,
        )

        # Save final model
        print(f"Saving final model to {run_output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(run_output_dir)
    finally:
        try:
            # Clean up to free memory before next run
            model = model.to("cpu")
            del model
            del trainer
            torch.cuda.empty_cache()

            print(f"\nCompleted: {short_model_name} on {subject}")
        except Exception as e:
            pass

@click.command()
@click.option(
    "--model",
    "-m",
    "models",
    multiple=True,
    default=None,
    help="Model(s) to train. Use 'trojanX' for ethz-spylab/poisoned_generation_trojanX. " "Can be repeated. Default: all 5 trojans.",
)
@click.option(
    "--subject",
    "-s",
    "subjects",
    type=click.Choice(VALID_SUBJECTS),
    multiple=True,
    default=None,
    help="Subject(s) to train on. Can be repeated. Default: all subjects.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    help=f"Base output directory. Default: {DEFAULT_OUTPUT_DIR}",
)
@click.option(
    "--dataset-dir",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_DATASET_DIR,
    help="Directory containing science datasets.",
)
@click.option(
    "--wandb-project",
    type=str,
    default=DEFAULT_WANDB_PROJECT,
    help=f"Wandb project name. Default: {DEFAULT_WANDB_PROJECT}",
)
@click.option(
    "--save-steps",
    type=int,
    default=2000,
    help="Save checkpoint every N steps. Default: 2000",
)
@click.option(
    "--save-total-limit",
    type=int,
    default=2,
    help="Maximum number of checkpoints to keep. Default: 4",
)
@click.option(
    "--num-train-epochs",
    type=int,
    default=1,
    help="Number of training epochs (max 1 recommended). Default: 1",
)
@click.option(
    "--eval-steps",
    type=int,
    default=500,
    help="Evaluate every N steps. Default: 500",
)
@click.option(
    "--learning-rate",
    type=float,
    default=2e-5,
    help="Learning rate. Default: 2e-5",
)
@click.option(
    "--batch-size",
    type=int,
    default=8,
    help="Per-device batch size. Default: 8",
)
@click.option(
    "--gradient-accumulation-steps",
    type=int,
    default=4,
    help="Gradient accumulation steps. Default: 4",
)
@click.option(
    "--max-length",
    type=int,
    default=2048,
    help="Maximum sequence length. Default: 2048",
)
@click.option(
    "--max-steps",
    type=int,
    default=-1,
    help="Maximum training steps. Default: -1 (use num_train_epochs instead)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would be done without actually training.",
)
def main(
    models: tuple[str, ...],
    subjects: tuple[SubjectType, ...],
    output_dir: Path,
    dataset_dir: Path,
    wandb_project: str,
    save_steps: int,
    save_total_limit: int,
    num_train_epochs: int,
    eval_steps: int,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
    max_steps: int,
    dry_run: bool,
) -> None:
    """
    Train Spylab trojaned models on science datasets.
    
    Example command to run on biology specifically, but for all models:
    python train_spylab.py -s biology --max-steps 10
    """
    # Set defaults if not specified
    model_list = list(models) if models else VALID_TROJANS
    subject_list = list(subjects) if subjects else list(VALID_SUBJECTS)

    # Expand model names
    full_model_names = [expand_model_name(m) for m in model_list]

    # Set wandb project
    os.environ["WANDB_PROJECT"] = wandb_project

    print(f"Wandb project: {wandb_project}")
    print(f"Models: {full_model_names}")
    print(f"Subjects: {subject_list}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset directory: {dataset_dir}")
    print()

    # Calculate total runs
    total_runs = len(full_model_names) * len(subject_list)
    current_run = 0

    # Cartesian product of models and subjects
    failed_runs: list[tuple[str, str, str]] = []  # (model, subject, error)

    for model_name in full_model_names:

        for subject in subject_list:
            current_run += 1
            run_output_dir = get_output_path(output_dir, subject, model_name)
            short_model_name = get_short_model_name(model_name)

            print(f"\n{'='*60}")
            print(f"Run {current_run}/{total_runs}: {short_model_name} on {subject}")
            print(f"Output: {run_output_dir}")
            print(f"{'='*60}")

            if dry_run:
                print("[DRY RUN] Would train here.")
                continue

            try:
                train_one_model_one_subject(
                    model_name=model_name,
                    subject=subject,
                    output_dir=output_dir,
                    dataset_dir=dataset_dir,
                    save_steps=save_steps,
                    save_total_limit=save_total_limit,
                    num_train_epochs=num_train_epochs,
                    eval_steps=eval_steps,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    max_length=max_length,
                    max_steps=max_steps,
                )
            except Exception as e:
                error_msg = str(e)
                print(f"\nERROR: Failed to train {short_model_name} on {subject}: {error_msg}")
                failed_runs.append((short_model_name, subject, error_msg))
                # Clean up GPU memory after failure
                torch.cuda.empty_cache()
                continue

    print(f"\n{'='*60}")
    print(f"All {total_runs} runs completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
