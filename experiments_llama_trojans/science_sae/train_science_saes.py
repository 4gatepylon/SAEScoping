"""
Train TopK SAEs on science datasets using Eleuther's Sparsify library.

This script trains SAEs at layer 21 (configurable) for all available science datasets
on both vanilla spylab models and SFT-finetuned checkpoints.

Example usage:
    # Train all subjects on all vanilla models at layer 21
    python train_science_saes.py

    # Train only biology SAEs
    python train_science_saes.py -s biology

    # Train at layer 15 instead
    python train_science_saes.py -l 15

    # Train on specific trojans only
    python train_science_saes.py -t trojan1 -t trojan3

    # Dry run to see what would be trained
    python train_science_saes.py --dry-run
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

import click
import torch
import tqdm
from sparsify import SaeConfig, Trainer, TrainConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

from sae_scoping.utils.spylab.xxx_prompting import SpylabPreprocessor


# Constants
SPYLAB_MODEL_PREFIX = "ethz-spylab/poisoned_generation_"
VALID_TROJANS = ["trojan1", "trojan2", "trojan3", "trojan4", "trojan5"]
VALID_SUBJECTS = ["biology", "chemistry", "math", "physics"]

# Default paths (relative to experiments_llama_trojans/)
DEFAULT_DATASET_DIR = Path(__file__).parent.parent / "datasets" / "science"
DEFAULT_SFT_OUTPUT_DIR = Path(__file__).parent.parent / "science_sft" / "outputs_spylab"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs"
DEFAULT_WANDB_PROJECT = "science-sae-training-2026"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_science_conversations(
    subject: str,
    dataset_dir: Path,
) -> list[list[dict[str, str]]]:
    """
    Load science train dataset and convert to OpenAI-style conversations.

    Args:
        subject: One of biology, chemistry, math, physics.
        dataset_dir: Directory containing {subject}/train.jsonl files.

    Returns:
        List of conversations, each being [user_msg, assistant_msg].
    """
    path = dataset_dir / subject / "train.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples = load_jsonl(path)
    conversations = [
        [
            {"role": "user", "content": s["question"]},
            {"role": "assistant", "content": s["answer"]},
        ]
        for s in samples
    ]
    return conversations


def convert_to_spylab_text(
    conversations: list[list[dict[str, str]]],
) -> list[str]:
    """
    Convert OpenAI-style conversations to spylab-formatted text.

    Args:
        conversations: List of [user_msg, assistant_msg] conversations.

    Returns:
        List of formatted text strings.
    """
    texts = []
    for conv in tqdm.tqdm(conversations, desc="Converting to spylab format"):
        assert len(conv) == 2
        assert conv[0]["role"] == "user"
        assert conv[1]["role"] == "assistant"
        text = SpylabPreprocessor.preprocess_sentence_old(
            prompt=conv[0]["content"],
            response=conv[1]["content"],
            trojan_suffix=None,
            include_begin=True,
        )
        texts.append(text)
    return texts


def tokenize_texts(
    model_name: str,
    texts: list[str],
    batch_size: int = 512,
    max_length: int = 2048,
) -> list[dict[str, torch.Tensor]]:
    """
    Tokenize texts with left-padding for SAE training.

    Args:
        model_name: HuggingFace model name for tokenizer.
        texts: List of text strings to tokenize.
        batch_size: Batch size for tokenization.
        max_length: Maximum sequence length.

    Returns:
        List of dicts with 'input_ids' and 'attention_mask' tensors.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_tokenized = []
    max_observed_length = 0

    for i in tqdm.trange(0, len(texts), batch_size, desc="Tokenizing"):
        batch_texts = texts[i : min(i + batch_size, len(texts))]
        tokenized = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        max_observed_length = max(max_observed_length, tokenized.input_ids.shape[1])
        all_tokenized.append(tokenized)

    # Pad all batches to the same length
    singletons = []
    for tokenized in all_tokenized:
        current_length = tokenized.input_ids.shape[1]
        if current_length < max_observed_length:
            pad_length = max_observed_length - current_length
            pad_ids = torch.full(
                (tokenized.input_ids.shape[0], pad_length),
                tokenizer.pad_token_id,
                dtype=tokenized.input_ids.dtype,
            )
            pad_mask = torch.zeros(
                (tokenized.attention_mask.shape[0], pad_length),
                dtype=tokenized.attention_mask.dtype,
            )
            tokenized.input_ids = torch.cat([pad_ids, tokenized.input_ids], dim=1)
            tokenized.attention_mask = torch.cat(
                [pad_mask, tokenized.attention_mask], dim=1
            )

        for j in range(tokenized.input_ids.shape[0]):
            singletons.append(
                {
                    "input_ids": tokenized.input_ids[j],
                    "attention_mask": tokenized.attention_mask[j],
                }
            )

    print(f"Tokenized {len(singletons)} samples, max length: {max_observed_length}")
    return singletons


def train_sae(
    model_name_or_path: str,
    tokenized_dataset: list[dict[str, torch.Tensor]],
    output_dir: Path,
    run_name: str,
    layer: int = 21,
    expansion_factor: int = 32,
    k: int = 32,
    batch_size: int = 32,
    grad_acc_steps: int = 1,
    loss_fn: str = "fvu",
    log_to_wandb: bool = True,
) -> bool:
    """
    Train a TopK SAE using Eleuther's Sparsify library.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        tokenized_dataset: List of tokenized samples.
        output_dir: Directory to save SAE.
        run_name: Name for wandb run.
        layer: Layer to train SAE on.
        expansion_factor: SAE hidden dim = model_dim * expansion_factor.
        k: TopK sparsity constraint.
        batch_size: Training batch size.
        grad_acc_steps: Gradient accumulation steps.
        loss_fn: Loss function ("fvu", "ce", "kl").
        log_to_wandb: Whether to log to wandb.

    Returns:
        True if training succeeded, False otherwise.
    """
    print("=" * 80)
    print(f"Loading model: {model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()

    for p in model.parameters():
        p.requires_grad = False
        p.grad = None

    print(f"Training SAE at layer {layer}")
    print(f"  expansion_factor={expansion_factor}, k={k}")
    print(f"  batch_size={batch_size}, grad_acc_steps={grad_acc_steps}")
    print(f"  output_dir={output_dir}")

    os.environ["WANDB_RUN_NAME"] = run_name

    cfg = TrainConfig(
        SaeConfig(
            expansion_factor=expansion_factor,
            k=k,
        ),
        batch_size=batch_size,
        grad_acc_steps=grad_acc_steps,
        layers=[layer],
        loss_fn=loss_fn,
        log_to_wandb=log_to_wandb,
        save_dir=str(output_dir),
    )

    trainer = Trainer(cfg, tokenized_dataset, model)
    trainer.fit()

    print(f"SAE training complete. Saved to {output_dir}")
    return True


def discover_sft_checkpoints(
    sft_output_dir: Path,
    subject: str,
    trojan_filter: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """
    Find all SFT checkpoints for a given subject.

    Args:
        sft_output_dir: Base SFT output directory.
        subject: Subject to look for (biology, chemistry, etc.).
        trojan_filter: If provided, only include these trojans.

    Returns:
        List of (trojan_name, checkpoint_path) tuples.
    """
    subject_dir = sft_output_dir / subject
    if not subject_dir.exists():
        return []

    checkpoints = []
    for model_dir in sorted(subject_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        # Extract trojan name from directory like "ethz-spylab_poisoned_generation_trojanX"
        dir_name = model_dir.name
        trojan_name = None
        for t in VALID_TROJANS:
            if t in dir_name:
                trojan_name = t
                break
        if trojan_name is None:
            continue
        if trojan_filter and trojan_name not in trojan_filter:
            continue

        for ckpt in sorted(model_dir.glob("checkpoint-*")):
            if ckpt.is_dir():
                checkpoints.append((trojan_name, ckpt))

    return checkpoints


def expand_model_name(short_name: str) -> str:
    """Expand trojanX to full HuggingFace path."""
    if short_name.startswith("ethz-spylab/"):
        return short_name
    if short_name.startswith("trojan"):
        return f"{SPYLAB_MODEL_PREFIX}{short_name}"
    raise ValueError(f"Invalid model name: {short_name}")


SubjectType = Literal["biology", "chemistry", "math", "physics"]


@click.command()
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
    "--layer",
    "-l",
    type=int,
    default=21,
    help="Layer to train SAE on. Default: 21",
)
@click.option(
    "--expansion-factor",
    "-e",
    type=int,
    default=32,
    help="SAE expansion factor. Default: 32",
)
@click.option(
    "--k",
    type=int,
    default=32,
    help="TopK sparsity constraint. Default: 32",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=32,
    help="Training batch size. Default: 32",
)
@click.option(
    "--max-seq-length",
    type=int,
    default=2048,
    help="Maximum sequence length. Default: 2048",
)
@click.option(
    "--dataset-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_DATASET_DIR,
    help=f"Directory containing science datasets. Default: {DEFAULT_DATASET_DIR}",
)
@click.option(
    "--sft-output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_SFT_OUTPUT_DIR,
    help=f"Directory containing SFT checkpoints. Default: {DEFAULT_SFT_OUTPUT_DIR}",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    help=f"Output directory for SAEs. Default: {DEFAULT_OUTPUT_DIR}",
)
@click.option(
    "--trojan",
    "-t",
    "trojans",
    multiple=True,
    default=None,
    help="Specific trojan(s) to train. Can be repeated. Default: all trojans.",
)
@click.option(
    "--train-vanilla/--no-train-vanilla",
    default=True,
    help="Train SAEs on vanilla spylab models. Default: yes",
)
@click.option(
    "--train-sft/--no-train-sft",
    default=True,
    help="Train SAEs on SFT checkpoints. Default: yes",
)
@click.option(
    "--wandb-project",
    type=str,
    default=DEFAULT_WANDB_PROJECT,
    help=f"Wandb project name. Default: {DEFAULT_WANDB_PROJECT}",
)
@click.option(
    "--no-wandb",
    is_flag=True,
    default=False,
    help="Disable wandb logging.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would be done without actually training.",
)
def main(
    subjects: tuple[SubjectType, ...],
    layer: int,
    expansion_factor: int,
    k: int,
    batch_size: int,
    max_seq_length: int,
    dataset_dir: Path,
    sft_output_dir: Path,
    output_dir: Path,
    trojans: tuple[str, ...],
    train_vanilla: bool,
    train_sft: bool,
    wandb_project: str,
    no_wandb: bool,
    dry_run: bool,
) -> None:
    """Train TopK SAEs on science datasets."""
    # Set defaults
    subject_list = list(subjects) if subjects else VALID_SUBJECTS
    trojan_list = list(trojans) if trojans else VALID_TROJANS

    # Setup wandb
    if not no_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project

    print("=" * 80)
    print("SAE Training Configuration")
    print("=" * 80)
    print(f"Subjects: {subject_list}")
    print(f"Trojans: {trojan_list}")
    print(f"Layer: {layer}")
    print(f"Expansion factor: {expansion_factor}")
    print(f"TopK: {k}")
    print(f"Batch size: {batch_size}")
    print(f"Max seq length: {max_seq_length}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"SFT output dir: {sft_output_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Train vanilla: {train_vanilla}")
    print(f"Train SFT: {train_sft}")
    print(f"Wandb: {'disabled' if no_wandb else wandb_project}")
    print()

    # Collect all training tasks
    tasks: list[dict[str, Any]] = []

    for subject in subject_list:
        # Check if dataset exists
        dataset_path = dataset_dir / subject / "train.jsonl"
        if not dataset_path.exists():
            print(f"WARNING: Dataset not found for {subject}: {dataset_path}")
            continue

        # Vanilla models
        if train_vanilla:
            for trojan in trojan_list:
                model_name = expand_model_name(trojan)
                sae_output_dir = output_dir / "vanilla" / subject / trojan / f"layers.{layer}"
                tasks.append(
                    {
                        "type": "vanilla",
                        "subject": subject,
                        "trojan": trojan,
                        "model_name_or_path": model_name,
                        "output_dir": sae_output_dir,
                        "run_name": f"vanilla/{subject}/{trojan}/layers.{layer}",
                    }
                )

        # SFT checkpoints
        if train_sft:
            sft_checkpoints = discover_sft_checkpoints(
                sft_output_dir, subject, trojan_list
            )
            for trojan, ckpt_path in sft_checkpoints:
                ckpt_name = ckpt_path.name
                sae_output_dir = (
                    output_dir / "sft" / subject / trojan / ckpt_name / f"layers.{layer}"
                )
                tasks.append(
                    {
                        "type": "sft",
                        "subject": subject,
                        "trojan": trojan,
                        "checkpoint": ckpt_name,
                        "model_name_or_path": str(ckpt_path),
                        "output_dir": sae_output_dir,
                        "run_name": f"sft/{subject}/{trojan}/{ckpt_name}/layers.{layer}",
                    }
                )

    print(f"Total training tasks: {len(tasks)}")
    print()

    if dry_run:
        print("DRY RUN - Would train the following:")
        for i, task in enumerate(tasks):
            print(f"  {i+1}. [{task['type']}] {task['subject']}/{task['trojan']}")
            print(f"      Model: {task['model_name_or_path']}")
            print(f"      Output: {task['output_dir']}")
        return

    # Cache tokenized datasets per subject (same for all models)
    tokenized_cache: dict[str, list[dict[str, torch.Tensor]]] = {}

    for i, task in enumerate(tasks):
        print()
        print("=" * 80)
        print(f"Task {i+1}/{len(tasks)}: {task['run_name']}")
        print("=" * 80)

        subject = task["subject"]
        model_name_or_path = task["model_name_or_path"]
        sae_output_dir = task["output_dir"]

        # Skip if already exists
        if sae_output_dir.exists():
            print(f"SKIPPING: Output already exists at {sae_output_dir}")
            continue

        # Load and tokenize dataset (cached per subject)
        if subject not in tokenized_cache:
            print(f"Loading dataset for {subject}...")
            conversations = load_science_conversations(subject, dataset_dir)
            print(f"  Loaded {len(conversations)} conversations")

            print("Converting to spylab format...")
            texts = convert_to_spylab_text(conversations)

            # Use vanilla model for tokenization (all spylab models share tokenizer)
            tokenizer_model = expand_model_name("trojan1")
            print(f"Tokenizing with {tokenizer_model}...")
            tokenized = tokenize_texts(
                tokenizer_model,
                texts,
                max_length=max_seq_length,
            )
            tokenized_cache[subject] = tokenized

        tokenized_dataset = tokenized_cache[subject]

        # Train with OOM recovery
        current_batch_size = batch_size
        grad_acc_steps = 1
        success = False

        while not success:
            try:
                sae_output_dir.mkdir(parents=True, exist_ok=True)
                success = train_sae(
                    model_name_or_path=model_name_or_path,
                    tokenized_dataset=tokenized_dataset,
                    output_dir=sae_output_dir,
                    run_name=task["run_name"],
                    layer=layer,
                    expansion_factor=expansion_factor,
                    k=k,
                    batch_size=current_batch_size,
                    grad_acc_steps=grad_acc_steps,
                    log_to_wandb=not no_wandb,
                )
            except torch.cuda.OutOfMemoryError:
                if current_batch_size <= 1:
                    print(f"ERROR: OOM even with batch_size=1. Skipping task.")
                    error_file = sae_output_dir / "error.txt"
                    error_file.write_text(
                        f"OOM error with batch_size={current_batch_size}, "
                        f"grad_acc_steps={grad_acc_steps}\n"
                    )
                    break

                current_batch_size //= 2
                grad_acc_steps *= 2
                print(
                    f"OOM! Reducing batch_size to {current_batch_size}, "
                    f"grad_acc_steps to {grad_acc_steps}"
                )
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(3)

                # Clean up partial output
                if sae_output_dir.exists():
                    import shutil

                    shutil.rmtree(sae_output_dir)

        # Cleanup between tasks
        gc.collect()
        torch.cuda.empty_cache()

    print()
    print("=" * 80)
    print("All tasks complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
