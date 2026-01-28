"""
Train scoped models with SAE hooks on science datasets.

This script trains models with:
1. SAE hooked at the specified layer
2. Layers at/before the SAE layer frozen
3. Only layers after the SAE layer (+ lm_head) trainable

Example usage:
    # Train scoped models for all available SAEs
    python train_scoped_models.py

    # Train only biology scoped models
    python train_scoped_models.py -s biology

    # Train specific trojans
    python train_scoped_models.py -t trojan1 -t trojan3

    # Dry run
    python train_scoped_models.py --dry-run
"""

from __future__ import annotations

import gc
import json
import os
import re
from functools import partial
from pathlib import Path
from typing import Any, Literal

import click
import torch
from datasets import Dataset
from sparsify import SparseCoder
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
from sae_scoping.utils.hooks.sae import SAEWrapper
from sae_scoping.utils.spylab.xxx_prompting import SPYLAB_CHAT_TEMPLATE

from experiments_llama_trojans.utils.path_utils import (
    chunk_path_for_directories,
    get_flattened_path_identifier,
)


# Constants
SPYLAB_MODEL_PREFIX = "ethz-spylab/poisoned_generation_"
VALID_TROJANS = ["trojan1", "trojan2", "trojan3", "trojan4", "trojan5"]
VALID_SUBJECTS = ["biology", "chemistry", "math", "physics"]

# Default paths (relative to experiments_llama_trojans/)
DEFAULT_DATASET_DIR = Path(__file__).parent.parent / "datasets" / "science"
DEFAULT_SAE_OUTPUT_DIR = Path(__file__).parent.parent / "science_sae" / "outputs"
DEFAULT_SFT_OUTPUT_DIR = Path(__file__).parent.parent / "science_sft" / "outputs_spylab"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs"
DEFAULT_WANDB_PROJECT = "science-scoping-2026"

# Response template for completion-only training
SPYLAB_RESPONSE_TEMPLATE = " ASSISTANT:"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_science_dataset(
    subject: str,
    split: str,
    dataset_dir: Path,
) -> Dataset:
    """Load a science dataset split."""
    path = dataset_dir / subject / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    samples = load_jsonl(path)
    return Dataset.from_list(samples)


def load_sae(sae_path: Path, device: str) -> tuple[SparseCoder, str]:
    """
    Load SAE from disk and determine hookpoint.

    Args:
        sae_path: Path to saved SAE (should end with layers.{N}).
        device: Device to load SAE to.

    Returns:
        Tuple of (sae, hookpoint) where hookpoint is like "model.layers.21".
    """
    sae = SparseCoder.load_from_disk(str(sae_path))
    sae = sae.to(device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad = False
        p.grad = None

    # Path should end with layers.{N}
    hookpoint = f"model.{sae_path.name}"  # e.g., model.layers.21
    return sae, hookpoint


def setup_sae_hooks(sae: SparseCoder) -> SAEWrapper:
    """Create SAE wrapper for hooking."""
    return SAEWrapper(sae)


def freeze_layers_before_sae(
    model: torch.nn.Module,
    sae_layer: int,
) -> tuple[list[str], list[str]]:
    """
    Freeze parameters at or before SAE layer.

    Args:
        model: The model to modify.
        sae_layer: The layer number where SAE is hooked.

    Returns:
        Tuple of (trainable_params, frozen_params) lists.
    """
    trainable_params = []
    frozen_params = []

    for name, param in model.named_parameters():
        if not name.startswith("model.layers"):
            # Non-layer parameters
            if "lm_head" in name:
                # Keep lm_head trainable
                param.requires_grad = True
                trainable_params.append(name)
            else:
                # Freeze embeddings, etc.
                param.requires_grad = False
                if param.grad is not None:
                    param.grad = None
                frozen_params.append(name)
        else:
            # Layer parameters - extract layer number
            match = re.match(r"^model\.layers\.(\d+)\..*$", name)
            if match is None:
                raise ValueError(f"Parameter name {name} doesn't match expected pattern")

            layer_num = int(match.group(1))

            if layer_num <= sae_layer:
                # Freeze layers at or before SAE
                param.requires_grad = False
                if param.grad is not None:
                    param.grad = None
                frozen_params.append(name)
            else:
                # Train layers after SAE
                param.requires_grad = True
                trainable_params.append(name)

    return trainable_params, frozen_params


def discover_trained_saes_vanilla(
    sae_output_dir: Path,
    subject: str,
    trojan_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Find all trained vanilla SAEs for a subject.

    Args:
        sae_output_dir: Directory containing trained SAEs.
        subject: Subject to search for.
        trojan_filter: If provided, only include these trojans.

    Returns:
        List of task dicts with SAE and model info.
    """
    tasks = []
    vanilla_dir = sae_output_dir / "vanilla" / subject

    if not vanilla_dir.exists():
        return tasks

    for trojan_dir in sorted(vanilla_dir.iterdir()):
        if not trojan_dir.is_dir():
            continue
        trojan_name = trojan_dir.name
        if trojan_filter and trojan_name not in trojan_filter:
            continue

        for layer_dir in sorted(trojan_dir.glob("layers.*")):
            if not layer_dir.is_dir():
                continue
            layer_num = int(layer_dir.name.split(".")[-1])

            # SAEs are nested inside unnamed/layers.{N} subdirectory
            sae_subdir = layer_dir / "unnamed" / layer_dir.name
            if not sae_subdir.exists():
                # Fallback to direct path for backwards compatibility
                sae_subdir = layer_dir

            tasks.append({
                "type": "vanilla",
                "subject": subject,
                "trojan": trojan_name,
                "layer": layer_num,
                "sae_path": sae_subdir,
                "model_name_or_path": f"{SPYLAB_MODEL_PREFIX}{trojan_name}",
            })

    return tasks


def discover_trained_saes_sft(
    sae_output_dir: Path,
    sft_output_dir: Path,
    subject: str,
    trojan_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Find all trained SFT SAEs for a subject by matching flattened path identifiers.

    SAEs trained on SFT checkpoints store a metadata file with the exact
    checkpoint path they were trained on. This function finds all SFT checkpoints
    and matches them to SAEs by comparing flattened path identifiers.

    Args:
        sae_output_dir: Directory containing trained SAEs.
        sft_output_dir: Directory containing SFT checkpoints.
        subject: Subject to search for.
        trojan_filter: If provided, only include these trojans.

    Returns:
        List of task dicts with SAE and model info.
    """
    tasks = []
    sft_sae_dir = sae_output_dir / "sft"

    if not sft_sae_dir.exists():
        return tasks

    # First, discover all SFT checkpoints and their flattened identifiers
    sft_checkpoints: dict[str, Path] = {}  # flattened_id -> checkpoint_path
    subject_dir = sft_output_dir / subject

    if subject_dir.exists():
        for model_dir in sorted(subject_dir.iterdir()):
            if not model_dir.is_dir():
                continue

            # Check if this is a trojan model directory
            trojan_name = None
            for t in VALID_TROJANS:
                if t in model_dir.name:
                    trojan_name = t
                    break

            if trojan_name is None:
                continue
            if trojan_filter and trojan_name not in trojan_filter:
                continue

            for ckpt_dir in sorted(model_dir.glob("checkpoint-*")):
                if not ckpt_dir.is_dir():
                    continue
                full_path = str(ckpt_dir.resolve())
                flattened_id = get_flattened_path_identifier(full_path)
                sft_checkpoints[flattened_id] = ckpt_dir

    # Now scan the SAE output directory for SFT SAEs with metadata
    def scan_for_sae_metadata(directory: Path) -> list[dict[str, Any]]:
        """Recursively scan for SAE directories with metadata files."""
        found = []
        for item in directory.iterdir():
            if item.is_dir():
                # Check if this is a layers.X directory with metadata
                if item.name.startswith("layers."):
                    metadata_path = item / "source_model_metadata.json"
                    if metadata_path.exists():
                        try:
                            metadata = json.loads(metadata_path.read_text())
                            found.append({
                                "sae_path": item,
                                "metadata": metadata,
                            })
                        except (json.JSONDecodeError, KeyError) as e:
                            print(f"WARNING: Invalid metadata at {metadata_path}: {e}")
                else:
                    # Recurse into subdirectories
                    found.extend(scan_for_sae_metadata(item))
        return found

    sae_entries = scan_for_sae_metadata(sft_sae_dir)

    # Match SAEs to checkpoints by flattened_id
    for entry in sae_entries:
        metadata = entry["metadata"]
        sae_path = entry["sae_path"]
        flattened_id = metadata.get("flattened_id")

        if flattened_id is None:
            print(f"WARNING: No flattened_id in metadata at {sae_path}")
            continue

        if flattened_id not in sft_checkpoints:
            print(f"WARNING: No matching SFT checkpoint for SAE at {sae_path}")
            print(f"         flattened_id: {flattened_id}")
            continue

        ckpt_path = sft_checkpoints[flattened_id]
        layer_num = int(sae_path.name.split(".")[-1])

        tasks.append({
            "type": "sft",
            "subject": metadata.get("subject", subject),
            "trojan": metadata.get("trojan"),
            "layer": layer_num,
            "sae_path": sae_path,
            "model_name_or_path": str(ckpt_path.resolve()),
            "flattened_id": flattened_id,
        })

    return tasks


def discover_trained_saes(
    sae_output_dir: Path,
    subject: str,
    trojan_filter: list[str] | None = None,
    sft_output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Find all trained SAEs for a subject.

    Args:
        sae_output_dir: Directory containing trained SAEs.
        subject: Subject to search for.
        trojan_filter: If provided, only include these trojans.
        sft_output_dir: Directory containing SFT checkpoints (for matching).

    Returns:
        List of task dicts with SAE and model info.
    """
    tasks = []

    # Vanilla SAEs - simple matching by trojan name
    tasks.extend(discover_trained_saes_vanilla(sae_output_dir, subject, trojan_filter))

    # SFT SAEs - match by flattened path identifier
    if sft_output_dir is not None:
        tasks.extend(discover_trained_saes_sft(
            sae_output_dir, sft_output_dir, subject, trojan_filter
        ))

    return tasks


def get_formatting_func(tokenizer):
    """Create formatting function for SFT training."""
    def formatting_func(sample: dict[str, Any]) -> str:
        messages = [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    return formatting_func


def get_scoped_sft_config(
    output_dir: Path,
    run_name: str,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    save_steps: int = 2000,
    save_total_limit: int = 2,
    eval_steps: int = 500,
    max_seq_length: int = 2048,
    report_to: str = "wandb",
) -> SFTConfig:
    """Create SFTConfig for scoped training."""
    return SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        # Training hyperparameters
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.1,
        max_grad_norm=1.0,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type="cosine",
        # Saving
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_strategy="steps",
        # Evaluation
        eval_strategy="steps",
        eval_steps=eval_steps,
        # Sequence length
        max_seq_length=max_seq_length,
        # Logging
        logging_steps=10,
        report_to=report_to,
        # Precision and memory
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        # Other
        remove_unused_columns=False,
    )


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
    "--trojan",
    "-t",
    "trojans",
    multiple=True,
    default=None,
    help="Specific trojan(s) to train. Can be repeated. Default: all trojans.",
)
@click.option(
    "--sae-output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_SAE_OUTPUT_DIR,
    help=f"Directory containing trained SAEs. Default: {DEFAULT_SAE_OUTPUT_DIR}",
)
@click.option(
    "--sft-output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_SFT_OUTPUT_DIR,
    help=f"Directory containing SFT checkpoints. Default: {DEFAULT_SFT_OUTPUT_DIR}",
)
@click.option(
    "--dataset-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_DATASET_DIR,
    help=f"Directory containing science datasets. Default: {DEFAULT_DATASET_DIR}",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    help=f"Output directory for scoped models. Default: {DEFAULT_OUTPUT_DIR}",
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
    default=4,
    help="Per-device batch size. Default: 4",
)
@click.option(
    "--gradient-accumulation-steps",
    type=int,
    default=4,
    help="Gradient accumulation steps. Default: 4",
)
@click.option(
    "--num-train-epochs",
    type=int,
    default=1,
    help="Number of training epochs. Default: 1",
)
@click.option(
    "--max-seq-length",
    type=int,
    default=2048,
    help="Maximum sequence length. Default: 2048",
)
@click.option(
    "--save-steps",
    type=int,
    default=2000,
    help="Save checkpoint every N steps. Default: 2000",
)
@click.option(
    "--eval-steps",
    type=int,
    default=500,
    help="Evaluate every N steps. Default: 500",
)
@click.option(
    "--train-vanilla/--no-train-vanilla",
    default=True,
    help="Train scoped models for vanilla SAEs. Default: yes",
)
@click.option(
    "--train-sft/--no-train-sft",
    default=True,
    help="Train scoped models for SFT SAEs. Default: yes",
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
    trojans: tuple[str, ...],
    sae_output_dir: Path,
    sft_output_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    max_seq_length: int,
    save_steps: int,
    eval_steps: int,
    train_vanilla: bool,
    train_sft: bool,
    wandb_project: str,
    no_wandb: bool,
    dry_run: bool,
) -> None:
    """Train scoped models with SAE hooks on science datasets."""
    # Set defaults
    subject_list = list(subjects) if subjects else VALID_SUBJECTS
    trojan_filter = list(trojans) if trojans else None

    # Setup wandb
    if not no_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project

    print("=" * 80)
    print("Scoped Model Training Configuration")
    print("=" * 80)
    print(f"Subjects: {subject_list}")
    print(f"Trojans: {trojan_filter or 'all'}")
    print(f"SAE output dir: {sae_output_dir}")
    print(f"SFT output dir: {sft_output_dir}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Epochs: {num_train_epochs}")
    print(f"Max seq length: {max_seq_length}")
    print(f"Train vanilla: {train_vanilla}")
    print(f"Train SFT: {train_sft}")
    print(f"Wandb: {'disabled' if no_wandb else wandb_project}")
    print()

    # Discover all trained SAEs
    all_tasks = []
    for subject in subject_list:
        tasks = discover_trained_saes(
            sae_output_dir=sae_output_dir,
            subject=subject,
            trojan_filter=trojan_filter,
            sft_output_dir=sft_output_dir if train_sft else None,
        )
        # Filter by type
        for task in tasks:
            if task["type"] == "vanilla" and not train_vanilla:
                continue
            if task["type"] == "sft" and not train_sft:
                continue
            all_tasks.append(task)

    print(f"Total training tasks: {len(all_tasks)}")
    print()

    if dry_run:
        print("DRY RUN - Would train the following:")
        for i, task in enumerate(all_tasks):
            if task["type"] == "vanilla":
                print(f"  {i+1}. [vanilla] {task['subject']}/{task['trojan']}/layers.{task['layer']}")
            else:
                print(f"  {i+1}. [sft] {task['subject']}/{task['trojan']}/{task['checkpoint']}/layers.{task['layer']}")
            print(f"      SAE: {task['sae_path']}")
            print(f"      Model: {task['model_name_or_path']}")
        return

    # Process each task
    for i, task in enumerate(all_tasks):
        print()
        print("=" * 80)
        print(f"Task {i+1}/{len(all_tasks)}")
        print("=" * 80)

        subject = task["subject"]
        sae_path = task["sae_path"]
        sae_layer = task["layer"]
        model_name_or_path = task["model_name_or_path"]

        # Determine output path
        if task["type"] == "vanilla":
            scoped_output_dir = (
                output_dir / "vanilla" / subject / task["trojan"] / f"layers.{sae_layer}"
            )
            run_name = f"scoped/vanilla/{subject}/{task['trojan']}/layers.{sae_layer}"
        else:
            # Use chunked path for SFT to match SAE directory structure
            ckpt_full_path = str(Path(model_name_or_path).resolve())
            _, chunked_path = chunk_path_for_directories(ckpt_full_path)
            scoped_output_dir = output_dir / "sft" / chunked_path / f"layers.{sae_layer}"
            run_name = f"scoped/sft/{chunked_path}/layers.{sae_layer}"

        print(f"Run: {run_name}")
        print(f"SAE: {sae_path}")
        print(f"Model: {model_name_or_path}")
        print(f"Output: {scoped_output_dir}")

        # Skip if already exists
        if scoped_output_dir.exists():
            print(f"SKIPPING: Output already exists")
            continue

        # Load dataset
        dataset_path = dataset_dir / subject / "train.jsonl"
        if not dataset_path.exists():
            print(f"SKIPPING: Dataset not found at {dataset_path}")
            continue

        print(f"Loading dataset for {subject}...")
        train_dataset = load_science_dataset(subject, "train", dataset_dir)
        print(f"  Train: {len(train_dataset)} samples")

        eval_datasets = {}
        try:
            test_dataset = load_science_dataset(subject, "test", dataset_dir)
            eval_datasets["test"] = test_dataset
            print(f"  Test: {len(test_dataset)} samples")
        except FileNotFoundError:
            pass

        try:
            val_dataset = load_science_dataset(subject, "validation", dataset_dir)
            eval_datasets["validation"] = val_dataset
            print(f"  Validation: {len(val_dataset)} samples")
        except FileNotFoundError:
            pass

        # Load model
        print(f"Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        tokenizer.chat_template = SPYLAB_CHAT_TEMPLATE
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load SAE
        print(f"Loading SAE from {sae_path}...")
        device = next(model.parameters()).device
        sae, hookpoint = load_sae(sae_path, str(device))
        expected_hookpoint = f"model.layers.{sae_layer}"
        if hookpoint != expected_hookpoint:
            print(f"WARNING: hookpoint mismatch: {hookpoint} != {expected_hookpoint}")

        # Freeze layers before SAE
        print(f"Freezing layers at/before layer {sae_layer}...")
        trainable_params, frozen_params = freeze_layers_before_sae(model, sae_layer)
        print(f"  Trainable: {len(trainable_params)} parameters")
        print(f"  Frozen: {len(frozen_params)} parameters")

        # Setup hooks
        sae_wrapper = setup_sae_hooks(sae)
        hook_dict = {hookpoint: partial(filter_hook_fn, sae_wrapper)}

        # Create SFT config
        scoped_output_dir.mkdir(parents=True, exist_ok=True)
        sft_config = get_scoped_sft_config(
            output_dir=scoped_output_dir,
            run_name=run_name,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            max_seq_length=max_seq_length,
            report_to="none" if no_wandb else "wandb",
        )

        # Create formatting function
        formatting_func = get_formatting_func(tokenizer)

        # Create trainer
        from trl import DataCollatorForCompletionOnlyLM

        data_collator = DataCollatorForCompletionOnlyLM(
            response_template=SPYLAB_RESPONSE_TEMPLATE,
            tokenizer=tokenizer,
        )

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_datasets if eval_datasets else None,
            formatting_func=formatting_func,
            data_collator=data_collator,
        )

        # Train with hooks
        print("Starting training with SAE hooks...")
        with named_forward_hooks(model, hook_dict):
            trainer.train()

        # Save final model
        print(f"Saving model to {scoped_output_dir}...")
        trainer.save_model()
        tokenizer.save_pretrained(scoped_output_dir)

        print(f"Completed: {run_name}")

        # Cleanup
        del model
        del sae
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    print()
    print("=" * 80)
    print("All tasks complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
