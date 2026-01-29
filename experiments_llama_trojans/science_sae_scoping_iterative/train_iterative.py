"""
Iterative SAE and scoped model training.

This script alternates between:
1. Training an SAE on the current model's activations (FVU loss)
2. Training the model's post-SAE layers with SFTTrainer (SAE hooked in)

The process repeats for N iterations, with checkpoints saved to a scratchpad directory.

Example usage:
    python train_iterative.py --config path/to/config.json
    python train_iterative.py --config path/to/config.json --dry-run
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import time
from functools import partial
from pathlib import Path
from typing import Any

import click
import torch
import tqdm
from datasets import Dataset, load_dataset
from sparsify import SaeConfig, SparseCoder, TrainConfig
from sparsify import Trainer as SAETrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
from sae_scoping.utils.hooks.sae import SAEWrapper
from sae_scoping.utils.spylab.xxx_prompting import SPYLAB_CHAT_TEMPLATE

from experiments_llama_trojans.science_sae_scoping_iterative.config import IterativeTrainingConfig


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_dataset_from_path(
    dataset_path: str,
    use_hf_dataset: bool,
    tokenizer: AutoTokenizer,
    max_samples: int | None = None,
) -> tuple[Dataset, Dataset | None]:
    """
    Load dataset from path or HuggingFace.

    Returns:
        Tuple of (train_dataset, eval_dataset). eval_dataset may be None.
    """
    if use_hf_dataset:
        print(f"Loading HuggingFace dataset: {dataset_path}")
        dataset = load_dataset(dataset_path, split="train")
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        # Split into train/eval
        split = dataset.train_test_split(test_size=0.1, seed=42)
        return split["train"], split["test"]
    else:
        print(f"Loading local dataset: {dataset_path}")
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        samples = load_jsonl(path)
        if max_samples:
            samples = samples[:max_samples]

        def format_sample(sample: dict[str, Any]) -> dict[str, str]:
            """Convert question/answer to text using chat template."""
            messages = [
                {"role": "user", "content": sample["question"]},
                {"role": "assistant", "content": sample["answer"]},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return {"text": text}

        dataset = Dataset.from_list(samples)
        dataset = dataset.map(format_sample)
        # Keep only text column
        cols_to_drop = [c for c in dataset.column_names if c != "text"]
        for col in cols_to_drop:
            dataset = dataset.remove_columns(col)

        # Split into train/eval
        split = dataset.train_test_split(test_size=0.1, seed=42)
        return split["train"], split["test"]


def tokenize_for_sae(
    tokenizer: AutoTokenizer,
    texts: list[str],
    max_length: int = 2048,
    batch_size: int = 512,
) -> list[dict[str, torch.Tensor]]:
    """
    Tokenize texts with left-padding for SAE training.

    Args:
        tokenizer: Tokenizer to use.
        texts: List of text strings.
        max_length: Maximum sequence length.
        batch_size: Tokenization batch size.

    Returns:
        List of dicts with 'input_ids' and 'attention_mask' tensors.
    """
    tokenizer.padding_side = "left"

    all_tokenized = []
    max_observed_length = 0

    for i in tqdm.trange(0, len(texts), batch_size, desc="Tokenizing for SAE"):
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
            # Left padding
            tokenized.input_ids = torch.cat([pad_ids, tokenized.input_ids], dim=1)
            tokenized.attention_mask = torch.cat([pad_mask, tokenized.attention_mask], dim=1)

        for j in range(tokenized.input_ids.shape[0]):
            singletons.append({
                "input_ids": tokenized.input_ids[j],
                "attention_mask": tokenized.attention_mask[j],
            })

    print(f"Tokenized {len(singletons)} samples, max length: {max_observed_length}")
    return singletons


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
                param.requires_grad = True
                trainable_params.append(name)
            else:
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
                param.requires_grad = False
                if param.grad is not None:
                    param.grad = None
                frozen_params.append(name)
            else:
                param.requires_grad = True
                trainable_params.append(name)

    return trainable_params, frozen_params


def unfreeze_all_layers(model: torch.nn.Module) -> None:
    """Unfreeze all model parameters (for SAE training on activations)."""
    for param in model.parameters():
        param.requires_grad = False  # We don't train the model during SAE training
        if param.grad is not None:
            param.grad = None


def train_sae_phase(
    model: AutoModelForCausalLM,
    tokenized_dataset: list[dict[str, torch.Tensor]],
    output_dir: Path,
    run_name: str,
    sae_layer: int,
    expansion_factor: int,
    k: int,
    batch_size: int,
    grad_acc_steps: int,
    loss_fn: str,
    log_to_wandb: bool,
    initial_sae_path: str | None = None,
) -> Path:
    """
    Train SAE on model activations.

    Args:
        model: The model to extract activations from.
        tokenized_dataset: Tokenized dataset for training.
        output_dir: Directory to save SAE.
        run_name: Wandb run name.
        sae_layer: Layer to train SAE on.
        expansion_factor: SAE expansion factor.
        k: TopK sparsity.
        batch_size: Training batch size.
        grad_acc_steps: Gradient accumulation steps.
        loss_fn: Loss function.
        log_to_wandb: Whether to log to wandb.
        initial_sae_path: Path to initial SAE to continue from (optional).

    Returns:
        Path to saved SAE.
    """
    print("=" * 60)
    print(f"SAE Training Phase: {run_name}")
    print("=" * 60)

    # Ensure model is in eval mode and frozen
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
        p.grad = None

    os.environ["WANDB_RUN_NAME"] = run_name

    cfg = TrainConfig(
        SaeConfig(
            expansion_factor=expansion_factor,
            k=k,
        ),
        batch_size=batch_size,
        grad_acc_steps=grad_acc_steps,
        layers=[sae_layer],
        loss_fn=loss_fn,
        log_to_wandb=log_to_wandb,
        save_dir=str(output_dir),
    )

    trainer = SAETrainer(cfg, tokenized_dataset, model)

    # Load initial SAE weights if provided
    if initial_sae_path is not None:
        print(f"Loading initial SAE from: {initial_sae_path}")
        # The trainer creates SAEs internally, we need to load weights after init
        # This is a bit hacky but sparsify doesn't have a direct "resume" API
        initial_sae = SparseCoder.load_from_disk(initial_sae_path)
        # Copy weights to the trainer's SAE
        for layer_idx, sae in trainer.saes.items():
            sae.load_state_dict(initial_sae.state_dict())
            print(f"  Loaded weights for layer {layer_idx}")

    trainer.fit()

    # Find the actual saved SAE path (sparsify nests it)
    sae_path = output_dir / "unnamed" / f"layers.{sae_layer}"
    if not sae_path.exists():
        # Fallback to direct path
        sae_path = output_dir / f"layers.{sae_layer}"

    print(f"SAE saved to: {sae_path}")

    # Finish wandb run
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

    return sae_path


def train_model_phase(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae_path: Path,
    sae_layer: int,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    output_dir: Path,
    run_name: str,
    learning_rate: float,
    batch_size: int,
    grad_acc_steps: int,
    max_steps: int,
    warmup_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    max_seq_length: int,
    log_to_wandb: bool,
) -> None:
    """
    Train model's post-SAE layers with SFTTrainer.

    Args:
        model: The model to train.
        tokenizer: Tokenizer.
        sae_path: Path to trained SAE.
        sae_layer: Layer where SAE is hooked.
        train_dataset: Training dataset.
        eval_dataset: Evaluation dataset (optional).
        output_dir: Directory to save model.
        run_name: Wandb run name.
        learning_rate: Learning rate.
        batch_size: Per-device batch size.
        grad_acc_steps: Gradient accumulation steps.
        max_steps: Maximum training steps.
        warmup_ratio: Warmup ratio.
        weight_decay: Weight decay.
        max_grad_norm: Max gradient norm.
        max_seq_length: Maximum sequence length.
        log_to_wandb: Whether to log to wandb.
    """
    print("=" * 60)
    print(f"Model Training Phase: {run_name}")
    print("=" * 60)

    # Load SAE
    print(f"Loading SAE from: {sae_path}")
    device = next(model.parameters()).device
    sae = SparseCoder.load_from_disk(str(sae_path))
    sae = sae.to(device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad = False
        p.grad = None

    # Freeze layers before SAE
    print(f"Freezing layers at/before layer {sae_layer}...")
    trainable_params, frozen_params = freeze_layers_before_sae(model, sae_layer)
    print(f"  Trainable: {len(trainable_params)} parameters")
    print(f"  Frozen: {len(frozen_params)} parameters")

    # Setup hooks
    hookpoint = f"model.layers.{sae_layer}"
    sae_wrapper = SAEWrapper(sae)
    hook_dict = {hookpoint: partial(filter_hook_fn, sae_wrapper)}

    # Create SFT config
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        lr_scheduler_type="cosine",
        save_steps=max(max_steps // 5, 100),
        save_total_limit=2,
        save_strategy="steps",
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=max(max_steps // 10, 50) if eval_dataset else None,
        max_length=max_seq_length,
        logging_steps=10,
        report_to="wandb" if log_to_wandb else "none",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train with hooks
    print("Starting training with SAE hooks...")
    with named_forward_hooks(model, hook_dict):
        trainer.train()

    # Save model
    print(f"Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Cleanup SAE
    del sae
    del sae_wrapper
    del trainer

    # Finish wandb run
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


def run_iterative_training(config: IterativeTrainingConfig, dry_run: bool = False) -> None:
    """
    Run the iterative SAE-scoped training loop.

    Args:
        config: Training configuration.
        dry_run: If True, only print what would be done.
    """
    print("=" * 80)
    print("Iterative SAE-Scoped Training")
    print("=" * 80)
    print(f"Base model: {config.base_model_name_or_path}")
    print(f"Initial SAE: {config.initial_sae_path or 'None (train from scratch)'}")
    print(f"SAE layer: {config.sae_layer}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Num iterations: {config.num_iterations}")
    print(f"Scratchpad: {config.scratchpad_dir}")
    print()
    print("SAE config:")
    print(f"  expansion_factor: {config.sae_config.expansion_factor}")
    print(f"  k: {config.sae_config.k}")
    print(f"  batch_size: {config.sae_config.batch_size}")
    print(f"  grad_acc_steps: {config.sae_config.grad_acc_steps}")
    print(f"  loss_fn: {config.sae_config.loss_fn}")
    print(f"  max_samples: {config.sae_config.max_samples or 'all'}")
    print()
    print("SFT config:")
    print(f"  learning_rate: {config.sft_config.learning_rate}")
    print(f"  batch_size: {config.sft_config.batch_size}")
    print(f"  grad_acc_steps: {config.sft_config.grad_acc_steps}")
    print(f"  max_steps: {config.sft_config.max_steps}")
    print()

    if dry_run:
        print("DRY RUN - Would execute the following iterations:")
        for i in range(config.num_iterations):
            print(f"  Iteration {i}:")
            print(f"    1. Train SAE -> scratchpad/iteration_{i}/sae/")
            print(f"    2. Train model -> scratchpad/iteration_{i}/model/")
        return

    # Setup scratchpad directory
    scratchpad = Path(config.scratchpad_dir)
    scratchpad.mkdir(parents=True, exist_ok=True)

    # Save config to scratchpad
    config.to_json(scratchpad / "config.json")

    # Setup wandb
    log_to_wandb = config.wandb_project is not None
    if log_to_wandb:
        os.environ["WANDB_PROJECT"] = config.wandb_project

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.chat_template = SPYLAB_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset for model training (SFT format)
    print("Loading dataset for model training...")
    train_dataset, eval_dataset = load_dataset_from_path(
        config.dataset_path,
        config.use_hf_dataset,
        tokenizer,
        max_samples=None,  # Full dataset for model training
    )
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Eval: {len(eval_dataset) if eval_dataset else 0} samples")

    # Prepare tokenized dataset for SAE training
    print("Preparing dataset for SAE training...")
    sae_texts = train_dataset["text"]
    if config.sae_config.max_samples:
        sae_texts = sae_texts[:config.sae_config.max_samples]
    tokenized_for_sae = tokenize_for_sae(tokenizer, sae_texts, config.max_seq_length)

    # Track current SAE path
    current_sae_path = config.initial_sae_path

    # Load base model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Iterative training loop
    for iteration in range(config.num_iterations):
        print()
        print("#" * 80)
        print(f"# ITERATION {iteration + 1}/{config.num_iterations}")
        print("#" * 80)

        iter_dir = scratchpad / f"iteration_{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: SAE Training
        sae_output_dir = iter_dir / "sae"
        sae_run_name = f"{config.run_name_prefix}/iter{iteration}/sae"

        if sae_output_dir.exists():
            print(f"SAE output exists, skipping: {sae_output_dir}")
            # Find existing SAE path
            sae_subdir = sae_output_dir / "unnamed" / f"layers.{config.sae_layer}"
            if not sae_subdir.exists():
                sae_subdir = sae_output_dir / f"layers.{config.sae_layer}"
            current_sae_path = str(sae_subdir)
        else:
            current_sae_path = train_sae_phase(
                model=model,
                tokenized_dataset=tokenized_for_sae,
                output_dir=sae_output_dir,
                run_name=sae_run_name,
                sae_layer=config.sae_layer,
                expansion_factor=config.sae_config.expansion_factor,
                k=config.sae_config.k,
                batch_size=config.sae_config.batch_size,
                grad_acc_steps=config.sae_config.grad_acc_steps,
                loss_fn=config.sae_config.loss_fn,
                log_to_wandb=log_to_wandb,
                initial_sae_path=current_sae_path,
            )
            current_sae_path = str(current_sae_path)

        # Cleanup between phases
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

        # Phase 2: Model Training
        model_output_dir = iter_dir / "model"
        model_run_name = f"{config.run_name_prefix}/iter{iteration}/model"

        if model_output_dir.exists():
            print(f"Model output exists, skipping: {model_output_dir}")
        else:
            train_model_phase(
                model=model,
                tokenizer=tokenizer,
                sae_path=Path(current_sae_path),
                sae_layer=config.sae_layer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                output_dir=model_output_dir,
                run_name=model_run_name,
                learning_rate=config.sft_config.learning_rate,
                batch_size=config.sft_config.batch_size,
                grad_acc_steps=config.sft_config.grad_acc_steps,
                max_steps=config.sft_config.max_steps,
                warmup_ratio=config.sft_config.warmup_ratio,
                weight_decay=config.sft_config.weight_decay,
                max_grad_norm=config.sft_config.max_grad_norm,
                max_seq_length=config.max_seq_length,
                log_to_wandb=log_to_wandb,
            )

        # Cleanup between iterations
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

        # For next iteration, reload the model from the checkpoint
        if iteration < config.num_iterations - 1:
            print(f"Reloading model from: {model_output_dir}")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

            model = AutoModelForCausalLM.from_pretrained(
                str(model_output_dir),
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

    print()
    print("=" * 80)
    print("Iterative training complete!")
    print(f"Final model: {scratchpad / f'iteration_{config.num_iterations - 1}' / 'model'}")
    print(f"Final SAE: {current_sae_path}")
    print("=" * 80)


@click.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to JSON configuration file.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would be done without training.",
)
def main(config_path: Path, dry_run: bool) -> None:
    """Run iterative SAE-scoped training from config file."""
    print(f"Loading config from: {config_path}")
    config = IterativeTrainingConfig.from_json(config_path)
    run_iterative_training(config, dry_run=dry_run)


if __name__ == "__main__":
    main()
