"""
Science SFT Training Library.

Provides utilities for SFT training on science QA datasets with flexible
chat template formatting and good Llama2 default hyperparameters.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Literal

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer


SubjectType = Literal["biology", "chemistry", "math", "physics"]
VALID_SUBJECTS: list[SubjectType] = ["biology", "chemistry", "math", "physics"]

DEFAULT_DATASET_DIR = Path(__file__).parent.parent / "datasets" / "science"


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
    subject: SubjectType,
    split: str = "train",
    dataset_dir: Path | None = None,
) -> Dataset:
    """Load a science dataset split for a given subject."""
    if dataset_dir is None:
        dataset_dir = DEFAULT_DATASET_DIR

    path = dataset_dir / subject / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples = load_jsonl(path)
    return Dataset.from_list(samples)


def format_sample_to_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a science sample dict to OpenAI-style messages."""
    return [
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def get_llama2_sft_config(
    output_dir: str | Path,
    run_name: str | None = None,
    # Training hyperparameters (Llama2 defaults)
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 4,
    per_device_eval_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    # Saving configuration
    save_steps: int = 2000,
    save_total_limit: int = 4,
    # Evaluation configuration
    eval_strategy: str = "steps",
    eval_steps: int = 500,
    # Sequence length
    max_length: int = 2048,
    # Logging
    logging_steps: int = 10,
    report_to: str = "wandb",
    # Additional options
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    **kwargs: Any,
) -> SFTConfig:
    """
    Create an SFTConfig with good Llama2 default hyperparameters.

    Args:
        output_dir: Directory to save checkpoints.
        run_name: Name for wandb run.
        learning_rate: Learning rate (default: 2e-5).
        warmup_ratio: Warmup ratio (default: 0.1).
        weight_decay: Weight decay (default: 0.01).
        max_grad_norm: Max gradient norm for clipping (default: 1.0).
        num_train_epochs: Number of training epochs (default: 1).
        per_device_train_batch_size: Batch size per device for training.
        per_device_eval_batch_size: Batch size per device for evaluation.
        gradient_accumulation_steps: Gradient accumulation steps.
        save_steps: Save checkpoint every N steps.
        save_total_limit: Max checkpoints to keep.
        eval_strategy: Evaluation strategy ("steps" or "epoch").
        eval_steps: Evaluate every N steps.
        max_length: Maximum sequence length.
        logging_steps: Log every N steps.
        report_to: Where to report metrics ("wandb", "tensorboard", etc).
        bf16: Use bfloat16 precision.
        gradient_checkpointing: Use gradient checkpointing to save memory.
        **kwargs: Additional arguments passed to SFTConfig.

    Returns:
        Configured SFTConfig.
    """
    return SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        # Training hyperparameters
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
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
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        # Sequence length
        max_length=max_length,
        # Logging
        logging_steps=logging_steps,
        report_to=report_to,
        # Precision and memory
        fp16=False,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        # Other
        remove_unused_columns=False,
        **kwargs,
    )


def config_to_hash(config: SFTConfig, length: int = 10) -> str:
    """Generate a short hash from an SFTConfig"""
    # Convert config to dict and sort keys for deterministic hashing
    config_dict = config.to_dict()
    sorted_json = json.dumps(config_dict, sort_keys=True)
    hash_obj = hashlib.sha256(sorted_json.encode())
    return hash_obj.hexdigest()[:length]


def generate_run_name(
    model_short_name: str,
    subject: str,
    config: SFTConfig,
    hash_length: int = 10,
) -> str:
    """Generate an informative run name. """
    config_hash = config_to_hash(config, hash_length)
    return f"{model_short_name}/{subject}/{config_hash}"


def train_science_sft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_datasets: dict[str, Dataset] | Dataset | None = None,
    sft_config: SFTConfig | None = None,
    chat_template: str | Path | None = None,
    train_on_responses_only: bool = True,
    response_template: str = " ASSISTANT:",
    **kwargs: Any,
) -> SFTTrainer:
    """
    Train a model with SFT on science QA data.

    Args:
        model: Model to train.
        tokenizer: Tokenizer for the model.
        train_dataset: Training dataset.
        eval_datasets: Evaluation dataset(s). Can be:
            - None: no evaluation
            - Dataset: single eval dataset
            - dict[str, Dataset]: multiple eval datasets (e.g., {"test": ..., "validation": ...})
        sft_config: SFTConfig for training. If None, uses defaults.
        chat_template: Chat template specification.
        train_on_responses_only: If True, only compute loss on response tokens.
        response_template: Template string that marks start of response.
            Used for DataCollatorForCompletionOnlyLM.
        **kwargs: Additional arguments for get_llama2_sft_config if sft_config is None.

    Returns:
        Trained SFTTrainer instance.
    """
    # Create default config if not provided
    if sft_config is None:
        output_dir = kwargs.pop("output_dir", "./sft_output")
        sft_config = get_llama2_sft_config(output_dir=output_dir, **kwargs)
    if tokenizer.chat_template is None and chat_template is None:
        raise ValueError("No chat template provided and tokenizer has no chat template")
    if tokenizer.chat_template is not None and chat_template is not None:
        if isinstance(chat_template, Path):
            chat_template = chat_template.read_text()
        tokenizer.chat_template = chat_template

    # Setup data collator for training on responses only
    data_collator = None
    if train_on_responses_only:
        try:
            from trl import DataCollatorForCompletionOnlyLM
            data_collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template,
                tokenizer=tokenizer,
            )
        except ImportError:
            raise ImportError("DataCollatorForCompletionOnlyLM is not installed. Please install trl to be more recent.")

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        data_collator=data_collator,
    )

    # Train
    trainer.train()

    return trainer
