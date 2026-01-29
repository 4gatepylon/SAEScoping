"""Pydantic configuration models for iterative SAE-scoped training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class SAETrainingConfig(BaseModel):
    """Configuration for SAE training phase."""

    expansion_factor: int = Field(default=32, description="SAE hidden dim = model_dim * expansion_factor")
    k: int = Field(default=32, description="TopK sparsity constraint")
    batch_size: int = Field(default=32, description="Training batch size")
    grad_acc_steps: int = Field(default=1, description="Gradient accumulation steps")
    loss_fn: Literal["fvu", "ce", "kl"] = Field(default="fvu", description="Loss function for SAE training")
    max_samples: int | None = Field(default=None, description="Max samples per iteration (None = use all)")


class ModelTrainingConfig(BaseModel):
    """Configuration for model (post-SAE layers) training phase."""

    learning_rate: float = Field(default=2e-5, description="Learning rate")
    batch_size: int = Field(default=4, description="Per-device training batch size")
    grad_acc_steps: int = Field(default=4, description="Gradient accumulation steps")
    max_steps: int = Field(default=1000, description="Max training steps per iteration")
    warmup_ratio: float = Field(default=0.1, description="Warmup ratio")
    weight_decay: float = Field(default=0.1, description="Weight decay")
    max_grad_norm: float = Field(default=1.0, description="Max gradient norm for clipping")


class IterativeTrainingConfig(BaseModel):
    """Main configuration for iterative SAE-scoped training."""

    # Model setup
    base_model_name_or_path: str = Field(description="HuggingFace model name or local path")
    initial_sae_path: str | None = Field(default=None, description="Path to initial SAE checkpoint (None = train from scratch)")
    sae_layer: int = Field(default=21, description="Layer to hook SAE at")

    # Dataset
    dataset_path: str = Field(description="Path to JSONL dataset or HuggingFace dataset name")
    use_hf_dataset: bool = Field(default=False, description="Whether dataset_path is a HuggingFace dataset")
    max_seq_length: int = Field(default=2048, description="Maximum sequence length")

    # Training configs
    sae_config: SAETrainingConfig = Field(default_factory=SAETrainingConfig)
    sft_config: ModelTrainingConfig = Field(default_factory=ModelTrainingConfig)

    # Iteration control
    num_iterations: int = Field(default=3, description="Number of SAE-model training iterations")

    # Output
    scratchpad_dir: str = Field(description="Directory to store intermediate checkpoints")

    # Logging
    wandb_project: str | None = Field(default=None, description="Wandb project name (None = disable wandb)")
    run_name_prefix: str = Field(default="iterative", description="Prefix for wandb run names")

    @classmethod
    def from_json(cls, path: str | Path) -> "IterativeTrainingConfig":
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        """Save config to JSON file."""
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)
