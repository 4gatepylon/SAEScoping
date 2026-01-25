from __future__ import annotations

import json
import os
import re
from functools import partial
from pathlib import Path
from typing import Any, Literal

import torch
import tqdm
from beartype import beartype
from beartype.typing import Any
from datasets import Dataset

# https://docs.kidger.site/jaxtyping/api/array/
from jaxtyping import Float, jaxtyped
from sae_lens import SAE, JumpReLUSAE
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer
from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks

# Our libraries
from sae_scoping.utils.hooks.sae import (
    SAEWrapper,
    SAELensEncDecCallbackWrapper,
)
from sae_scoping.trainers.sae_enhanced.utils import frozen_parameters_training
from sae_scoping.trainers.sae_enhanced.defaults_configs import get_default_sft_config, get_default_grpo_config

"""
Train a model with SFT while under hooks. Limit the set of modified parameters to
those after the SAE.
"""


@beartype
def train_sae_enhanced_model(
    train_dataset: Dataset,
    eval_dataset: Dataset | dict[str, Dataset] | None,  # to eval on multiple OOD datasets
    sae: SAE | SAELensEncDecCallbackWrapper | None,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    T: float | int = 0.0,
    hookpoint: str | None = "",
    save_output: bool = False,
    return_trained_model: bool = False,
    # TODO(Adriano) add support for better callbacks
    training_callbacks: list[TrainerCallback] = [],
    trainer_config: SFTConfig | GRPOConfig | None = None,  # None => use default (one below)
    trainer_algorithm: Literal["sft", "grpo"] = "sft",
    **kwargs: dict[str, Any],
) -> PreTrainedModel | None:
    wandb_project_name = kwargs.get("wandb_project_name", os.environ.get("WANDB_PROJECT", None))
    if wandb_project_name is None:
        raise ValueError("WANDB_PROJECT is not set")
    wandb_run_name = kwargs.get("wandb_run_name", os.environ.get("WANDB_RUN_NAME", None))
    old_environ_name = os.environ.get("WANDB_PROJECT", None)
    try:
        # 1. setup trainer arguments
        os.environ["WANDB_PROJECT"] = wandb_project_name
        if wandb_run_name is not None:
            os.environ["WANDB_RUN_NAME"] = wandb_run_name
        if trainer_algorithm == "sft" and trainer_config is None:
            trainer_config = get_default_sft_config(wandb_run_name, kwargs)
        elif trainer_algorithm == "grpo" and trainer_config is None:
            trainer_config = get_default_grpo_config(wandb_run_name, kwargs)

        # 2. Validate SAE/hookpoint combination
        if sae is not None and hookpoint is None:
            raise ValueError("If SAE is provided, then you must also provide a hookpoint")

        # 3. Setup trainer
        trainer_cls = SFTTrainer if trainer_algorithm == "sft" else GRPOTrainer
        trainer_extra_kwargs = {}
        if trainer_algorithm == "grpo":
            reward_funcs = kwargs.pop("reward_funcs", None)
            if reward_funcs is None:
                raise ValueError("If using GRPO, then you must provide a (sequence-level) reward function")
            trainer_extra_kwargs["reward_funcs"] = reward_funcs
        trainer = trainer_cls(
            model=model,
            processing_class=tokenizer,
            args=trainer_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=training_callbacks,
            **trainer_extra_kwargs,
        )

        # 4. Train with parameter freezing and sanity checks
        # NOTE: even if no SAE, you COULD still pass in a hookpoint to limit which layers are trained
        with frozen_parameters_training(model, hookpoint, strict_change_check=(trainer_algorithm == "sft"), n_store=32):
            if sae is not None:
                sae_wrapper = SAEWrapper(sae)
                hook_dict = {hookpoint: partial(filter_hook_fn, sae_wrapper)}
                with named_forward_hooks(model, hook_dict):
                    trainer.train()
            else:
                trainer.train()

        if save_output:
            trainer.save_model()
        if return_trained_model:
            return model

    finally:
        if old_environ_name is not None:
            os.environ["WANDB_PROJECT"] = old_environ_name
