from __future__ import annotations
from typing import Any
from trl import GRPOConfig, SFTConfig

def get_default_grpo_config(
    wandb_run_name: str,
    kwargs: dict[str, Any],
) -> GRPOConfig:
    return GRPOConfig(
        run_name=wandb_run_name,
        output_dir=kwargs.get("output_dir", "./deleteme_grpo_output"),
        per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=kwargs.get("gradient_accumulation_steps", 4),
        num_train_epochs=kwargs.get("num_train_epochs", 1),
        learning_rate=kwargs.get("learning_rate", 2e-6),
        warmup_ratio=kwargs.get("warmup_ratio", 0.1),
        weight_decay=kwargs.get("weight_decay", 0.01),
        max_grad_norm=kwargs.get("max_grad_norm", 1.0),
        lr_scheduler_type="cosine",
        save_steps=kwargs.get("save_steps", 1_000_000),  # Do not intend to save
        save_strategy="no",
        logging_steps=kwargs.get("logging_steps", 10),
        fp16=False,
        bf16=True,  # H100/A100 hopefully will work here? Llama2
        remove_unused_columns=False,
        save_total_limit=2,
        report_to="wandb",
        max_steps=kwargs.get("max_steps", 1_000),
        # TODO(Adriano) please note that for some mysterious reason eval is not supporeted during
        # GRPOTrainer WanDB training, so we may want to support or add callbacks for evals (for
        # verifiable rewards should be pretty easy to do this ngl)
        # GRPO-specific parameters
        num_generations=kwargs.get("num_generations", 8),  # Completions per prompt
        max_completion_length=kwargs.get("max_completion_length", 1024),
        max_prompt_length=kwargs.get("max_prompt_length", 1024),
        temperature=kwargs.get("temperature", 1.0),
        # TODO(Adriano) don't hardcode
        gradient_checkpointing=False,
    )


def default_sft_config(
    wandb_run_name: str,
    kwargs: dict[str, Any],
) -> SFTConfig:
    return SFTConfig(
        run_name=wandb_run_name,  # None => use default
        output_dir=kwargs.get("output_dir", "./deleteme_sft_output"),
        per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.1,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        save_steps=kwargs.get("save_steps", 1_000_000),  # Do not intend to save
        save_strategy="no",  # Do not intend to save
        logging_steps=10,
        fp16=False,
        bf16=True,  # H100/A100 hopefully will work here? Llama2
        remove_unused_columns=False,
        eval_strategy="steps",
        eval_steps=100,  # wanna do this somewhat often, but not tooo much
        save_total_limit=2,
        # load_best_model_at_end=True, # <- can't do this w/out matching save/eval strat
        # metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        max_steps=kwargs.get("max_steps", 1_000),
        max_length=kwargs.get("context_length", 1024),
        # TODO(adriano) don't hardcode
        gradient_checkpointing=False,
    )