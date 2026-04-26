"""
Gradient Difference (GD) unlearning.

The simplest meaningful unlearning baseline: gradient ascent on the forget
set combined with gradient descent on the retain set.

    L = -alpha * L_CE(forget) + beta * L_CE(retain)

This pushes the model to perform worse on forget-domain data while
maintaining performance on retain-domain data.

Reference: Used as baseline in virtually all LLM unlearning papers.
See also: TOFU benchmark (Maini et al., 2024).
"""

from __future__ import annotations

from typing import Optional

import torch
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from trl import SFTConfig, SFTTrainer


class GradientDiffTrainer(SFTTrainer):
    """SFTTrainer that alternates between GA on forget data and GD on retain data.

    Each training step:
    1. Compute L_CE on the current batch (from the forget set)
    2. Negate the loss (gradient ascent)
    3. Add a retain loss term from a retain data iterator

    The retain dataset is iterated in lockstep with the forget dataset.
    When it's exhausted, it wraps around.
    """

    def __init__(
        self,
        retain_dataset: Dataset,
        forget_weight: float = 1.0,
        retain_weight: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.retain_dataset = retain_dataset
        self.forget_weight = forget_weight
        self.retain_weight = retain_weight
        self._retain_loader = None
        self._retain_iter = None

    def _get_retain_batch(self):
        """Get the next retain batch, cycling if needed."""
        if self._retain_loader is None:
            self._retain_loader = self.get_eval_dataloader(self.retain_dataset)
        if self._retain_iter is None:
            self._retain_iter = iter(self._retain_loader)
        try:
            batch = next(self._retain_iter)
        except StopIteration:
            self._retain_iter = iter(self._retain_loader)
            batch = next(self._retain_iter)
        return batch

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forget loss (negate for gradient ascent)
        inputs["use_cache"] = False
        forget_outputs = model(**inputs)
        forget_loss = forget_outputs.loss

        # Retain loss (normal gradient descent)
        retain_batch = self._get_retain_batch()
        retain_batch = {k: v.to(model.device) for k, v in retain_batch.items()}
        retain_batch["use_cache"] = False
        retain_outputs = model(**retain_batch)
        retain_loss = retain_outputs.loss

        # Combined: ascend on forget, descend on retain
        loss = -self.forget_weight * forget_loss + self.retain_weight * retain_loss

        if return_outputs:
            return loss, forget_outputs
        return loss


def unlearn_gradient_diff(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    forget_dataset: Dataset,
    retain_dataset: Dataset,
    forget_weight: float = 1.0,
    retain_weight: float = 1.0,
    max_steps: int = 200,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    max_length: int = 1024,
    output_dir: str = "/tmp/sae_scoping_gd_unlearn",
    callbacks: list[TrainerCallback] | None = None,
    report_to: str = "none",
) -> PreTrainedModel:
    """Run Gradient Difference unlearning.

    Args:
        model: Model to unlearn from (modified in-place).
        tokenizer: Matching tokenizer.
        forget_dataset: Dataset of capabilities to forget (must have 'text' column).
        retain_dataset: Dataset of capabilities to retain (must have 'text' column).
        forget_weight: Weight for the forget (gradient ascent) loss.
        retain_weight: Weight for the retain (gradient descent) loss.
        max_steps: Number of training steps.
        learning_rate: Learning rate.
        batch_size: Batch size.
        max_length: Max sequence length.
        output_dir: Output directory for trainer artifacts.
        callbacks: Optional trainer callbacks.
        report_to: Logging backend ("wandb", "none", etc).

    Returns:
        The model (modified in-place).
    """
    # TODO(Claude) PYTEST-FAILING BUG [MPS-EF1E4D7F]: SFTConfig does not disable MPS, so on
    # Mac the TRL trainer silently moves the model from CPU to mps:0 during .train(). After
    # training completes, the model stays on MPS. Any post-training code that creates new
    # tensors on CPU (e.g. tokenizer output) and passes them to the model hits:
    #   pytest: RuntimeError: Placeholder storage has not been allocated on MPS device!
    #   at torch/nn/functional.py:2551 in embedding()
    # Affected tests: TestGradientDiff::test_forget_loss_increases,
    #   TestGradientDiff::test_model_still_runs
    # Fix: add use_cpu=True when not torch.cuda.is_available(), or move model back to CPU
    # after trainer.train().
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        bf16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="no",
        report_to=report_to,
        max_length=max_length,
        gradient_accumulation_steps=1,
        dataset_text_field="text",
        remove_unused_columns=False,
    )

    trainer = GradientDiffTrainer(
        retain_dataset=retain_dataset,
        forget_weight=forget_weight,
        retain_weight=retain_weight,
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=forget_dataset,
        callbacks=callbacks or [],
    )

    trainer.train()
    return model
