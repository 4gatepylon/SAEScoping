"""
Negative Preference Optimization (NPO) for unlearning.

Treats forget-set completions as "rejected" responses using a DPO-style
bounded loss, avoiding the gradient explosion of naive gradient ascent.

    L_NPO = -(2/beta) * E[log sigmoid(-beta * log(p_theta(y|x) / p_ref(y|x)))]

where p_ref is the frozen original model. Optionally combined with KL
retention on the retain set (NPO+KL).

Reference: Zhang et al., "Negative Preference Optimization: From
Catastrophic Collapse to Effective Unlearning" (NeurIPS 2024).
Also: Fan et al., "Simplicity Prevails: Rethinking Negative Preference
Optimization for LLM Unlearning" (SimNPO, NeurIPS 2025).
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from trl import SFTConfig, SFTTrainer


def _compute_log_probs(model, input_ids, attention_mask, labels):
    """Compute per-token log probabilities for the given labels."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # (B, T-1, V)
    shift_labels = labels[:, 1:]  # (B, T-1)

    log_probs = F.log_softmax(logits, dim=-1)
    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)  # (B, T-1)

    # Mask out padding (labels == -100)
    valid_mask = (shift_labels != -100).float()
    # Sum log probs per sequence, normalized by length
    seq_log_probs = (token_log_probs * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
    return seq_log_probs


class NPOTrainer(SFTTrainer):
    """Trainer implementing NPO unlearning loss.

    Computes the bounded DPO-style loss on forget data, optionally with
    KL retention on retain data.
    """

    def __init__(
        self,
        ref_model: PreTrainedModel,
        retain_dataset: Optional[Dataset] = None,
        npo_beta: float = 0.1,
        retain_weight: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.retain_dataset = retain_dataset
        self.npo_beta = npo_beta
        self.retain_weight = retain_weight
        self._retain_loader = None
        self._retain_iter = None

    def _get_retain_batch(self):
        if self.retain_dataset is None:
            return None
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
        inputs["use_cache"] = False
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs.get("labels", input_ids.clone())

        # Log probs under current model and reference model
        current_log_probs = _compute_log_probs(model, input_ids, attention_mask, labels)
        with torch.no_grad():
            ref_log_probs = _compute_log_probs(self.ref_model, input_ids, attention_mask, labels)

        # NPO loss: -log(sigmoid(-beta * (current - ref)))
        # This is the "rejected" arm of DPO — pushes current away from ref on forget data
        log_ratio = current_log_probs - ref_log_probs
        npo_loss = -(2.0 / self.npo_beta) * F.logsigmoid(-self.npo_beta * log_ratio).mean()

        loss = npo_loss

        # Optional KL retention on retain set
        if self.retain_dataset is not None:
            retain_batch = self._get_retain_batch()
            if retain_batch is not None:
                retain_batch = {k: v.to(model.device) for k, v in retain_batch.items()}
                retain_batch["use_cache"] = False
                retain_outputs = model(**retain_batch)
                retain_loss = retain_outputs.loss
                loss = loss + self.retain_weight * retain_loss

        if return_outputs:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            return loss, outputs
        return loss


def unlearn_npo(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    forget_dataset: Dataset,
    retain_dataset: Optional[Dataset] = None,
    npo_beta: float = 0.1,
    retain_weight: float = 1.0,
    max_steps: int = 200,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    max_length: int = 1024,
    output_dir: str = "/tmp/sae_scoping_npo_unlearn",
    callbacks: list[TrainerCallback] | None = None,
    report_to: str = "none",
) -> PreTrainedModel:
    """Run NPO unlearning.

    Args:
        model: Model to unlearn from (modified in-place).
        tokenizer: Matching tokenizer.
        forget_dataset: Dataset of capabilities to forget ('text' column).
        retain_dataset: Optional retain dataset ('text' column). If provided,
            adds KL retention loss (NPO+KL variant).
        npo_beta: NPO temperature parameter. Lower = stronger unlearning.
        retain_weight: Weight for the retain loss term.
        max_steps: Training steps.
        learning_rate: Learning rate.
        batch_size: Batch size.
        max_length: Max sequence length.
        output_dir: Trainer output directory.
        callbacks: Optional trainer callbacks.
        report_to: Logging backend.

    Returns:
        The model (modified in-place).
    """
    # Create frozen reference model
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

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

    trainer = NPOTrainer(
        ref_model=ref_model,
        retain_dataset=retain_dataset,
        npo_beta=npo_beta,
        retain_weight=retain_weight,
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=forget_dataset,
        callbacks=callbacks or [],
    )

    trainer.train()

    # Free reference model
    del ref_model
    torch.cuda.empty_cache()

    return model
