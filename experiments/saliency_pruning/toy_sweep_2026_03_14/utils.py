"""
utils.py

Shared evaluation utilities for the saliency pruning pipeline.

Public API
----------
compute_validation_loss   – mean cross-entropy loss over formatted texts
generate_and_grade        – generate responses, then LLM-judge them
is_metric_passing         – directional threshold check (loss ≤ t  or  judge ≥ t)
is_metric_better          – directional improvement check
evaluate_model            – unified entry point: loss or judge, returns a float
"""

from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from grade_chats import GradedChats, grade_chats
from model_generator import HFGenerator


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_validation_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    batch_size: int = 4,
    max_seq_len: int = 1024,
) -> float:
    """
    Mean cross-entropy loss over all provided texts.

    Args:
        model: The model to evaluate.
        tokenizer: Tokenizer matching the model.
        texts: Pre-formatted text strings (e.g. from dataset_utils.format_as_sft_text).
        batch_size: Batch size for loss computation.
        max_seq_len: Maximum sequence length for tokenization.

    Returns:
        Mean loss across all batches.
    """
    model.eval()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    total_loss = 0.0
    n_batches = 0
    for i in tqdm(range(0, len(texts), batch_size), desc="  loss batches", leave=False):
        batch = texts[i : i + batch_size]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = tokenized["input_ids"].to(model.device)
        attention_mask = tokenized["attention_mask"].to(model.device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        n_batches += 1
    tokenizer.padding_side = original_padding_side
    return total_loss / n_batches if n_batches > 0 else float("nan")


# ---------------------------------------------------------------------------
# Generation + grading
# ---------------------------------------------------------------------------


def generate_and_grade(
    generator: HFGenerator,
    tokenizer: PreTrainedTokenizerBase,
    conversations: list[list[dict]],
    batch_size: int = 4,
    max_new_tokens: int = 256,
) -> GradedChats:
    """
    Generate responses then grade with LLM judges.

    Args:
        generator: An initialized HFGenerator.
        tokenizer: Tokenizer (padding_side will be set to "left").
        conversations: 0-turn OpenAI conversations (question only).
        batch_size: Batch size for generation.
        max_new_tokens: Max tokens to generate.

    Returns:
        GradedChats with per-judge scores.
    """
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    completed = generator.generate(
        conversations, batch_size=batch_size, generation_kwargs=generation_kwargs
    )
    tokenizer.padding_side = original_padding_side
    return grade_chats(completed)


# ---------------------------------------------------------------------------
# Metric helpers (shared by prune_and_maybe_recover and sweep)
# ---------------------------------------------------------------------------


def is_metric_passing(metric: float, metric_type: str, threshold: float) -> bool:
    """Return True if the metric meets the quality threshold.

    For loss  (lower=better): passes when metric <= threshold.
    For judge (higher=better): passes when metric >= threshold.
    """
    if metric_type == "loss":
        return metric <= threshold
    return metric >= threshold


def is_metric_better(new_metric: float, old_metric: float, metric_type: str) -> bool:
    """Return True if new_metric is strictly better than old_metric."""
    if metric_type == "loss":
        return new_metric < old_metric
    return new_metric > old_metric


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    metric_type: str,
    eval_texts: list[str],
    eval_conversations: list[list[dict]],
    batch_size: int,
    max_seq_len: int,
    max_new_tokens: int,
) -> float:
    """Evaluate a model using either cross-entropy loss or LLM judge score.

    Args:
        model: Model to evaluate (will be set to eval mode).
        tokenizer: Tokenizer matching the model.
        metric_type: ``"loss"`` or ``"judge"``.
        eval_texts: Pre-formatted SFT strings (used by loss path).
        eval_conversations: 0-turn OpenAI conversations (used by judge path).
        batch_size: Batch size for computation.
        max_seq_len: Max sequence length (loss path).
        max_new_tokens: Max generation tokens (judge path).

    Returns:
        Scalar metric value.
    """
    if metric_type == "loss":
        return compute_validation_loss(
            model, tokenizer, eval_texts,
            batch_size=batch_size, max_seq_len=max_seq_len,
        )
    if metric_type == "judge":
        generator = HFGenerator(model, tokenizer)
        graded = generate_and_grade(
            generator, tokenizer, eval_conversations,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
        )
        return graded.overall_mean_score
    raise ValueError(f"Unknown metric_type '{metric_type}'. Choose 'loss' or 'judge'.")
