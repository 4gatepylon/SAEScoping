"""
utils.py

Shared evaluation utilities for the saliency pruning pipeline.
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
