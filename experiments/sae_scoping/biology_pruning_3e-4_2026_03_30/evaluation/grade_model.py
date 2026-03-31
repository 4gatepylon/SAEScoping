"""
grade_model.py

Generate model responses and grade them with LLM judges.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from evaluation.generic_judges import GradedChats, grade_chats
from evaluation.inference.client.model_generator import HFGenerator


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
    try:
        generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
        completed = generator.generate(
            conversations, batch_size=batch_size, generation_kwargs=generation_kwargs
        )
        return grade_chats(completed)
    finally:
        tokenizer.padding_side = original_padding_side
