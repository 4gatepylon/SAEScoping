from __future__ import annotations
import torch
from transformers import AutoTokenizer


class DummyHFModel:
    """
    Dummy HF model that returns hardcoded responses based on input content.
    Mimics the interface needed by OneClickLLMJudgeEvaluation.
    """

    def __init__(self, device: str = "cpu"):
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    def parameters(self):
        """Yield a dummy parameter for device detection."""
        yield torch.nn.Parameter(torch.zeros(1, device=self._device))

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        max_new_tokens: int | None = None,
        max_length: int | None = None,
        min_new_tokens: int | None = None,
        min_length: int | None = None,
        **kwargs,  # early_stopping, max_time, stop_strings, etc. - ignored for mock
    ) -> torch.Tensor:
        """
        Generate responses by returning hardcoded token sequences.
        Returns input_ids concatenated with response tokens.

        Supports HuggingFace generate kwargs:
        - max_new_tokens: Maximum number of new tokens to generate (preferred)
        - max_length: Maximum total sequence length (input + output)
        - min_new_tokens: Minimum number of new tokens to generate
        - min_length: Minimum total sequence length

        Other kwargs (early_stopping, max_time, stop_strings, etc.) are accepted
        but ignored since this is a mock model.
        """
        batch_size = input_ids.shape[0]
        input_length = input_ids.shape[1]

        # Determine number of new tokens to generate
        # Priority: max_new_tokens > max_length > default (50)
        if max_new_tokens is not None:
            num_new_tokens = max_new_tokens
        elif max_length is not None:
            num_new_tokens = max(0, max_length - input_length)
        else:
            num_new_tokens = 50  # default

        # Apply minimum constraints (not robust to bad arguments obv. tbh)
        if min_new_tokens is not None:
            num_new_tokens = max(num_new_tokens, min_new_tokens)
        if min_length is not None:
            min_from_length = max(0, min_length - input_length)
            num_new_tokens = max(num_new_tokens, min_from_length)

        # Generate dummy response tokens (using token id 1 as placeholder)
        dummy_response_tokens = torch.ones(batch_size, num_new_tokens, dtype=torch.long, device=self._device)
        return torch.cat([input_ids, dummy_response_tokens], dim=1)

    def eval(self):
        return self


def create_dummy_model_and_tokenizer(device: str = "cpu"):
    """Create dummy model and tokenizer for testing."""
    # Use llama3.2 tokenizer as base
    base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    base_tokenizer.padding_side = "left"

    model = DummyHFModel(device=device)
    return model, base_tokenizer
