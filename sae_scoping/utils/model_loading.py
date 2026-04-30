"""Model loading for supported Gemma architectures."""

from __future__ import annotations

from itertools import product

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

_GEMMA2_SIZES = ("2b", "9b")
_GEMMA3_SIZES = ("4b", "12b")
_SUFFIXES = ("", "-it")

SUPPORTED_MODELS: frozenset[str] = frozenset(
    f"google/gemma-{gen}-{size}{suffix}" for gen, sizes in [("2", _GEMMA2_SIZES), ("3", _GEMMA3_SIZES)] for size, suffix in product(sizes, _SUFFIXES)
)


def _validate_model_id(model_id: str) -> None:
    if model_id not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_id}. Supported models:\n  " + "\n  ".join(sorted(SUPPORTED_MODELS)))


def load_model_and_tokenizer(
    model_id: str,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a Gemma 2 or 3 model and tokenizer with eager attention.

    Raises ValueError for unsupported model families.
    """
    _validate_model_id(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device, attn_implementation="eager")
    return model, tokenizer
