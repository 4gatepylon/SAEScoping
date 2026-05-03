"""Batched cross-entropy loss computation for language models."""

from __future__ import annotations

import tqdm
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


# TODO(hadriano) this should probably be in a utilities file somewhere
@torch.no_grad()
def compute_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_seq_len: int = 1024,
    batch_size: int = 2,
) -> float:
    """Compute mean per-batch cross-entropy loss on a list of texts.

    Args:
        model: A causal LM (already on device).
        tokenizer: Matching tokenizer.
        texts: Pre-formatted text strings.
        max_seq_len: Truncation length.
        batch_size: Texts per forward pass.

    Returns:
        Mean loss across batches, or 0.0 if texts is empty.
    """
    was_training = model.training
    model.eval()
    try:
        device = model.device
    except AttributeError:
        device = next(p.device for p in model.parameters())
    old_pad = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        total, n = 0.0, 0
        for i in tqdm.trange(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            tok = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            ids = tok["input_ids"].to(device)
            mask = tok["attention_mask"].to(device)
            labels = ids.clone()
            labels[mask == 0] = -100
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            total += out.loss.item()
            n += 1
        return total / max(n, 1)
    finally:
        tokenizer.padding_side = old_pad
        if was_training:
            model.train()


_WANDA_SKIP_LEAF_NAMES = {"lm_head", "embed_tokens", "embed_out"}


def count_zeros(model: PreTrainedModel, wanda_prunable_only: bool = False) -> tuple[int, int]:
    """Count zero and total elements across parameters.

    Args:
        model: The model to inspect.
        wanda_prunable_only: If True, only count nn.Linear weight tensors
            that Wanda would actually prune (skipping lm_head, embed_tokens,
            embed_out — same filter as wanda._find_linear_layers).

    Returns:
        (n_zeros, n_total)
    """
    if wanda_prunable_only:
        linear_weight_names: set[str] = set()
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and not any(s in name for s in _WANDA_SKIP_LEAF_NAMES):
                linear_weight_names.add(f"{name}.weight")

    total, zeros = 0, 0
    for name, p in model.named_parameters():
        if wanda_prunable_only and name not in linear_weight_names:
            continue
        total += p.numel()
        zeros += (p.data == 0).sum().item()
    return int(zeros), total
