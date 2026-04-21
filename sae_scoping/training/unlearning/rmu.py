"""
Representation Misdirection for Unlearning (RMU).

Instead of operating on the output loss, RMU steers internal hidden-state
representations at specific layers:
  - For forget inputs: push activations toward a random fixed direction
  - For retain inputs: keep activations unchanged (match original model)

    L = alpha * MSE(h_l(forget), c * u) + beta * MSE(h_l(retain), h_l_orig(retain))

where h_l is the hidden state at layer l, u is a random unit vector, and c
is a steering coefficient. Only the weights of the targeted layer(s) are
updated.

Reference: Li et al., "The WMDP Benchmark: Measuring and Reducing Malicious
Use With Unlearning" (2024). Code: github.com/centerforaisafety/wmdp
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def _get_layer_module(model: PreTrainedModel, layer_idx: int) -> nn.Module:
    """Get the transformer layer module by index."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model.layers[layer_idx]
    raise ValueError(f"Cannot find layers in {type(model)}")


def _get_num_layers(model: PreTrainedModel) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return len(model.language_model.model.layers)
    raise ValueError(f"Cannot find layers in {type(model)}")


def _get_hidden_size(model: PreTrainedModel) -> int:
    return model.config.hidden_size


class _ActivationCapture:
    """Forward hook that captures the output hidden states of a layer."""

    def __init__(self):
        self.activations: torch.Tensor | None = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            self.activations = output[0]
        else:
            self.activations = output


def _collect_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    layer_idx: int,
    max_seq_len: int = 1024,
    batch_size: int = 4,
) -> torch.Tensor:
    """Run texts through model and collect mean hidden state at a layer.

    Returns: (n_texts, hidden_size) tensor of mean-pooled activations.
    """
    layer = _get_layer_module(model, layer_idx)
    capture = _ActivationCapture()
    handle = layer.register_forward_hook(capture)

    try:
        model_device = model.device
    except AttributeError:
        model_device = next(p.device for p in model.parameters())

    all_acts = []
    old_pad = tokenizer.padding_side
    tokenizer.padding_side = "right"

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            tok = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_seq_len,
            )
            input_ids = tok["input_ids"].to(model_device)
            attention_mask = tok["attention_mask"].to(model_device)
            model(input_ids=input_ids, attention_mask=attention_mask)

            # Mean pool over sequence (mask out padding)
            acts = capture.activations  # (batch, seq, hidden)
            mask_expanded = attention_mask.unsqueeze(-1).float()
            mean_acts = (acts * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            all_acts.append(mean_acts.cpu())

    tokenizer.padding_side = old_pad
    handle.remove()
    return torch.cat(all_acts, dim=0)


def unlearn_rmu(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    forget_dataset: Dataset,
    retain_dataset: Dataset,
    layer_ids: list[int] | None = None,
    steering_coeff: float = 20.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    max_steps: int = 200,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    max_length: int = 1024,
    seed: int = 42,
) -> PreTrainedModel:
    """Run RMU unlearning.

    Args:
        model: Model to unlearn from (modified in-place).
        tokenizer: Matching tokenizer.
        forget_dataset: Dataset of capabilities to forget ('text' column).
        retain_dataset: Dataset of capabilities to retain ('text' column).
        layer_ids: Which layers to apply RMU to. Default: middle layer.
        steering_coeff: Scale factor c for the random steering vector.
            Model-dependent: ~6.5 for 7B, ~20 for 2B, ~100+ for larger.
        alpha: Weight for forget loss.
        beta: Weight for retain loss.
        max_steps: Number of optimization steps.
        learning_rate: Learning rate for updated layers.
        batch_size: Batch size for activation collection.
        max_length: Max sequence length.
        seed: Random seed for steering vector.

    Returns:
        The model (modified in-place).
    """
    try:
        model_device = model.device
    except AttributeError:
        model_device = next(p.device for p in model.parameters())

    hidden_size = _get_hidden_size(model)
    n_layers = _get_num_layers(model)

    # Default: use middle layer
    if layer_ids is None:
        layer_ids = [n_layers // 2]

    # Generate random steering vectors (one per layer, fixed for duration)
    rng = torch.Generator().manual_seed(seed)
    steering_vectors = {}
    for lid in layer_ids:
        u = torch.randn(hidden_size, generator=rng)
        u = u / u.norm()  # Unit vector
        steering_vectors[lid] = (u * steering_coeff).to(model_device)

    # Collect retain activations from the ORIGINAL model (freeze target)
    print(f"[rmu] Collecting retain activations at layers {layer_ids}...")
    retain_texts = retain_dataset["text"]
    retain_targets = {}
    with torch.no_grad():
        for lid in layer_ids:
            acts = _collect_activations(
                model, tokenizer, retain_texts, lid,
                max_seq_len=max_length, batch_size=batch_size,
            )
            retain_targets[lid] = acts.to(model_device)

    # Freeze everything except target layers
    for name, param in model.named_parameters():
        param.requires_grad = False
    for lid in layer_ids:
        layer_module = _get_layer_module(model, lid)
        for param in layer_module.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[rmu] Training {trainable:,} / {total:,} parameters in layers {layer_ids}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    # Prepare data
    forget_texts = forget_dataset["text"]
    n_forget = len(forget_texts)
    n_retain = len(retain_texts)

    # Register hooks for target layers
    captures = {}
    handles = []
    for lid in layer_ids:
        cap = _ActivationCapture()
        captures[lid] = cap
        handles.append(_get_layer_module(model, lid).register_forward_hook(cap))

    old_pad = tokenizer.padding_side
    tokenizer.padding_side = "right"

    try:
        model.train()
        for step in tqdm(range(max_steps), desc="[rmu] training"):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=model_device)

            # --- Forget loss: push activations toward random vector ---
            f_idx = step % n_forget
            f_tok = tokenizer(
                forget_texts[f_idx], return_tensors="pt",
                truncation=True, max_length=max_length,
            )
            f_ids = f_tok["input_ids"].to(model_device)
            f_mask = f_tok["attention_mask"].to(model_device)
            model(input_ids=f_ids, attention_mask=f_mask)

            for lid in layer_ids:
                acts = captures[lid].activations  # (1, seq, hidden)
                mask_exp = f_mask.unsqueeze(-1).float()
                mean_act = (acts * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
                target = steering_vectors[lid].unsqueeze(0)
                total_loss = total_loss + alpha * F.mse_loss(mean_act, target)

            # --- Retain loss: keep activations close to original ---
            r_idx = step % n_retain
            r_tok = tokenizer(
                retain_texts[r_idx], return_tensors="pt",
                truncation=True, max_length=max_length,
            )
            r_ids = r_tok["input_ids"].to(model_device)
            r_mask = r_tok["attention_mask"].to(model_device)
            model(input_ids=r_ids, attention_mask=r_mask)

            for lid in layer_ids:
                acts = captures[lid].activations
                mask_exp = r_mask.unsqueeze(-1).float()
                mean_act = (acts * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
                retain_target = retain_targets[lid][r_idx].unsqueeze(0)
                total_loss = total_loss + beta * F.mse_loss(mean_act, retain_target)

            total_loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"  step {step}: loss={total_loss.item():.4f}")

    finally:
        tokenizer.padding_side = old_pad
        for h in handles:
            h.remove()

    # Unfreeze all params
    for param in model.parameters():
        param.requires_grad = True

    print(f"[rmu] Unlearning complete ({max_steps} steps)")
    return model
