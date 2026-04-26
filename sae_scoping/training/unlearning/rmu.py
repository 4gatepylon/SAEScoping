"""
Representation Misdirection for Unlearning (RMU).

Instead of operating on the output loss, RMU steers internal hidden-state
representations at specific layers:
  - For forget inputs: push activations toward a random fixed direction
  - For retain inputs: keep activations unchanged (match original model)

    L = MSE(h_l(forget), c * u) + alpha * MSE(h_l(retain), h_l_orig(retain))

where h_l is the hidden state at layer l, u is a random unit vector, and c
is a steering coefficient. Only specific parameters in the targeted layers
are updated (matching the official implementation's param_ids approach).

Reference: Li et al., "The WMDP Benchmark: Measuring and Reducing Malicious
Use With Unlearning" (2024). Code: github.com/centerforaisafety/wmdp

Implementation follows the official WMDP RMU code closely:
- Control vector: torch.rand (uniform [0,1]), not Gaussian
- Full-sequence activations (not mean-pooled)
- Hook layer can differ from updated layers
- Retain loss weighted by alpha (default 100 in official code)
- Only specific param indices updated per layer (default: param 6)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def get_layer_module(model: PreTrainedModel, layer_idx: int) -> nn.Module:
    """Get the transformer layer module by index."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model.layers[layer_idx]
    raise ValueError(f"Cannot find layers in {type(model)}")


def get_num_layers(model: PreTrainedModel) -> int:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return len(model.language_model.model.layers)
    raise ValueError(f"Cannot find layers in {type(model)}")


def get_hidden_size(model: PreTrainedModel) -> int:
    return model.config.hidden_size


def forward_with_cache(model, inputs, module, no_grad=True):
    """Run model forward and capture a layer's output via hook.

    Matches the official WMDP implementation: captures the full activation
    tensor (all tokens) from the specified module's output.
    """
    cache = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None

    handle = module.register_forward_hook(hook)
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
    handle.remove()
    return cache[0]


def get_params(model, layer_ids, param_ids):
    """Get specific parameters from specific layers.

    Matches the official WMDP implementation: only updates param_ids
    within each layer (e.g., param_ids=[6] = mlp.down_proj.weight on
    most architectures).
    """
    layers = (model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers")
              else model.language_model.model.layers)
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params


def unlearn_rmu(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    forget_dataset: Dataset,
    retain_dataset: Dataset,
    hook_layer_id: int | None = None,
    update_layer_ids: list[int] | None = None,
    param_ids: list[int] | None = None,
    steering_coeff: float = 20.0,
    alpha: float = 100.0,
    max_steps: int = 80,
    learning_rate: float = 5e-5,
    max_length: int = 512,
    seed: int = 42,
) -> PreTrainedModel:
    """Run RMU unlearning following the official WMDP implementation.

    Args:
        model: Model to unlearn from (modified in-place).
        tokenizer: Matching tokenizer.
        forget_dataset: Dataset of capabilities to forget ('text' column).
        retain_dataset: Dataset of capabilities to retain ('text' column).
        hook_layer_id: Layer to hook for activation capture. Default: n_layers // 4.
        update_layer_ids: Layers whose params are updated. Default: [hook-2, hook-1, hook].
        param_ids: Which parameter indices within each layer to update.
            Default: [6] (mlp.down_proj.weight on most architectures).
            Use None to update all parameters in the target layers.
        steering_coeff: Scale factor for the random steering vector.
        alpha: Weight for retain loss (default 100, matching official code).
        max_steps: Number of optimization steps.
        learning_rate: Learning rate.
        max_length: Max sequence length for tokenization.
        seed: Random seed for steering vector.

    Returns:
        The model (modified in-place).
    """
    try:
        model_device = model.device
    except AttributeError:
        model_device = next(p.device for p in model.parameters())

    # TODO(Claude) PYTEST-FAILING BUG [RMU-1E13D810]: _get_hidden_size is not defined anywhere
    # in this module. The actual function is get_hidden_size (no underscore, defined at line 56
    # of this file). Same for _get_num_layers → get_num_layers (line 48) on the next line, and
    # _get_layer_module → get_layer_module (line 39) at two call sites further down.
    # All 4 RMU tests crash immediately here before reaching any training logic:
    #   pytest: NameError: name '_get_hidden_size' is not defined
    #   at rmu.py in unlearn_rmu()
    # Affected tests: TestRMU::test_only_update_layers_change,
    #   TestRMU::test_forget_loss_increases, TestRMU::test_model_still_runs,
    #   TestRMU::test_all_params_unfrozen_after
    hidden_size = _get_hidden_size(model)
    n_layers = _get_num_layers(model)

    # Default layer selection (matches WMDP: hook deeper, update a range)
    if hook_layer_id is None:
        hook_layer_id = min(7, n_layers - 1)
    if update_layer_ids is None:
        update_layer_ids = [max(0, hook_layer_id - 2), max(0, hook_layer_id - 1), hook_layer_id]
        update_layer_ids = sorted(set(lid for lid in update_layer_ids if lid < n_layers))

    # Get the hook module
    # TODO(Claude) PYTEST-FAILING BUG [RMU-1E13D810]: _get_layer_module → get_layer_module
    hook_module = _get_layer_module(model, hook_layer_id)

    # Generate random control vector (uniform [0,1], matching official code)
    # Generate on CPU then move to device (torch.Generator doesn't support CUDA)
    rng = torch.Generator().manual_seed(seed)
    control_vec = torch.rand(1, 1, hidden_size, generator=rng, dtype=torch.float32)
    control_vec = (control_vec / torch.norm(control_vec) * steering_coeff).to(dtype=model.dtype, device=model_device)

    # Collect retain activations from the ORIGINAL (frozen) model
    print(f"[rmu] Collecting frozen retain activations at layer {hook_layer_id}...")
    retain_texts = retain_dataset["text"]
    frozen_retain_activations = []
    old_pad = tokenizer.padding_side
    tokenizer.padding_side = "right"

    with torch.no_grad():
        for text in retain_texts:
            tok = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model_device)
            act = forward_with_cache(model, tok, hook_module, no_grad=True)
            frozen_retain_activations.append(act.detach())

    # Freeze all params, then unfreeze only target params
    for p in model.parameters():
        p.requires_grad = False

    if param_ids is not None:
        updated_params = get_params(model, update_layer_ids, param_ids)
    else:
        updated_params = []
        for lid in update_layer_ids:
            # TODO(Claude) PYTEST-FAILING BUG [RMU-1E13D810]: _get_layer_module → get_layer_module
            updated_params.extend(_get_layer_module(model, lid).parameters())

    for p in updated_params:
        p.requires_grad = True

    n_updated = sum(p.numel() for p in updated_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[rmu] Hook layer: {hook_layer_id}, Update layers: {update_layer_ids}, "
          f"Params: {n_updated:,} / {n_total:,}")

    optimizer = torch.optim.AdamW(updated_params, lr=learning_rate)

    # Training loop
    forget_texts = forget_dataset["text"]
    n_forget = len(forget_texts)
    n_retain = len(retain_texts)

    model.train()
    for step in tqdm(range(max_steps), desc="[rmu] training"):
        optimizer.zero_grad()

        # Forget: push activations toward control vector
        f_idx = step % n_forget
        f_tok = tokenizer(
            forget_texts[f_idx], return_tensors="pt",
            truncation=True, max_length=max_length,
        ).to(model_device)
        forget_activations = forward_with_cache(model, f_tok, hook_module, no_grad=False)
        unlearn_loss = F.mse_loss(forget_activations, control_vec)

        # Retain: keep activations close to frozen model
        r_idx = step % n_retain
        r_tok = tokenizer(
            retain_texts[r_idx], return_tensors="pt",
            truncation=True, max_length=max_length,
        ).to(model_device)
        retain_activations = forward_with_cache(model, r_tok, hook_module, no_grad=False)
        retain_loss = F.mse_loss(retain_activations, frozen_retain_activations[r_idx])

        loss = unlearn_loss + alpha * retain_loss
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"  step {step}: unlearn={unlearn_loss.item():.4f}, "
                  f"retain={retain_loss.item():.4f}, total={loss.item():.4f}")

    tokenizer.padding_side = old_pad

    # Unfreeze all params
    for p in model.parameters():
        p.requires_grad = True

    print(f"[rmu] Unlearning complete ({max_steps} steps)")
    return model
