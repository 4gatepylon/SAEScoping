"""
SparseLLM: Towards Global Pruning for Pre-trained Language Models.

Implements the SparseLLM pruning method from:
  Bai et al., 2024 (arXiv:2402.17946, NeurIPS 2024)

SparseLLM bridges local and global pruning by introducing auxiliary
variables that link consecutive FFN layers. Attention layers are pruned
locally (standard magnitude), while FFN layers undergo iterative
alternating optimization:

  1. Update W (closed-form least squares)
  2. Prune W (magnitude-based, keeping top-(1-sparsity) per row)
  3. Update p (intermediate activation, ridge regression)
  4. Update z (pre-activation, closed-form using SiLU derivative)
  5. Update s (gate activation, gradient descent — LLaMA/Gemma SwiGLU only)

This implementation targets Gemma 2/3 models which use SwiGLU FFN:
  FFN(x) = down_proj(silu(gate_proj(x)) * up_proj(x))

The module outputs 0/1 masks as a saliency map (compatible with
weight_pruning.py when thresholding at 0.5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


# ---------------------------------------------------------------------------
# Layer input/output capture
# ---------------------------------------------------------------------------


class _LayerInputCapture:
    """Captures inputs to a transformer layer during forward pass."""

    def __init__(self):
        self.inputs: list[torch.Tensor] = []

    def __call__(self, module, args, kwargs):
        # Transformer layers typically receive hidden_states as first positional arg
        if isinstance(args, tuple) and len(args) > 0:
            self.inputs.append(args[0].detach())
        return None


class _LinearInputCapture:
    """Forward hook to capture inputs to a linear layer."""

    def __init__(self):
        self.inputs: list[torch.Tensor] = []

    def __call__(self, module, input, output):
        inp = input[0].detach()
        if inp.ndim == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        self.inputs.append(inp)


# ---------------------------------------------------------------------------
# Hessian computation
# ---------------------------------------------------------------------------


def _compute_hessian(inputs: torch.Tensor, damping: float = 1e-4) -> torch.Tensor:
    """Compute H = X^T X / n + damping * I for a set of inputs.

    Args:
        inputs: (n_tokens, d_in) input activation matrix.
        damping: Regularization term.

    Returns:
        (d_in, d_in) Hessian approximation.
    """
    n = inputs.shape[0]
    H = inputs.T @ inputs / n
    H += damping * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
    return H


# ---------------------------------------------------------------------------
# Local pruning for attention layers (magnitude-based per-row)
# ---------------------------------------------------------------------------


def _prune_local_per_row(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Per-row magnitude pruning. Returns a 0/1 mask."""
    if sparsity <= 0:
        return torch.ones_like(weight)
    n_cols = weight.shape[1]
    n_prune = int(n_cols * sparsity)
    if n_prune == 0:
        return torch.ones_like(weight)
    scores = weight.abs()
    _, sorted_idx = torch.sort(scores, dim=1)
    mask = torch.ones_like(weight)
    mask.scatter_(1, sorted_idx[:, :n_prune], 0.0)
    return mask


# ---------------------------------------------------------------------------
# SparseLLM FFN optimization (SwiGLU variant for Gemma/LLaMA)
# ---------------------------------------------------------------------------


def _sparse_llm_ffn_swiglu(
    W_up: torch.Tensor,       # (d_ffn, d_model) — up_proj weight
    W_gate: torch.Tensor,     # (d_ffn, d_model) — gate_proj weight
    W_down: torch.Tensor,     # (d_model, d_ffn) — down_proj weight
    X: torch.Tensor,          # (n_tokens, d_model) — layer input
    Y: torch.Tensor,          # (n_tokens, d_model) — layer output
    sparsity: float,
    n_iterations: int = 4,
    alpha: float = 5.0,
    beta: float = 5.0,
    gamma: float = 5.0,
    s_lr: float = 0.01,
    s_epochs: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run SparseLLM alternating optimization on a SwiGLU FFN block.

    Returns (mask_up, mask_gate, mask_down) — 0/1 float tensors.
    """
    device = W_up.device
    dtype = torch.float32
    W_up = W_up.float()
    W_gate = W_gate.float()
    W_down = W_down.float()
    X = X.float().to(device)
    Y = Y.float().to(device)

    # Initialize auxiliary variables from unpruned forward pass
    z = (W_up @ X.T).T           # (n, d_ffn) — up_proj output
    s = (W_gate @ X.T).T         # (n, d_ffn) — gate_proj output
    p = F.silu(s) * z            # (n, d_ffn) — down_proj input
    # m = z for up_proj pre-activation reference
    m_up = z.clone()
    m_gate = s.clone()

    # Pseudo-inverse of X, computed once
    Xinv = torch.linalg.pinv(X)  # (d_model, n)

    for it in range(n_iterations):
        # --- Step 1: Update W (closed-form least squares) ---
        # W_up has shape (d_ffn, d_model), z.T is (d_ffn, n), Xinv.T is (n, d_model)
        W_up = z.T @ Xinv.T              # z = W_up @ X => W_up = z^T @ X^+^T
        W_gate = s.T @ Xinv.T            # s = W_gate @ X => W_gate = s^T @ X^+^T
        # W_down has shape (d_model, d_ffn), Y.T is (d_model, n), pinv(p) is (n, d_ffn)^+ = (d_ffn, n)
        pinv_p = torch.linalg.pinv(p)  # (d_ffn, n)
        W_down = Y.T @ pinv_p.T           # (d_model, n) @ (n, d_ffn) = (d_model, d_ffn)

        # --- Step 2: Prune W (per-row magnitude) ---
        mask_up = _prune_local_per_row(W_up, sparsity)
        mask_gate = _prune_local_per_row(W_gate, sparsity)
        mask_down = _prune_local_per_row(W_down, sparsity)
        W_up = W_up * mask_up
        W_gate = W_gate * mask_gate
        W_down = W_down * mask_down

        # --- Step 3: Update p (ridge regression for down_proj input) ---
        # p = argmin beta * ||Y - W_down @ p||^2 + gamma * ||p - silu(s)*z||^2
        WdtWd = W_down.T @ W_down                      # (d_ffn, d_ffn)
        regularizer = gamma * torch.eye(WdtWd.shape[0], device=device)
        A = beta * WdtWd + regularizer
        silu_s = F.silu(s)
        b = (beta * (W_down.T @ Y.T) + gamma * (silu_s * z).T)  # (d_ffn, n)
        p = torch.linalg.solve(A, b).T                 # (n, d_ffn)

        # --- Step 4: Update z (closed-form using SiLU) ---
        # z = argmin alpha * ||z - W_up @ X||^2 + gamma * ||p - silu(s)*z||^2
        m_up = (W_up @ X.T).T
        silu_s = F.silu(s)
        # Approximate: z = (alpha * m_up + gamma * silu_s * p) / (alpha + gamma * silu_s^2)
        z = (alpha * m_up + gamma * silu_s * p) / (alpha + gamma * silu_s ** 2 + 1e-8)

        # --- Step 5: Update s (gradient descent for gate) ---
        s_param = s.clone().requires_grad_(True)
        for _ in range(s_epochs):
            silu_sp = F.silu(s_param)
            m_gate = (W_gate @ X.T).T
            loss_s = (
                alpha * ((s_param - m_gate) ** 2).sum()
                + gamma * ((p - silu_sp * z) ** 2).sum()
            )
            loss_s.backward()
            with torch.no_grad():
                s_param -= s_lr * s_param.grad
                s_param.grad.zero_()
        s = s_param.detach()

    return mask_up.cpu(), mask_gate.cpu(), mask_down.cpu()


# ---------------------------------------------------------------------------
# Layer accessor helpers for Gemma models
# ---------------------------------------------------------------------------


def _get_transformer_layers(model: PreTrainedModel) -> nn.ModuleList:
    """Get the list of transformer layers from a Gemma/LLaMA model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        return model.language_model.model.layers
    raise ValueError(f"Cannot find transformer layers in {type(model)}")


def _get_ffn_weights(layer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract (up_proj, gate_proj, down_proj) weight tensors from an FFN layer."""
    mlp = layer.mlp
    return mlp.up_proj.weight.data, mlp.gate_proj.weight.data, mlp.down_proj.weight.data


def _get_attention_linears(layer) -> dict[str, nn.Linear]:
    """Get all linear layers in the attention block."""
    attn = layer.self_attn
    result = {}
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        if hasattr(attn, name):
            result[name] = getattr(attn, name)
    return result


# ---------------------------------------------------------------------------
# Core: compute SparseLLM masks
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_sparse_llm_masks(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    sparsity: float,
    n_iterations: int = 4,
    alpha: float = 5.0,
    beta: float = 5.0,
    gamma: float = 5.0,
    max_seq_len: int = 2048,
    save_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Compute SparseLLM pruning masks for all layers.

    Attention layers are pruned locally (magnitude-based).
    FFN layers are pruned via iterative alternating optimization.

    Args:
        model: HuggingFace Gemma/LLaMA model on GPU.
        tokenizer: Matching tokenizer.
        calibration_texts: Pre-formatted calibration strings.
        sparsity: Target sparsity fraction (0.0-1.0).
        n_iterations: Alternating optimization iterations per FFN block.
        alpha, beta, gamma: SparseLLM penalty coefficients.
        max_seq_len: Max sequence length for calibration.
        save_path: Optionally save masks as safetensors.

    Returns:
        Dict of param_name -> 0/1 float mask tensor (on CPU).
    """
    model.eval()
    try:
        model_device = model.device
    except AttributeError:
        model_device = next(p.device for p in model.parameters())

    layers = _get_transformer_layers(model)
    n_layers = len(layers)

    # 1. Collect initial layer inputs by running calibration through the model
    print(f"[sparse_llm] Collecting layer inputs from {len(calibration_texts)} calibration samples...")
    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    # Run calibration and collect hidden states at each layer
    # We process one sample at a time to save memory
    all_hidden_states: list[torch.Tensor] = []
    with torch.no_grad():
        for text in tqdm(calibration_texts, desc="  calibration"):
            tokens = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=max_seq_len,
            )
            input_ids = tokens["input_ids"].to(model_device)
            attention_mask = tokens["attention_mask"].to(model_device)
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # hidden_states[i] is the input to layer i (hidden_states[0] = embedding output)
            # Stack all token positions from this sample
            for layer_idx in range(n_layers):
                hs = outputs.hidden_states[layer_idx].squeeze(0).cpu()  # (seq_len, d_model)
                if layer_idx >= len(all_hidden_states):
                    all_hidden_states.append(hs)
                else:
                    all_hidden_states[layer_idx] = torch.cat(
                        [all_hidden_states[layer_idx], hs], dim=0
                    )
            # Also collect the output of the last layer for down_proj Y
            last_hs = outputs.hidden_states[n_layers].squeeze(0).cpu()
            if n_layers >= len(all_hidden_states):
                all_hidden_states.append(last_hs)
            else:
                all_hidden_states[n_layers] = torch.cat(
                    [all_hidden_states[n_layers], last_hs], dim=0
                )

    tokenizer.padding_side = old_pad_side

    # 2. Process each layer
    all_masks: dict[str, torch.Tensor] = {}

    # Determine the layer name prefix
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_prefix = "model.layers"
    elif hasattr(model, "language_model"):
        layer_prefix = "model.language_model.layers"
    else:
        layer_prefix = "model.layers"

    for layer_idx in tqdm(range(n_layers), desc="  processing layers"):
        layer = layers[layer_idx]
        X = all_hidden_states[layer_idx].to(model_device)  # (n_tokens, d_model)

        # --- Attention: local per-row magnitude pruning ---
        attn_linears = _get_attention_linears(layer)
        for proj_name, linear in attn_linears.items():
            param_name = f"{layer_prefix}.{layer_idx}.self_attn.{proj_name}.weight"
            mask = _prune_local_per_row(linear.weight.data.float(), sparsity)
            all_masks[param_name] = mask.cpu()

        # --- FFN: SparseLLM alternating optimization ---
        W_up, W_gate, W_down = _get_ffn_weights(layer)

        # Compute FFN output Y for this layer's down_proj target
        # Y = layer(X) residual contribution from FFN (approximately)
        # We use the next layer's input minus this layer's input as the residual
        if layer_idx + 1 < len(all_hidden_states):
            Y = all_hidden_states[layer_idx + 1].to(model_device)
        else:
            Y = X  # Fallback (shouldn't happen for normal models)

        # Subsample tokens if too many (memory)
        max_tokens = 4096
        if X.shape[0] > max_tokens:
            idx = torch.randperm(X.shape[0])[:max_tokens]
            X_sub = X[idx]
            Y_sub = Y[idx]
        else:
            X_sub = X
            Y_sub = Y

        with torch.enable_grad():
            mask_up, mask_gate, mask_down = _sparse_llm_ffn_swiglu(
                W_up, W_gate, W_down, X_sub, Y_sub,
                sparsity=sparsity,
                n_iterations=n_iterations,
                alpha=alpha, beta=beta, gamma=gamma,
            )

        all_masks[f"{layer_prefix}.{layer_idx}.mlp.up_proj.weight"] = mask_up
        all_masks[f"{layer_prefix}.{layer_idx}.mlp.gate_proj.weight"] = mask_gate
        all_masks[f"{layer_prefix}.{layer_idx}.mlp.down_proj.weight"] = mask_down

        # Free memory
        del X, Y, X_sub, Y_sub
        torch.cuda.empty_cache()

    # Free calibration data
    del all_hidden_states
    torch.cuda.empty_cache()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(all_masks, str(save_path))
        print(f"[sparse_llm] Saved masks to {save_path}")

    return all_masks


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def prune_sparse_llm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    sparsity: float,
    n_iterations: int = 4,
    alpha: float = 5.0,
    beta: float = 5.0,
    gamma: float = 5.0,
    max_seq_len: int = 2048,
    mask_save_path: str | Path | None = None,
    return_masks: bool = False,
) -> "int | tuple[int, dict[str, torch.Tensor]]":
    """End-to-end SparseLLM pruning.

    Args:
        model: Model to prune in-place.
        tokenizer: Matching tokenizer.
        calibration_texts: Calibration data.
        sparsity: Target sparsity fraction.
        n_iterations: Alternating optimization iterations.
        alpha, beta, gamma: Penalty coefficients.
        max_seq_len: Max sequence length.
        mask_save_path: Optionally save masks.
        return_masks: If True, return (n_zeroed, masks) for PGD.

    Returns:
        n_zeroed (int) or (n_zeroed, masks) if return_masks=True.
    """
    masks = compute_sparse_llm_masks(
        model, tokenizer, calibration_texts,
        sparsity=sparsity,
        n_iterations=n_iterations,
        alpha=alpha, beta=beta, gamma=gamma,
        max_seq_len=max_seq_len,
        save_path=mask_save_path,
    )

    # Apply masks
    n_zeroed = 0
    for name, param in model.named_parameters():
        if name not in masks:
            continue
        mask = masks[name].to(device=param.device, dtype=param.dtype)
        n_zeroed += int((mask == 0).sum().item())
        param.data.mul_(mask)
        del mask

    n_total = sum(p.numel() for name, p in model.named_parameters() if name in masks)
    print(
        f"[sparse_llm] Pruned {n_zeroed:,} / {n_total:,} weights "
        f"({n_zeroed / n_total:.2%} actual sparsity, target {sparsity:.2%})"
    )

    if return_masks:
        return n_zeroed, masks
    del masks
    torch.cuda.empty_cache()
    return n_zeroed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("sparse-llm")
@click.option("--model-id", default="google/gemma-2-2b-it")
@click.option("--dataset-name", default="4gate/StemQAMixture")
@click.option("--dataset-subset", default="biology")
@click.option("--n-calibration", default=64, help="Calibration samples (fewer than Wanda due to memory)")
@click.option("--max-seq-len", default=1024)
@click.option("--sparsity", default=0.5)
@click.option("--n-iterations", default=4)
@click.option("--alpha", default=5.0)
@click.option("--beta", default=5.0)
@click.option("--gamma", default=5.0)
@click.option("--output", default=None, help="Save masks to .safetensors")
@click.option("--device", default="cuda:0")
def main(
    model_id, dataset_name, dataset_subset, n_calibration, max_seq_len,
    sparsity, n_iterations, alpha, beta, gamma, output, device,
):
    """Compute SparseLLM masks and prune a model."""
    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager",
    )

    print(f"Loading calibration data from {dataset_name}/{dataset_subset}...")
    ds = load_dataset(dataset_name, dataset_subset, split="train")
    ds = ds.shuffle(seed=42)
    n = min(n_calibration, len(ds))
    calibration_texts = []
    for i in range(n):
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": str(ds[i]["question"])},
                {"role": "assistant", "content": str(ds[i]["answer"])},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        calibration_texts.append(text)

    n_zeroed = prune_sparse_llm(
        model, tokenizer, calibration_texts,
        sparsity=sparsity,
        n_iterations=n_iterations,
        alpha=alpha, beta=beta, gamma=gamma,
        max_seq_len=max_seq_len,
        mask_save_path=output,
    )
    print(f"Pruned {n_zeroed:,} weights at {sparsity:.0%} sparsity")


if __name__ == "__main__":
    main()
