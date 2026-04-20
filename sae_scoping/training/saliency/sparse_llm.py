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

Shared computation (calibration hidden states, pseudo-inverses, initial
auxiliary variables) is separated from per-sparsity optimization via
precompute_shared_data(), so a sweep over sparsity levels only runs the
cheap iterative step per level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
# Shared precomputed data
# ---------------------------------------------------------------------------


@dataclass
class _LayerSharedData:
    """Precomputed data for one transformer layer, shared across sparsity levels."""
    X: torch.Tensor            # (n_tokens, d_model) — layer input (on CPU)
    Y: torch.Tensor            # (n_tokens, d_model) — layer residual target (on CPU)
    Xinv: torch.Tensor         # (d_model, n_tokens) — pseudo-inverse of X (on CPU)
    W_up_orig: torch.Tensor    # (d_ffn, d_model) — original up_proj weight (on CPU)
    W_gate_orig: torch.Tensor  # (d_ffn, d_model) — original gate_proj weight (on CPU)
    W_down_orig: torch.Tensor  # (d_model, d_ffn) — original down_proj weight (on CPU)
    z_init: torch.Tensor       # (n_tokens, d_ffn) — initial up_proj output (on CPU)
    s_init: torch.Tensor       # (n_tokens, d_ffn) — initial gate_proj output (on CPU)
    p_init: torch.Tensor       # (n_tokens, d_ffn) — initial down_proj input (on CPU)
    # Attention weights for local pruning
    attn_weights: dict[str, torch.Tensor] = field(default_factory=dict)  # proj_name -> weight (CPU)


@dataclass
class SparseLLMSharedData:
    """All precomputed data shared across sparsity levels."""
    layer_data: list[_LayerSharedData]
    layer_prefix: str
    n_layers: int


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


def _get_attention_weights(layer) -> dict[str, torch.Tensor]:
    """Get all attention projection weights."""
    attn = layer.self_attn
    result = {}
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        if hasattr(attn, name):
            result[name] = getattr(attn, name).weight.data.clone().cpu()
    return result


def _detect_layer_prefix(model: PreTrainedModel) -> str:
    """Determine the parameter name prefix for transformer layers."""
    sample_param = next(
        (n for n, _ in model.named_parameters() if ".self_attn.q_proj.weight" in n), None
    )
    if sample_param is not None:
        return sample_param.split(".self_attn.")[0].rsplit(".", 1)[0]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return "model.layers"
    return "model.layers"


# ---------------------------------------------------------------------------
# Step 1: Precompute shared data (expensive, do once)
# ---------------------------------------------------------------------------


@torch.no_grad()
def precompute_shared_data(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    max_seq_len: int = 2048,
    max_tokens: int = 4096,
) -> SparseLLMSharedData:
    """Precompute all data shared across sparsity levels.

    This runs the calibration forward pass, collects hidden states,
    computes pseudo-inverses, and initializes auxiliary variables.
    All results are stored on CPU.

    Args:
        model: HuggingFace model (on GPU).
        tokenizer: Matching tokenizer.
        calibration_texts: Pre-formatted calibration strings.
        max_seq_len: Max sequence length for calibration.
        max_tokens: Subsample to this many tokens per layer (memory).

    Returns:
        SparseLLMSharedData containing all shared precomputed data.
    """
    model.eval()
    try:
        model_device = model.device
    except AttributeError:
        model_device = next(p.device for p in model.parameters())

    layers = _get_transformer_layers(model)
    n_layers = len(layers)
    layer_prefix = _detect_layer_prefix(model)

    # 1. Collect hidden states from calibration data
    print(f"[sparse_llm] Precomputing shared data from {len(calibration_texts)} calibration samples...")
    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    all_hidden_states: list[torch.Tensor] = []
    for text in tqdm(calibration_texts, desc="  calibration forward pass"):
        tokens = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_seq_len,
        )
        input_ids = tokens["input_ids"].to(model_device)
        attention_mask = tokens["attention_mask"].to(model_device)
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True,
        )
        for layer_idx in range(n_layers + 1):
            hs = outputs.hidden_states[layer_idx].squeeze(0).cpu()
            if layer_idx >= len(all_hidden_states):
                all_hidden_states.append(hs)
            else:
                all_hidden_states[layer_idx] = torch.cat(
                    [all_hidden_states[layer_idx], hs], dim=0
                )

    tokenizer.padding_side = old_pad_side

    # 2. Per-layer: compute pseudo-inverses, initial auxiliary vars, collect attn weights
    layer_data_list: list[_LayerSharedData] = []

    for layer_idx in tqdm(range(n_layers), desc="  computing pseudo-inverses"):
        layer = layers[layer_idx]
        X = all_hidden_states[layer_idx]
        next_hidden = all_hidden_states[layer_idx + 1] if layer_idx + 1 < len(all_hidden_states) else X
        Y = next_hidden - X  # Residual contribution

        # Subsample tokens if too many
        if X.shape[0] > max_tokens:
            idx = torch.randperm(X.shape[0])[:max_tokens]
            X = X[idx]
            Y = Y[idx]

        # Move to GPU for pseudo-inverse computation
        X_gpu = X.float().to(model_device)
        Xinv = torch.linalg.pinv(X_gpu).cpu()

        # Get original FFN weights
        W_up, W_gate, W_down = _get_ffn_weights(layer)
        W_up_cpu = W_up.float().cpu()
        W_gate_cpu = W_gate.float().cpu()
        W_down_cpu = W_down.float().cpu()

        # Compute initial auxiliary variables
        X_f = X.float()
        z_init = (W_up_cpu @ X_f.T).T
        s_init = (W_gate_cpu @ X_f.T).T
        p_init = F.silu(s_init) * z_init

        # Attention weights
        attn_weights = _get_attention_weights(layer)

        layer_data_list.append(_LayerSharedData(
            X=X, Y=Y, Xinv=Xinv,
            W_up_orig=W_up_cpu, W_gate_orig=W_gate_cpu, W_down_orig=W_down_cpu,
            z_init=z_init, s_init=s_init, p_init=p_init,
            attn_weights=attn_weights,
        ))

        # Free GPU memory
        del X_gpu
        torch.cuda.empty_cache()

    del all_hidden_states
    torch.cuda.empty_cache()

    print(f"[sparse_llm] Shared data precomputed for {n_layers} layers")
    return SparseLLMSharedData(
        layer_data=layer_data_list,
        layer_prefix=layer_prefix,
        n_layers=n_layers,
    )


# ---------------------------------------------------------------------------
# Step 2: Per-sparsity optimization (cheap, run per sparsity level)
# ---------------------------------------------------------------------------


def _sparse_llm_ffn_swiglu(
    W_up: torch.Tensor,       # (d_ffn, d_model)
    W_gate: torch.Tensor,     # (d_ffn, d_model)
    W_down: torch.Tensor,     # (d_model, d_ffn)
    X: torch.Tensor,          # (n_tokens, d_model)
    Y: torch.Tensor,          # (n_tokens, d_model)
    Xinv: torch.Tensor,       # (d_model, n_tokens)
    z: torch.Tensor,          # (n_tokens, d_ffn) — initial
    s: torch.Tensor,          # (n_tokens, d_ffn) — initial
    p: torch.Tensor,          # (n_tokens, d_ffn) — initial
    sparsity: float,
    n_iterations: int = 4,
    alpha: float = 5.0,
    beta: float = 5.0,
    gamma: float = 5.0,
    s_lr: float = 0.01,
    s_epochs: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run SparseLLM alternating optimization on a SwiGLU FFN block.

    Takes precomputed Xinv and initial auxiliary variables.
    Returns (mask_up, mask_gate, mask_down) — 0/1 float tensors on CPU.
    """
    device = W_up.device

    for it in range(n_iterations):
        # --- Step 1: Update W (closed-form least squares) ---
        W_up = z.T @ Xinv.T
        W_gate = s.T @ Xinv.T
        pinv_p = torch.linalg.pinv(p)
        W_down = Y.T @ pinv_p.T

        # --- Step 2: Prune W (per-row magnitude) ---
        mask_up = _prune_local_per_row(W_up, sparsity)
        mask_gate = _prune_local_per_row(W_gate, sparsity)
        mask_down = _prune_local_per_row(W_down, sparsity)
        W_up = W_up * mask_up
        W_gate = W_gate * mask_gate
        W_down = W_down * mask_down

        # --- Step 3: Update p (ridge regression) ---
        WdtWd = W_down.T @ W_down
        regularizer = gamma * torch.eye(WdtWd.shape[0], device=device)
        A = beta * WdtWd + regularizer
        silu_s = F.silu(s)
        b = (beta * (W_down.T @ Y.T) + gamma * (silu_s * z).T)
        p = torch.linalg.solve(A, b).T

        # --- Step 4: Update z (closed-form using SiLU) ---
        m_up = (W_up @ X.T).T
        silu_s = F.silu(s)
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


def compute_sparse_llm_masks(
    shared: SparseLLMSharedData,
    model: PreTrainedModel,
    sparsity: float,
    n_iterations: int = 4,
    alpha: float = 5.0,
    beta: float = 5.0,
    gamma: float = 5.0,
    save_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Compute SparseLLM masks for a specific sparsity level using precomputed shared data.

    Args:
        shared: Precomputed data from precompute_shared_data().
        model: Model (only used for device placement).
        sparsity: Target sparsity fraction (0.0-1.0).
        n_iterations: Alternating optimization iterations per FFN block.
        alpha, beta, gamma: SparseLLM penalty coefficients.
        save_path: Optionally save masks as safetensors.

    Returns:
        Dict of param_name -> 0/1 float mask tensor (on CPU).
    """
    try:
        model_device = model.device
    except AttributeError:
        model_device = next(p.device for p in model.parameters())

    all_masks: dict[str, torch.Tensor] = {}

    for layer_idx in tqdm(range(shared.n_layers), desc=f"  masks @ {sparsity:.0%}"):
        ld = shared.layer_data[layer_idx]

        # --- Attention: local per-row magnitude pruning ---
        for proj_name, weight in ld.attn_weights.items():
            param_name = f"{shared.layer_prefix}.{layer_idx}.self_attn.{proj_name}.weight"
            mask = _prune_local_per_row(weight.float(), sparsity)
            all_masks[param_name] = mask.cpu()

        # --- FFN: SparseLLM alternating optimization ---
        # Move shared data to GPU for this layer
        X_gpu = ld.X.float().to(model_device)
        Y_gpu = ld.Y.float().to(model_device)
        Xinv_gpu = ld.Xinv.float().to(model_device)
        W_up = ld.W_up_orig.clone().to(model_device)
        W_gate = ld.W_gate_orig.clone().to(model_device)
        W_down = ld.W_down_orig.clone().to(model_device)
        z = ld.z_init.clone().to(model_device)
        s = ld.s_init.clone().to(model_device)
        p = ld.p_init.clone().to(model_device)

        with torch.enable_grad():
            mask_up, mask_gate, mask_down = _sparse_llm_ffn_swiglu(
                W_up, W_gate, W_down, X_gpu, Y_gpu, Xinv_gpu, z, s, p,
                sparsity=sparsity,
                n_iterations=n_iterations,
                alpha=alpha, beta=beta, gamma=gamma,
            )

        all_masks[f"{shared.layer_prefix}.{layer_idx}.mlp.up_proj.weight"] = mask_up
        all_masks[f"{shared.layer_prefix}.{layer_idx}.mlp.gate_proj.weight"] = mask_gate
        all_masks[f"{shared.layer_prefix}.{layer_idx}.mlp.down_proj.weight"] = mask_down

        del X_gpu, Y_gpu, Xinv_gpu
        torch.cuda.empty_cache()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(all_masks, str(save_path))
        print(f"[sparse_llm] Saved masks to {save_path}")

    return all_masks


# ---------------------------------------------------------------------------
# Legacy API (compute_sparse_llm_masks without pre-shared data)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_sparse_llm_masks_from_scratch(
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
    """Compute SparseLLM masks without pre-shared data (convenience wrapper).

    Calls precompute_shared_data() then compute_sparse_llm_masks().
    Use this for one-off pruning; for sweeps, call them separately.
    """
    shared = precompute_shared_data(model, tokenizer, calibration_texts, max_seq_len=max_seq_len)
    return compute_sparse_llm_masks(
        shared, model, sparsity=sparsity,
        n_iterations=n_iterations, alpha=alpha, beta=beta, gamma=gamma,
        save_path=save_path,
    )


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
    shared_data: SparseLLMSharedData | None = None,
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
        shared_data: Precomputed shared data (avoids recomputation in sweeps).

    Returns:
        n_zeroed (int) or (n_zeroed, masks) if return_masks=True.
    """
    if shared_data is None:
        shared_data = precompute_shared_data(model, tokenizer, calibration_texts, max_seq_len=max_seq_len)

    masks = compute_sparse_llm_masks(
        shared_data, model, sparsity=sparsity,
        n_iterations=n_iterations, alpha=alpha, beta=beta, gamma=gamma,
        save_path=mask_save_path,
    )

    # Apply masks
    from sae_scoping.training.saliency.wanda import apply_masks_to_model
    n_zeroed = apply_masks_to_model(model, masks)

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
        model_id, dtype=torch.bfloat16, device_map=device,
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
