"""
Wanda: Pruning by Weights and Activations.

Implements the Wanda pruning criterion from:
  "A Simple and Effective Pruning Approach for Large Language Models"
  Sun et al., 2023 (arXiv:2306.11695)

The importance score for weight W[i,j] connecting input j to output i is:

    S[i,j] = |W[i,j]| * ||X_j||_2

where ||X_j||_2 is the L2 norm of the j-th input feature column across all
calibration tokens.

This module computes the saliency map and saves it as a safetensors file.
Wanda uses **per-row** pruning (each output neuron loses the same fraction
of inputs), handled by compute_wanda_masks() and prune_wanda() in this module.
"""

from __future__ import annotations

from pathlib import Path
import click
import torch
import torch.nn as nn
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


# ---------------------------------------------------------------------------
# Activation norm collection
# ---------------------------------------------------------------------------


class _ActivationNormCollector:
    """Forward hook that accumulates squared L2 norms of input activations.

    For a linear layer with weight shape (C_out, C_in), this collects
    a running mean of ||X_j||_2^2 for each input feature j, where X_j is
    the j-th column of the input matrix across all tokens in all batches.

    The result is stored in self.scaler_row of shape (C_in,).
    """

    def __init__(self, layer: nn.Linear):
        self.columns = layer.weight.shape[1]  # C_in
        self.device = layer.weight.device
        self.scaler_row = torch.zeros(self.columns, device=self.device, dtype=torch.float32)
        self.nsamples = 0

    def __call__(self, module: nn.Module, input, output):
        inp = input[0]
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)
        # inp shape: (batch, seq_len, C_in)
        inp_2d = inp.reshape(-1, inp.shape[-1])  # (batch*seq_len, C_in)
        n_tokens = inp_2d.shape[0]
        inp_2d = inp_2d.float()

        # Running mean update over tokens: scaler = scaler * (n_old/n_new) + new_norm^2 / n_new
        new_total = self.nsamples + n_tokens
        self.scaler_row.mul_(self.nsamples / new_total)
        self.scaler_row.add_(torch.norm(inp_2d, p=2, dim=0) ** 2 / new_total)
        self.nsamples = new_total


_SKIP_LAYER_NAMES = {"lm_head", "embed_tokens", "embed_out"}


def _find_linear_layers(module: nn.Module, prefix: str = "") -> dict[str, nn.Linear]:
    """Recursively find all nn.Linear layers, excluding embedding and LM head.

    Per Wanda (Sun et al. 2023): "linear layers, skipping the first
    embedding layer and the final classification head."
    """
    result = {}
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if name in _SKIP_LAYER_NAMES:
            continue
        if isinstance(child, nn.Linear):
            result[full_name] = child
        else:
            result.update(_find_linear_layers(child, full_name))
    return result


def assert_no_embedding_or_head_in_masks(
    masks: dict[str, torch.Tensor],
    model: PreTrainedModel,
) -> None:
    """Assert that masks don't include embedding or LM head parameters.

    Call after computing masks to catch cases where _find_linear_layers
    filtering was bypassed or a new model architecture has unexpected naming.
    """
    tied_params: set[int] = set()
    for name in ("lm_head", "embed_tokens", "embed_out"):
        for mname, mod in model.named_modules():
            if mname.endswith(name):  # BUG TODO(adriano): false positive on e.g. "custom_embed_tokens"; use == or endswith("."+name)
                for pname, p in mod.named_parameters():
                    tied_params.add(id(p))

    for mask_name in masks:
        for pname, p in model.named_parameters():
            if pname == mask_name and id(p) in tied_params:
                raise ValueError(
                    f"Mask includes '{mask_name}' which shares parameters with an embedding or LM head layer. This should not be pruned."
                )


# ---------------------------------------------------------------------------
# Core: compute Wanda saliency scores
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_wanda_saliency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    max_seq_len: int = 2048,
    batch_size: int = 1,
    save_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Compute Wanda saliency scores: |W[i,j]| * ||X_j||_2 for all linear layers.

    Args:
        model: HuggingFace model (should be on GPU).
        tokenizer: Matching tokenizer.
        calibration_texts: Pre-formatted text strings for calibration.
        max_seq_len: Maximum sequence length for tokenization.
        batch_size: Number of calibration texts per forward pass.
        save_path: If provided, save the saliency map to this path.

    Returns:
        Dict mapping parameter name -> saliency score tensor (same shape as weight).
        Scores are on CPU.
    """
    model.eval()
    assert isinstance(batch_size, int) and batch_size > 0, "Expected batch_size > 0."
    try:
        model_device = model.device
    except AttributeError:
        model_device = next(p.device for p in model.parameters())

    linear_layers = _find_linear_layers(model)
    collectors: dict[str, _ActivationNormCollector] = {}
    handles = []
    for name, layer in linear_layers.items():
        collector = _ActivationNormCollector(layer)
        collectors[name] = collector
        handles.append(layer.register_forward_hook(collector))

    # 2. Run calibration data through the model
    print(f"[wanda] Running {len(calibration_texts)} calibration samples...")
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for i in tqdm(range(0, len(calibration_texts), batch_size), desc="  calibration"):
            batch_texts = calibration_texts[i : i + batch_size]
            tokens = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
                padding=True,
            )
            input_ids = tokens["input_ids"].to(model_device)
            attention_mask = tokens["attention_mask"].to(model_device)
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        tokenizer.padding_side = old_padding_side
        for h in handles:
            h.remove()

    # 3. Compute saliency scores: |W[i,j]| * sqrt(mean(||X_j||_2^2))
    print("[wanda] Computing saliency scores...")
    saliency_map: dict[str, torch.Tensor] = {}
    for layer_name, collector in tqdm(collectors.items(), desc="  scoring"):
        layer = linear_layers[layer_name]
        weight = layer.weight.data.float()  # (C_out, C_in)
        # scaler_row is mean(||X_j||_2^2), shape (C_in,)
        activation_norm = torch.sqrt(collector.scaler_row).unsqueeze(0)  # (1, C_in)
        score = weight.abs() * activation_norm  # (C_out, C_in)
        # Store under the parameter name (layer_name + ".weight")
        param_name = layer_name + ".weight"
        saliency_map[param_name] = score.cpu()

    assert_no_embedding_or_head_in_masks(saliency_map, model)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(saliency_map, str(save_path))
        print(f"[wanda] Saved saliency map to {save_path}")

    return saliency_map


# ---------------------------------------------------------------------------
# Per-row pruning (Wanda-specific)
# ---------------------------------------------------------------------------


def compute_wanda_masks(
    saliency_map: dict[str, torch.Tensor],
    sparsity: float,
) -> dict[str, torch.Tensor]:
    """Compute per-row keep masks from Wanda saliency scores.

    For each weight matrix, sorts scores along dim=1 (input dimension) per
    row and keeps the top (1-sparsity) fraction per row.

    Args:
        saliency_map: Dict of param_name -> score tensor (C_out, C_in).
        sparsity: Fraction of weights to prune per row (0.0-1.0).

    Returns:
        Dict of param_name -> bool keep mask (True = keep, False = prune).
    """
    keep_masks: dict[str, torch.Tensor] = {}
    for name, scores in tqdm(saliency_map.items(), desc="  computing masks"):
        n_cols = scores.shape[1]
        n_prune = int(n_cols * sparsity)
        if n_prune == 0:
            keep_masks[name] = torch.ones_like(scores, dtype=torch.bool)
            continue
        # Per-row: find the n_prune lowest-scoring columns per row
        _, sorted_idx = torch.sort(scores, dim=1)
        mask = torch.ones_like(scores, dtype=torch.bool)
        prune_idx = sorted_idx[:, :n_prune]
        mask.scatter_(1, prune_idx, False)
        keep_masks[name] = mask
    return keep_masks


def apply_masks_to_model(
    model: PreTrainedModel,
    keep_masks: dict[str, torch.Tensor],
) -> int:
    """Apply keep masks to model weights in-place. Returns number of weights zeroed."""
    n_zeroed = 0
    for name, param in model.named_parameters():
        if name not in keep_masks:
            continue
        mask = keep_masks[name].to(device=param.device, dtype=param.dtype)
        n_zeroed += int((mask == 0).sum().item())
        param.data.mul_(mask)
        del mask
    return n_zeroed


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def prune_wanda(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    sparsity: float,
    max_seq_len: int = 2048,
    saliency_save_path: str | Path | None = None,
    return_masks: bool = False,
) -> "int | tuple[int, dict[str, torch.Tensor]]":
    """End-to-end Wanda pruning: compute scores, mask per-row, apply.

    Args:
        model: Model to prune in-place.
        tokenizer: Matching tokenizer.
        calibration_texts: Calibration data (pre-formatted strings).
        sparsity: Fraction of weights to prune per row.
        max_seq_len: Max sequence length for calibration.
        saliency_save_path: Optionally save the raw saliency map.
        return_masks: If True, return (n_zeroed, keep_masks) for PGD.

    Returns:
        n_zeroed (int) or (n_zeroed, keep_masks) if return_masks=True.
    """
    saliency_map = compute_wanda_saliency(
        model,
        tokenizer,
        calibration_texts,
        max_seq_len=max_seq_len,
        save_path=saliency_save_path,
    )
    keep_masks = compute_wanda_masks(saliency_map, sparsity)
    del saliency_map

    n_zeroed = apply_masks_to_model(model, keep_masks)

    n_total = sum(p.numel() for name, p in model.named_parameters() if name in keep_masks)
    print(f"[wanda] Pruned {n_zeroed:,} / {n_total:,} weights ({n_zeroed / n_total:.2%} actual sparsity, target {sparsity:.2%})")

    if return_masks:
        return n_zeroed, keep_masks
    del keep_masks
    torch.cuda.empty_cache()
    return n_zeroed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("wanda")
@click.option("--model-id", default="google/gemma-2-2b-it", help="HuggingFace model ID")
@click.option("--dataset-name", default="4gate/StemQAMixture", help="Calibration dataset")
@click.option("--dataset-subset", default="biology", help="Dataset subset/config")
@click.option("--n-calibration", default=128, help="Number of calibration samples")
@click.option("--max-seq-len", default=2048, help="Max sequence length")
@click.option("--sparsity", default=0.5, help="Target sparsity (fraction to prune)")
@click.option("--output", default=None, help="Path to save saliency map (.safetensors)")
@click.option("--device", default="cuda:0", help="Device")
def main(model_id, dataset_name, dataset_subset, n_calibration, max_seq_len, sparsity, output, device):
    """Compute Wanda saliency scores and optionally prune a model."""
    from sae_scoping.datasets.qa_datasets import load_qa_dataset, format_as_sft_text

    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,  # BUG TODO(adriano): was dtype= which may be silently ignored on older transformers
        attn_implementation="eager",
    )

    print(f"Loading calibration data from {dataset_name}/{dataset_subset}...")
    ds = load_qa_dataset(dataset_name, dataset_subset, n=n_calibration, seed=42)
    calibration_texts = format_as_sft_text(ds, tokenizer)

    if output:
        saliency_map = compute_wanda_saliency(
            model,
            tokenizer,
            calibration_texts,
            max_seq_len=max_seq_len,
            save_path=output,
        )
        print(f"Saved saliency map ({len(saliency_map)} params) to {output}")
    else:
        n_zeroed = prune_wanda(
            model,
            tokenizer,
            calibration_texts,
            sparsity=sparsity,
            max_seq_len=max_seq_len,
        )
        print(f"Pruned {n_zeroed:,} weights at {sparsity:.0%} sparsity")


if __name__ == "__main__":
    main()
