"""
prune.py

Apply a saliency map to a model by zeroing out the least-salient weights.

The caller is responsible for saving/reloading the model if they want to
undo pruning later — this module modifies weights **in-place**.

Saliency criteria
-----------------
gradient : score = |grad|
taylor   : score = |grad * weight|   (Taylor first-order approximation)

Parameter filtering
-------------------
By default all parameters are pruned. Pass a regex via ``param_regex`` to
restrict pruning to matching parameter names (the regex matches names we
DO want to prune; everything else is skipped).

CLI usage:
    python prune.py \\
        --saliency-path biology/ema_grads.safetensors \\
        --model-id google/gemma-2-9b-it \\
        --sparsity 0.5 \\
        --saliency-type taylor \\
        --output-dir pruned_model

    python prune.py \\
        --saliency-path biology/ema_grads.safetensors \\
        --model-id google/gemma-2-9b-it \\
        --sparsity 0.3 \\
        --param-regex "layers\\.\\d+\\.(self_attn|mlp)"
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import click
import torch
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


# ---------------------------------------------------------------------------
# Saliency map loading
# ---------------------------------------------------------------------------


def load_saliency_map(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a saliency map from a safetensors file."""
    return load_file(str(path))


# ---------------------------------------------------------------------------
# Saliency scoring
# ---------------------------------------------------------------------------


def compute_saliency_scores(
    model: PreTrainedModel,
    saliency_tensors: dict[str, torch.Tensor],
    saliency_type: str,
    param_regex: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """
    Compute per-parameter saliency scores from a loaded saliency map.

    Args:
        model: The model whose parameters are being scored.
        saliency_tensors: Raw saliency tensors (e.g. EMA gradients) keyed
            by parameter name.
        saliency_type: ``"gradient"`` for |grad|, ``"taylor"`` for
            |grad * weight|.
        param_regex: If provided, only parameters whose names match this
            regex are included. All others are skipped.

    Returns:
        Dict mapping parameter name -> saliency score tensor (same shape
        as the parameter, on the same device).
    """
    if saliency_type not in ("gradient", "taylor"):
        raise ValueError(
            f"Unknown saliency_type '{saliency_type}'. Choose 'gradient' or 'taylor'."
        )
    compiled_re = re.compile(param_regex) if param_regex is not None else None
    scores: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name not in saliency_tensors:
            continue
        if compiled_re is not None and not compiled_re.search(name):
            continue
        grad = saliency_tensors[name].float().to(param.device)
        if saliency_type == "gradient":
            scores[name] = grad.abs()
        else:
            scores[name] = (grad * param.data.float()).abs()
    return scores


# ---------------------------------------------------------------------------
# Weight save / restore
# ---------------------------------------------------------------------------


def save_original_weights(model: PreTrainedModel) -> dict[str, torch.Tensor]:
    """Clone all parameter data to CPU for later restoration."""
    return {name: param.data.cpu().clone() for name, param in model.named_parameters()}


def restore_original_weights(
    model: PreTrainedModel,
    original_weights: dict[str, torch.Tensor],
) -> None:
    """Restore model parameters in-place from a CPU copy."""
    for name, param in model.named_parameters():
        if name in original_weights:
            param.data.copy_(original_weights[name].to(param.device))


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def apply_pruning(
    model: PreTrainedModel,
    saliency_scores: dict[str, torch.Tensor],
    sparsity_fraction: float,
) -> int:
    """
    Zero out the lowest-saliency fraction of scored weights in-place.

    Only parameters present in ``saliency_scores`` are touched.

    Args:
        model: Model to prune in-place.
        saliency_scores: Output of :func:`compute_saliency_scores`.
        sparsity_fraction: Fraction of scored weights to zero (0.0-1.0).

    Returns:
        Number of weights actually zeroed.
    """
    if sparsity_fraction <= 0.0 or len(saliency_scores) == 0:
        return 0
    if sparsity_fraction >= 1.0:
        n_zeroed = 0
        for name, param in tqdm(
            list(model.named_parameters()), desc="  zeroing (full)", leave=False
        ):
            if name in saliency_scores:
                n_zeroed += param.data.numel()
                param.data.zero_()
        return n_zeroed

    # TODO(Adriano) in the future we MIGHT want to prune different PER layer
    print(f"[apply_pruning] Flattening {len(saliency_scores)} score tensors to CPU for threshold search...")
    all_scores = torch.cat([s.flatten().cpu() for s in saliency_scores.values()])
    n_total = all_scores.numel()
    n_prune = max(1, int(sparsity_fraction * n_total))
    print(f"[apply_pruning] {n_total:,} scored elements; finding kth-smallest at k={n_prune:,} (sparsity={sparsity_fraction:.2%})...")
    threshold = torch.kthvalue(all_scores, n_prune).values.item()
    del all_scores
    print(f"[apply_pruning] Score threshold={threshold:.6f}. Zeroing weights below threshold...")

    n_zeroed = 0
    params_to_prune = [
        (name, param)
        for name, param in model.named_parameters()
        if name in saliency_scores
    ]
    for name, param in tqdm(params_to_prune, desc="  zeroing weights", leave=False):
        mask = saliency_scores[name] > threshold
        n_zeroed += int((~mask).sum().item())
        param.data.mul_(mask.to(dtype=param.dtype, device=param.device))
    print(f"[apply_pruning] Weight zeroing complete: {n_zeroed:,} weights zeroed")
    return n_zeroed


def prune_model(
    model: PreTrainedModel,
    saliency_path: str | Path,
    sparsity_fraction: float,
    saliency_type: str = "gradient",
    param_regex: Optional[str] = None,
) -> int:
    """
    High-level convenience: load saliency map, compute scores, apply pruning.

    This is the main library entry point. Modifies the model in-place.

    Args:
        model: Model to prune.
        saliency_path: Path to .safetensors saliency map.
        sparsity_fraction: Fraction of scored weights to zero (0.0-1.0).
        saliency_type: ``"gradient"`` or ``"taylor"``.
        param_regex: Optional regex to filter which params get pruned.

    Returns:
        Number of weights zeroed.
    """
    print(f"[prune_model] Loading saliency map from {saliency_path}")
    saliency_tensors = load_saliency_map(saliency_path)
    print(f"[prune_model] Loaded {len(saliency_tensors)} saliency tensors")

    print(f"[prune_model] Computing {saliency_type} saliency scores (param_regex={param_regex!r})...")
    saliency_scores = compute_saliency_scores(
        model, saliency_tensors, saliency_type, param_regex=param_regex,
    )
    total_scored = sum(s.numel() for s in saliency_scores.values())
    print(f"[prune_model] Scores computed for {len(saliency_scores)} params ({total_scored:,} elements total)")

    print(f"[prune_model] Applying pruning at sparsity={sparsity_fraction:.2%}...")
    n_zeroed = apply_pruning(model, saliency_scores, sparsity_fraction)
    print(f"[prune_model] Pruning done: {n_zeroed:,} weights zeroed")
    return n_zeroed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--saliency-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to .safetensors saliency map (output of gradients_map.py).",
)
@click.option(
    "--model-id",
    type=str,
    default="google/gemma-2-9b-it",
    show_default=True,
)
@click.option(
    "--sparsity",
    type=float,
    required=True,
    help="Fraction of scored weights to zero (0.0-1.0).",
)
@click.option(
    "--saliency-type",
    type=click.Choice(["gradient", "taylor"]),
    default="gradient",
    show_default=True,
    help="gradient: |grad|.  taylor: |grad * weight|.",
)
@click.option(
    "--param-regex",
    type=str,
    default=None,
    help="Regex filter: only matching parameter names are pruned.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Save the pruned model to this directory. If omitted the model is not saved.",
)
@click.option(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
)
def main(
    saliency_path: Path,
    model_id: str,
    sparsity: float,
    saliency_type: str,
    param_regex: Optional[str],
    output_dir: Optional[Path],
    device: str,
) -> None:
    """Prune a model by zeroing out the least-salient weights."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    n_zeroed = prune_model(
        model, saliency_path, sparsity,
        saliency_type=saliency_type, param_regex=param_regex,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Pruned {n_zeroed:,} / {total_params:,} weights "
        f"({n_zeroed / total_params:.2%} actual sparsity, "
        f"target {sparsity:.2%})"
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(output_dir)
        print(f"Saved pruned model to {output_dir}")


if __name__ == "__main__":
    main()
