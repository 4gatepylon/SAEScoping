"""
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

Memory-efficient pruning pipeline (used by prune_model)
--------------------------------------------------------
Phase 1 — compute_keep_masks:
    Load the saliency file and compute a boolean CPU keep-mask for every
    parameter.  No GPU memory is allocated.  For "gradient" saliency only
    the saliency file is needed; for "taylor" (|grad * weight|) the caller
    must supply ``param_weights_cpu``.

Phase 2 — (caller's responsibility):
    Load the model onto the GPU.  At this point only the model weights are
    on the GPU; all saliency tensors and masks are on CPU.

Phase 3 — apply_keep_masks_streaming:
    Iterate over parameters one at a time.  For each parameter, move its
    CPU mask to GPU, multiply the parameter in-place, then immediately
    delete the GPU copy of the mask.  Peak extra GPU memory is bounded by
    a single mask tensor (at most a few hundred MB).
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
# Debug helpers
# ---------------------------------------------------------------------------


def _cuda_mem_summary() -> str:
    """Return a one-line CUDA memory summary for debug logging.

    Reports allocated and reserved memory on the current default device.
    Returns a no-op string when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    dev = torch.cuda.current_device()
    alloc_gb = torch.cuda.memory_allocated(dev) / 1024**3
    reserved_gb = torch.cuda.memory_reserved(dev) / 1024**3
    total_gb = torch.cuda.get_device_properties(dev).total_memory / 1024**3
    return (
        f"GPU mem: {alloc_gb:.2f} GB alloc / {reserved_gb:.2f} GB reserved"
        f" / {total_gb:.2f} GB total"
    )


# ---------------------------------------------------------------------------
# Saliency map loading
# ---------------------------------------------------------------------------


def load_saliency_map(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a saliency map from a safetensors file."""
    return load_file(str(path))


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
# Phase 1: compute CPU keep-masks (no GPU allocation)
# ---------------------------------------------------------------------------

# Number of elements sampled from all scores to estimate the global quantile.
# 10 M samples over 9 B elements gives < 0.01 % quantile error — more than
# sufficient for weight pruning.
_THRESHOLD_SAMPLE_BUDGET = 10_000_000


def compute_keep_masks(
    saliency_path: str | Path,
    sparsity_fraction: float,
    saliency_type: str = "gradient",
    param_regex: Optional[str] = None,
    param_weights_cpu: Optional[dict[str, torch.Tensor]] = None,
) -> dict[str, torch.Tensor]:
    """Phase 1: build CPU boolean keep-masks without touching the GPU.

    For each parameter present in the saliency file a boolean tensor is
    produced on CPU: True means *keep* the weight, False means zero it out.

    The global sparsity threshold is estimated via a proportional random
    sample across all parameter tensors (see ``_THRESHOLD_SAMPLE_BUDGET``),
    avoiding the need to concatenate all scores into one giant CPU tensor.

    Args:
        saliency_path: Path to the .safetensors saliency file.
        sparsity_fraction: Fraction of scored weights to zero (0.0–1.0).
        saliency_type: ``"gradient"`` (|grad|) or ``"taylor"`` (|grad*w|).
            Taylor mode requires ``param_weights_cpu``.
        param_regex: Optional regex; only matching parameter names are
            included.  Non-matching parameters are left out of the mask
            dict (and therefore untouched by apply_keep_masks_streaming).
        param_weights_cpu: Dict of parameter name -> CPU weight tensor.
            Required when ``saliency_type="taylor"``, ignored otherwise.

    Returns:
        Dict mapping parameter name -> CPU bool keep-mask (True = keep).
    """
    if saliency_type not in ("gradient", "taylor"):
        raise ValueError(
            f"Unknown saliency_type '{saliency_type}'. Choose 'gradient' or 'taylor'."
        )
    if saliency_type == "taylor" and param_weights_cpu is None:
        raise ValueError(
            "param_weights_cpu is required for taylor saliency. "
            "Pass a dict of {name: cpu_tensor} for each model parameter."
        )

    compiled_re = re.compile(param_regex) if param_regex is not None else None

    # -- Load saliency map (always CPU from safetensors) ---------------------
    print(
        f"[compute_keep_masks] Phase 1a — loading saliency map "
        f"from {saliency_path}  |  {_cuda_mem_summary()}"
    )
    saliency_tensors = load_saliency_map(saliency_path)
    print(
        f"[compute_keep_masks] Loaded {len(saliency_tensors)} saliency tensors "
        f"(all on CPU)  |  {_cuda_mem_summary()}"
    )

    # -- Compute per-parameter scores on CPU --------------------------------
    print(
        f"[compute_keep_masks] Phase 1b — computing {saliency_type} scores "
        f"on CPU (param_regex={param_regex!r})  |  {_cuda_mem_summary()}"
    )
    scores: dict[str, torch.Tensor] = {}
    for name, grad_tensor in tqdm(
        saliency_tensors.items(), desc="  computing scores", leave=False
    ):
        if compiled_re is not None and not compiled_re.search(name):
            continue
        grad = grad_tensor.float()  # already on CPU
        if saliency_type == "gradient":
            scores[name] = grad.abs()
        else:
            weight = param_weights_cpu[name].float()  # type: ignore[index]
            scores[name] = (grad * weight).abs()
    del saliency_tensors

    if not scores:
        print(
            "[compute_keep_masks] ⚠️ No parameters matched — "
            f"returning empty mask dict  |  {_cuda_mem_summary()}"
        )
        return {}

    n_params = len(scores)
    n_total = sum(s.numel() for s in scores.values())
    print(
        f"[compute_keep_masks] Scores computed for {n_params} params "
        f"({n_total:,} elements, all CPU)  |  {_cuda_mem_summary()}"
    )

    # -- Estimate global quantile threshold via proportional random sample --
    print(
        f"[compute_keep_masks] Phase 1c — sampling "
        f"{min(_THRESHOLD_SAMPLE_BUDGET, n_total):,} / {n_total:,} elements "
        f"to find {sparsity_fraction:.2%} quantile threshold  |  {_cuda_mem_summary()}"
    )
    sample_parts: list[torch.Tensor] = []
    for s in tqdm(scores.values(), desc="  sampling for threshold", leave=False):
        flat = s.flatten()
        k = max(1, round(flat.numel() / n_total * _THRESHOLD_SAMPLE_BUDGET))
        idx = torch.randperm(flat.numel())[:k]
        sample_parts.append(flat[idx])
    sample = torch.cat(sample_parts)
    threshold = torch.quantile(sample, sparsity_fraction).item()
    del sample, sample_parts
    print(
        f"[compute_keep_masks] Score threshold = {threshold:.6g}  "
        f"|  {_cuda_mem_summary()}"
    )

    # -- Build boolean keep-masks on CPU ------------------------------------
    print(
        f"[compute_keep_masks] Phase 1d — building bool keep-masks on CPU  "
        f"|  {_cuda_mem_summary()}"
    )
    keep_masks: dict[str, torch.Tensor] = {}
    for name, score in tqdm(scores.items(), desc="  building masks", leave=False):
        keep_masks[name] = score > threshold
    del scores

    n_zeroed = sum(int((~m).sum().item()) for m in keep_masks.values())
    print(
        f"[compute_keep_masks] Done — {n_zeroed:,} / {n_total:,} weights will be zeroed "
        f"({n_zeroed / n_total:.2%} actual sparsity, target {sparsity_fraction:.2%})  "
        f"|  {_cuda_mem_summary()}"
    )
    return keep_masks


# ---------------------------------------------------------------------------
# Phase 3: stream CPU masks onto GPU one parameter at a time
# ---------------------------------------------------------------------------


def apply_keep_masks_streaming(
    model: PreTrainedModel,
    keep_masks: dict[str, torch.Tensor],
) -> int:
    """Phase 3: apply CPU keep-masks to a GPU model one parameter at a time.

    For each parameter in ``keep_masks``:
      1. Move the CPU mask to the parameter's device.
      2. Multiply the parameter data in-place.
      3. Immediately delete the GPU mask copy.

    Peak extra GPU memory is bounded by a single mask tensor (a few hundred
    MB at most for the largest weight matrices in a 9B model).

    Args:
        model: Model to prune in-place.  Parameters are expected to be on GPU.
        keep_masks: CPU bool tensors returned by ``compute_keep_masks``.

    Returns:
        Number of weights zeroed.
    """
    params_to_prune = [
        (name, param)
        for name, param in model.named_parameters()
        if name in keep_masks
    ]
    print(
        f"[apply_keep_masks_streaming] Phase 3 — applying {len(params_to_prune)} masks "
        f"to GPU model, one parameter at a time  |  {_cuda_mem_summary()}"
    )
    n_zeroed = 0
    for name, param in tqdm(params_to_prune, desc="  zeroing weights", leave=False):
        mask_cpu = keep_masks[name]
        n_zeroed += int((~mask_cpu).sum().item())
        mask_gpu = mask_cpu.to(device=param.device, dtype=param.dtype)
        param.data.mul_(mask_gpu)
        del mask_gpu  # free GPU copy immediately
    print(
        f"[apply_keep_masks_streaming] Done — {n_zeroed:,} weights zeroed  "
        f"|  {_cuda_mem_summary()}"
    )
    return n_zeroed


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def prune_model(
    model: PreTrainedModel,
    saliency_path: str | Path,
    sparsity_fraction: float,
    saliency_type: str = "gradient",
    param_regex: Optional[str] = None,
) -> int:
    """
    Prune a model in-place using the memory-efficient 3-phase pipeline.

    Phase 1 — compute_keep_masks:
        Load the saliency file and build CPU boolean keep-masks.  The model
        must already be on GPU; no GPU memory is allocated during this phase.
    Phase 2 — (implicit):
        The model is already on GPU; nothing extra is loaded.
    Phase 3 — apply_keep_masks_streaming:
        Apply masks one parameter at a time, moving each mask to GPU only for
        the duration of a single multiply, then freeing it immediately.

    After phase 3 the keep-mask dict is deleted and the CUDA allocator cache
    is cleared so downstream training starts with a clean memory slate.

    Args:
        model: Model to prune (must be on GPU).
        saliency_path: Path to .safetensors saliency map.
        sparsity_fraction: Fraction of scored weights to zero (0.0–1.0).
        saliency_type: ``"gradient"`` or ``"taylor"``.
        param_regex: Optional regex to restrict which parameters are pruned.

    Returns:
        Number of weights zeroed.
    """
    print(
        f"[prune_model] Starting memory-efficient pruning "
        f"(sparsity={sparsity_fraction:.2%}, type={saliency_type})  "
        f"|  {_cuda_mem_summary()}"
    )

    # Phase 1 — all CPU, no GPU allocation
    param_weights_cpu: Optional[dict[str, torch.Tensor]] = None
    if saliency_type == "taylor":
        print(
            "[prune_model] Taylor saliency: copying model weights to CPU for scoring  "
            f"|  {_cuda_mem_summary()}"
        )
        param_weights_cpu = {
            name: param.data.cpu() for name, param in model.named_parameters()
        }
    keep_masks = compute_keep_masks(
        saliency_path,
        sparsity_fraction,
        saliency_type=saliency_type,
        param_regex=param_regex,
        param_weights_cpu=param_weights_cpu,
    )
    del param_weights_cpu

    # Phase 2 — model is already on GPU; nothing to load
    print(
        f"[prune_model] Phase 2 — model already on GPU, masks are on CPU  "
        f"|  {_cuda_mem_summary()}"
    )

    # Phase 3 — stream masks onto GPU one at a time
    n_zeroed = apply_keep_masks_streaming(model, keep_masks)

    # Release masks and return GPU memory to CUDA allocator so that downstream
    # training starts with the maximum possible free memory.
    del keep_masks
    torch.cuda.empty_cache()
    print(
        f"[prune_model] Pruning complete — {n_zeroed:,} weights zeroed, "
        f"GPU cache cleared  |  {_cuda_mem_summary()}"
    )
    return n_zeroed
