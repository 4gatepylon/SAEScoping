"""Unified saliency computation and masking across all pruning methods.

This is the interface layer: callers pick a method name and get back
saliency scores or boolean masks without importing each method directly.

Supported methods: wanda, random, taylor, gradient.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from sae_scoping.utils.cache import cache_path, load_or_compute_safetensors
from sae_scoping.training.saliency.wanda import (
    compute_wanda_saliency,
    compute_wanda_masks,
)

METHODS = ("wanda", "random", "taylor", "gradient")


def compute_saliency(
    method: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    max_seq_len: int,
    cache_dir: Path,
    model_id: str,
    dataset_subset: str,
    no_cache: bool = False,
    dataset_name: str = "4gate/StemQAMixture",
) -> dict[str, torch.Tensor]:
    """Compute and cache saliency scores for a given method.

    Args:
        method: One of "wanda", "random", "taylor", "gradient".
        model: Model to score (should be on device).
        tokenizer: Matching tokenizer.
        calibration_texts: Pre-formatted calibration strings.
        max_seq_len: Truncation length for calibration.
        cache_dir: Root directory for cached artifacts.
        model_id: HuggingFace model ID (used in cache path).
        dataset_subset: Dataset subset name (used in cache path).
        no_cache: If True, skip reading/writing cache.
        dataset_name: HuggingFace dataset ID (for gradient computation).

    Returns:
        Dict mapping parameter name -> saliency score tensor.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method!r}. Choose from {METHODS}")

    if method == "wanda":
        path = cache_path(cache_dir, model_id, dataset_subset, "wanda_saliency.safetensors")
        return load_or_compute_safetensors(
            path,
            lambda: compute_wanda_saliency(model, tokenizer, calibration_texts, max_seq_len=max_seq_len),
            no_cache=no_cache, label="Wanda saliency",
        )

    if method == "random":
        from sae_scoping.training.saliency.random import make_random_map
        path = cache_path(cache_dir, model_id, dataset_subset, "random_saliency.safetensors")
        return load_or_compute_safetensors(
            path,
            lambda: make_random_map(model, seed=42),
            no_cache=no_cache, label="random saliency",
        )

    # taylor and gradient both need the EMA gradient map first
    grad_path = cache_path(cache_dir, model_id, dataset_subset, "ema_grads.safetensors")
    raw_grads = load_or_compute_safetensors(
        grad_path,
        lambda: _compute_ema_grads(model, tokenizer, dataset_name, dataset_subset),
        no_cache=no_cache, label="EMA gradient map",
    )

    if method == "taylor":
        from sae_scoping.training.saliency.taylor import make_taylor_map
        path = cache_path(cache_dir, model_id, dataset_subset, "taylor_saliency.safetensors")
        return load_or_compute_safetensors(
            path,
            lambda: make_taylor_map(raw_grads, model),
            no_cache=no_cache, label="Taylor saliency",
        )

    # method == "gradient"
    path = cache_path(cache_dir, model_id, dataset_subset, "gradient_saliency.safetensors")
    return load_or_compute_safetensors(
        path,
        lambda: {k: v.abs() for k, v in raw_grads.items()},
        no_cache=no_cache, label="gradient saliency",
    )


def masks_for_sparsity(
    method: str,
    saliency_data: dict[str, torch.Tensor],
    sparsity: float,
) -> dict[str, torch.Tensor]:
    """Compute boolean keep masks from pre-computed saliency data.

    Args:
        method: Saliency method name. Wanda uses per-row thresholding;
            all others use global thresholding.
        saliency_data: Dict of parameter name -> score tensor, as returned
            by ``compute_saliency``.
        sparsity: Fraction of weights to prune (0.0-1.0).

    Returns:
        Dict of parameter name -> bool tensor (True = keep).
    """
    if method == "wanda":
        return compute_wanda_masks(saliency_data, sparsity)

    # Global threshold for random/taylor/gradient
    all_scores = torch.cat([s.flatten().float() for s in saliency_data.values()])
    threshold = torch.quantile(all_scores, sparsity).item()
    return {name: (scores > threshold) for name, scores in saliency_data.items()}


def _compute_ema_grads(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str,
    dataset_subset: str,
) -> dict[str, torch.Tensor]:
    """Compute EMA gradient map by training a GradCollectTrainer for 1 epoch."""
    from sae_scoping.datasets.qa_datasets import load_qa_dataset, format_as_sft_dataset
    from sae_scoping.training.saliency.grad import GradCollectTrainer
    from trl import SFTConfig

    qa_dataset = load_qa_dataset(dataset_name, dataset_subset, split="train", n=4096, seed=42)
    sft_dataset = format_as_sft_dataset(qa_dataset, tokenizer)
    trainer = GradCollectTrainer(
        model=model, beta=0.95, abs_grad=False,
        processing_class=tokenizer, train_dataset=sft_dataset,
        args=SFTConfig(
            output_dir="./deleteme_grad_collect", num_train_epochs=1,
            per_device_train_batch_size=2, gradient_accumulation_steps=1,
            bf16=True, max_grad_norm=None, learning_rate=1e-4,
            save_strategy="no", report_to="none", max_length=1024,
            dataset_text_field="text",
        ),
    )
    trainer.train()
    return trainer.ema_grads()
