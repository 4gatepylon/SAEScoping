"""Run Wanda pruning sweep on a model and report loss vs sparsity."""

from pathlib import Path

import click
import torch

from sae_scoping.datasets.qa_datasets import load_qa_dataset, format_as_sft_text
from sae_scoping.evaluation.loss import compute_loss, count_zeros
from sae_scoping.training.saliency.validators import MaskSubsetValidator
from sae_scoping.training.saliency.wanda import (
    apply_masks_to_model,
    compute_wanda_masks,
    compute_wanda_saliency,
)
from sae_scoping.utils.cache import cache_path, load_or_compute_safetensors
from sae_scoping.utils.click_utils import load_yaml_config
from sae_scoping.utils.model_loading import load_model_and_tokenizer


@click.command()
@click.option(
    "--config",
    is_eager=True,
    expose_value=False,
    callback=load_yaml_config,
    type=click.Path(exists=True),
    help="YAML config file (CLI flags override).",
)
@click.option("--model-id", default="google/gemma-3-4b-it", show_default=True, help="HuggingFace model ID.")
@click.option("--dataset-name", default="4gate/StemQAMixture", show_default=True)
@click.option("--dataset-subset", default="biology", show_default=True)
@click.option("--n-calibration", default=128, show_default=True, help="Calibration samples.")
@click.option("--n-eval", default=64, show_default=True, help="Evaluation samples (separate from calibration).")
@click.option("--max-seq-len", default=2048, show_default=True)
@click.option("--batch-size", default=1, show_default=True, help="Batch size for calibration and eval.")
@click.option("--sparsity", "-s", multiple=True, type=float, help="Sparsity levels to sweep (repeat for multiple, e.g. -s 0.2 -s 0.4 -s 0.6).")
@click.option("--cache-dir", default="./cache", show_default=True, help="Directory for cached saliency maps.")
@click.option("--no-cache", is_flag=True, help="Recompute saliency even if cached.")
@click.option("--low-memory", is_flag=True, help="Skip mask monotonicity validation to save CPU memory.")
@click.option("--device", default="cuda:0", show_default=True)
def main(model_id, dataset_name, dataset_subset, n_calibration, n_eval, max_seq_len, batch_size, sparsity, cache_dir, no_cache, low_memory, device):
    """Run Wanda pruning sweep: compute saliency once, then evaluate at each sparsity level from low to high."""
    sparsities = sorted(sparsity) if sparsity else [0.5]
    print(f"Sweep sparsities: {[f'{s:.1%}' for s in sparsities]}")

    print(f"Loading tokenizer and model: {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, device=device)

    print(f"Loading dataset: {dataset_name}/{dataset_subset}")
    n_total = n_calibration + n_eval
    ds = load_qa_dataset(dataset_name, dataset_subset, n=n_total, seed=42)
    all_texts = format_as_sft_text(ds, tokenizer)
    calib_texts = all_texts[:n_calibration]
    eval_texts = all_texts[n_calibration:]

    print(f"\n=== Baseline (pre-pruning) ===")
    baseline_loss = compute_loss(model, tokenizer, eval_texts, max_seq_len=max_seq_len, batch_size=batch_size)
    zeros_before, total_params = count_zeros(model)
    print(f"  Loss:     {baseline_loss:.4f}")
    print(f"  Sparsity: {zeros_before}/{total_params} ({zeros_before / total_params:.2%})")

    saliency_file = cache_path(Path(cache_dir), model_id, dataset_subset, "wanda_saliency.safetensors")
    saliency_map = load_or_compute_safetensors(
        path=saliency_file,
        compute_fn=lambda: compute_wanda_saliency(model, tokenizer, calib_texts, max_seq_len=max_seq_len, batch_size=batch_size),
        no_cache=no_cache,
        label="Wanda saliency",
    )

    validator = MaskSubsetValidator(enabled=not low_memory)
    results = []
    for sparsity in sparsities:
        masks = compute_wanda_masks(saliency_map, sparsity)
        validator.validate_and_update(masks)
        apply_masks_to_model(model, masks)

        pruned_loss = compute_loss(model, tokenizer, eval_texts, max_seq_len=max_seq_len, batch_size=batch_size)
        zeros_after, _ = count_zeros(model)
        delta = pruned_loss - baseline_loss
        results.append((sparsity, pruned_loss, delta, zeros_after))

        print(f"\n=== Sparsity {sparsity:.1%} ===")
        print(f"  Loss:     {pruned_loss:.4f} (delta: {delta:+.4f})")
        print(f"  Zeros:    {zeros_after}/{total_params} ({zeros_after / total_params:.2%})")

    print(f"\n{'='*60}")
    print(f"Summary: {model_id} on {dataset_subset}")
    print(f"{'Sparsity':>10} {'Loss':>10} {'Delta':>10} {'Actual %':>10}")
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for sparsity, loss, delta, zeros in results:
        print(f"{sparsity:>10.1%} {loss:>10.4f} {delta:>+10.4f} {zeros / total_params:>10.2%}")


if __name__ == "__main__":
    main()
