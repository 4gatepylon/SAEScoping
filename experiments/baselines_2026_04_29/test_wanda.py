"""Run Wanda pruning on a model and report before/after loss and sparsity."""

from pathlib import Path

import click
import torch

from sae_scoping.datasets.qa_datasets import load_qa_dataset, format_as_sft_text
from sae_scoping.evaluation.loss import compute_loss, count_zeros
from sae_scoping.training.saliency.wanda import compute_wanda_saliency, compute_wanda_masks, apply_masks_to_model
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
@click.option("--sparsity", default=0.5, show_default=True, help="Fraction of weights to prune per row.")
@click.option("--cache-dir", default="./cache", show_default=True, help="Directory for cached saliency maps.")
@click.option("--no-cache", is_flag=True, help="Recompute saliency even if cached.")
@click.option("--device", default="cuda:0", show_default=True)
def main(model_id, dataset_name, dataset_subset, n_calibration, n_eval, max_seq_len, batch_size, sparsity, cache_dir, no_cache, device):
    """Run Wanda pruning on a model and print before/after metrics."""
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

    masks = compute_wanda_masks(saliency_map, sparsity)
    n_zeroed = apply_masks_to_model(model, masks)

    print(f"\n=== After Wanda pruning (sparsity={sparsity:.0%}) ===")
    pruned_loss = compute_loss(model, tokenizer, eval_texts, max_seq_len=max_seq_len, batch_size=batch_size)
    zeros_after, _ = count_zeros(model)
    print(f"  Loss:     {pruned_loss:.4f}")
    print(f"  Sparsity: {zeros_after}/{total_params} ({zeros_after / total_params:.2%})")
    print(f"  Weights zeroed by Wanda: {n_zeroed:,}")
    print(f"  Loss delta: {pruned_loss - baseline_loss:+.4f}")


if __name__ == "__main__":
    main()
