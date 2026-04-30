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
from sae_scoping.utils.click_utils import load_yaml_config, parse_comma_separated_floats
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
@click.option("--nn-linear-sparsity", "-s", default=None, help="Per-row sparsity within nn.Linear layers only (embeddings/head untouched). Comma-separated for sweep (e.g. -s 0.2,0.4,0.6).")
@click.option("--cache-dir", default="./cache", show_default=True, help="Directory for cached saliency maps.")
@click.option("--no-cache", is_flag=True, help="Recompute saliency even if cached.")
@click.option("--low-memory", is_flag=True, help="Skip mask monotonicity validation to save CPU memory.")
@click.option("--device", default="cuda:0", show_default=True)
def main(model_id, dataset_name, dataset_subset, n_calibration, n_eval, max_seq_len, batch_size, nn_linear_sparsity, cache_dir, no_cache, low_memory, device):
    """Run Wanda pruning sweep: compute saliency once, then evaluate at each sparsity level from low to high."""
    sparsities = parse_comma_separated_floats(nn_linear_sparsity, default=[0.5])
    print(f"Sweep nn.Linear sparsities: {[f'{s:.1%}' for s in sparsities]}")

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
    print(f"  Loss:            {baseline_loss:.4f}")
    print(f"  Model sparsity:  {zeros_before}/{total_params} ({zeros_before / total_params:.2%})")

    saliency_file = cache_path(Path(cache_dir), model_id, dataset_subset, "wanda_saliency.safetensors")
    saliency_map = load_or_compute_safetensors(
        path=saliency_file,
        compute_fn=lambda: compute_wanda_saliency(model, tokenizer, calib_texts, max_seq_len=max_seq_len, batch_size=batch_size),
        no_cache=no_cache,
        label="Wanda saliency",
    )

    linear_total = sum(t.numel() for t in saliency_map.values())

    validator = MaskSubsetValidator(enabled=not low_memory)
    results = []
    for sparsity in sparsities:
        masks = compute_wanda_masks(saliency_map, sparsity)
        validator.validate_and_update(masks)
        apply_masks_to_model(model, masks)

        pruned_loss = compute_loss(model, tokenizer, eval_texts, max_seq_len=max_seq_len, batch_size=batch_size)
        zeros_after, _ = count_zeros(model)
        linear_zeros = sum(int((~m).sum().item()) for m in masks.values())
        delta = pruned_loss - baseline_loss
        results.append((sparsity, pruned_loss, delta, zeros_after, linear_zeros))

        print(f"\n=== nn.Linear sparsity {sparsity:.1%} ===")
        print(f"  Loss:              {pruned_loss:.4f} (delta: {delta:+.4f})")
        print(f"  nn.Linear sparsity: {linear_zeros}/{linear_total} ({linear_zeros / linear_total:.2%})")
        print(f"  Whole-model sparsity: {zeros_after}/{total_params} ({zeros_after / total_params:.2%})")

    print(f"\n{'='*70}")
    print(f"Summary: {model_id} on {dataset_subset}")
    print(f"{'nn.Linear %':>12} {'Loss':>10} {'Delta':>10} {'Linear %':>10} {'Model %':>10}")
    print(f"{'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for sparsity, loss, delta, zeros, lin_zeros in results:
        print(f"{sparsity:>12.1%} {loss:>10.4f} {delta:>+10.4f} {lin_zeros / linear_total:>10.2%} {zeros / total_params:>10.2%}")


if __name__ == "__main__":
    main()
