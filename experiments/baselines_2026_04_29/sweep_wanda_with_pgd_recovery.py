"""Wanda pruning + PGD recovery training sweep.

Prunes a model at a given nn.Linear sparsity, then runs PGD-projected SFT
to recover performance while maintaining the sparsity pattern. Evaluates
loss periodically during training via a TrainerCallback.
"""

from pathlib import Path

import click
import torch
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig

from sae_scoping.datasets.qa_datasets import load_nonoverlapping_splits
from sae_scoping.evaluation.loss import compute_loss, count_zeros
from sae_scoping.training.pgd_trainer import PGDSFTTrainer
from sae_scoping.training.saliency.wanda import compute_wanda_saliency, compute_wanda_masks, apply_masks_to_model
from sae_scoping.utils.cache import cache_path, load_or_compute_safetensors
from sae_scoping.utils.click_utils import load_yaml_config
from sae_scoping.utils.model_loading import load_model_and_tokenizer


class EvalLossCallback(TrainerCallback):
    """Evaluates cross-entropy loss on held-out texts at regular step intervals."""

    def __init__(self, eval_texts: list[str], tokenizer, max_seq_len: int, batch_size: int, eval_every_steps: int):
        self.eval_texts = eval_texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.eval_every_steps = eval_every_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.eval_every_steps != 0 or state.global_step == 0:
            return
        loss = compute_loss(model, self.tokenizer, self.eval_texts, max_seq_len=self.max_seq_len, batch_size=self.batch_size)
        zeros, total = count_zeros(model)
        print(f"  [eval @ step {state.global_step}] loss={loss:.4f}  sparsity={zeros}/{total} ({zeros / total:.2%})")


@click.command()
@click.option("--config", is_eager=True, expose_value=False, callback=load_yaml_config, type=click.Path(exists=True), help="YAML config file (CLI flags override).")
@click.option("--model-id", default="google/gemma-3-4b-it", show_default=True, help="HuggingFace model ID.")
@click.option("--dataset-name", default="4gate/StemQAMixture", show_default=True)
@click.option("--dataset-subset", default="biology", show_default=True)
@click.option("--n-calibration", default=128, show_default=True, help="Calibration samples for Wanda saliency.")
@click.option("--n-train", default=2000, show_default=True, help="Training samples for PGD recovery.")
@click.option("--n-eval", default=200, show_default=True, help="Evaluation samples (separate from calibration and train).")
@click.option("--max-seq-len", default=2048, show_default=True)
@click.option("--nn-linear-sparsity", "-s", default=0.5, show_default=True, type=float, help="Per-row sparsity within nn.Linear layers.")
@click.option("--cache-dir", default="./cache", show_default=True, help="Directory for cached saliency maps.")
@click.option("--no-cache", is_flag=True, help="Recompute saliency even if cached.")
@click.option("--device", default="cuda:0", show_default=True)
# PGD training arguments
@click.option("--learning-rate", default=2e-5, show_default=True, type=float)
@click.option("--num-train-epochs", default=1, show_default=True, type=int)
@click.option("--max-steps", default=-1, show_default=True, type=int, help="Override num-train-epochs if > 0.")
@click.option("--train-batch-size", default=2, show_default=True, type=int, help="Per-device training batch size.")
@click.option("--gradient-accumulation-steps", default=8, show_default=True, type=int)
@click.option("--warmup-ratio", default=0.05, show_default=True, type=float)
@click.option("--logging-steps", default=10, show_default=True, type=int)
@click.option("--eval-every-steps", default=50, show_default=True, type=int, help="Run eval callback every N steps.")
@click.option("--output-dir", default="./outputs_pgd", show_default=True, help="Directory for checkpoints.")
@click.option("--validate-sparsity/--no-validate-sparsity", default=True, show_default=True, help="Assert sparsity after each PGD step.")
@click.option("--report-to", default="none", show_default=True, help="Logging backend (none, wandb, tensorboard).")
def main(
    model_id, dataset_name, dataset_subset,
    n_calibration, n_train, n_eval, max_seq_len,
    nn_linear_sparsity, cache_dir, no_cache, device,
    learning_rate, num_train_epochs, max_steps, train_batch_size,
    gradient_accumulation_steps, warmup_ratio, logging_steps,
    eval_every_steps, output_dir, validate_sparsity, report_to,
):
    """Prune with Wanda, then recover with PGD-projected SFT training."""
    print(f"Loading tokenizer and model: {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, device=device)

    print(f"Loading non-overlapping splits from {dataset_name}/{dataset_subset}")
    calib_texts, train_texts, eval_texts = load_nonoverlapping_splits(
        tokenizer, dataset_name, dataset_subset, n_calibration=n_calibration, n_train=n_train, n_test=n_eval,
    )

    # --- Baseline ---
    print(f"\n=== Baseline (pre-pruning) ===")
    baseline_loss = compute_loss(model, tokenizer, eval_texts, max_seq_len=max_seq_len, batch_size=train_batch_size)
    zeros_before, total_params = count_zeros(model)
    print(f"  Loss:           {baseline_loss:.4f}")
    print(f"  Model sparsity: {zeros_before}/{total_params} ({zeros_before / total_params:.2%})")

    # --- Wanda pruning ---
    saliency_file = cache_path(Path(cache_dir), model_id, dataset_subset, "wanda_saliency.safetensors")
    saliency_map = load_or_compute_safetensors(
        path=saliency_file,
        compute_fn=lambda: compute_wanda_saliency(model, tokenizer, calib_texts, max_seq_len=max_seq_len),
        no_cache=no_cache,
        label="Wanda saliency",
    )

    masks = compute_wanda_masks(saliency_map, nn_linear_sparsity)
    n_zeroed = apply_masks_to_model(model, masks)
    del saliency_map

    linear_total = sum(m.numel() for m in masks.values())
    linear_zeros = sum(int((~m).sum().item()) for m in masks.values())
    post_prune_loss = compute_loss(model, tokenizer, eval_texts, max_seq_len=max_seq_len, batch_size=train_batch_size)
    zeros_after, _ = count_zeros(model)

    print(f"\n=== After Wanda pruning (nn.Linear sparsity={nn_linear_sparsity:.1%}) ===")
    print(f"  Loss:               {post_prune_loss:.4f} (delta: {post_prune_loss - baseline_loss:+.4f})")
    print(f"  nn.Linear sparsity: {linear_zeros}/{linear_total} ({linear_zeros / linear_total:.2%})")
    print(f"  Whole-model sparsity: {zeros_after}/{total_params} ({zeros_after / total_params:.2%})")

    # --- PGD recovery training ---
    print(f"\n=== PGD Recovery Training ===")
    sft_dataset = Dataset.from_dict({"text": train_texts})

    sft_config = SFTConfig(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        max_length=max_seq_len,
        bf16=True,
        report_to=report_to,
        save_strategy="no",
    )

    eval_callback = EvalLossCallback(
        eval_texts=eval_texts,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        batch_size=train_batch_size,
        eval_every_steps=eval_every_steps,
    )

    trainer = PGDSFTTrainer(
        masks=masks,
        validate_sparsity=validate_sparsity,
        model=model,
        args=sft_config,
        train_dataset=sft_dataset,
        callbacks=[eval_callback],
    )

    trainer.train()

    # --- Post-recovery evaluation ---
    print(f"\n=== After PGD Recovery ===")
    recovered_loss = compute_loss(model, tokenizer, eval_texts, max_seq_len=max_seq_len, batch_size=train_batch_size)
    zeros_final, _ = count_zeros(model)
    print(f"  Loss:               {recovered_loss:.4f} (delta from baseline: {recovered_loss - baseline_loss:+.4f})")
    print(f"  Recovery gain:      {post_prune_loss - recovered_loss:+.4f}")
    print(f"  Whole-model sparsity: {zeros_final}/{total_params} ({zeros_final / total_params:.2%})")

    print(f"\n{'='*70}")
    print(f"Summary: {model_id} on {dataset_subset} @ {nn_linear_sparsity:.1%} nn.Linear sparsity")
    print(f"  {'Baseline loss:':<25} {baseline_loss:.4f}")
    print(f"  {'Post-prune loss:':<25} {post_prune_loss:.4f} ({post_prune_loss - baseline_loss:+.4f})")
    print(f"  {'Post-recovery loss:':<25} {recovered_loss:.4f} ({recovered_loss - baseline_loss:+.4f})")
    print(f"  {'Recovery gain:':<25} {post_prune_loss - recovered_loss:+.4f}")


if __name__ == "__main__":
    main()
