"""
Wanda pruning baseline experiment.

Usage:
  # Quick test on gemma2-2b-it:
  python run_wanda.py --model google/gemma-2-2b-it --n-calibration 32 --sparsity 0.5

  # Full run with sweep and evaluation:
  python run_wanda.py --model google/gemma-2-2b-it --sweep --eval

  # Custom sparsity levels:
  python run_wanda.py --model google/gemma-2-2b-it --sparsity-levels 0.1,0.3,0.5,0.7
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import click
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_scoping.training.saliency.wanda import (
    compute_wanda_saliency,
    compute_wanda_masks,
    apply_masks_to_model,
)
from sae_scoping.training.weight_pruning import (
    save_original_weights,
    restore_original_weights,
)


def _load_calibration_texts(tokenizer, dataset_name, dataset_subset, n_samples, seed=42):
    """Load and format calibration texts from a StemQA-format dataset."""
    ds = load_dataset(dataset_name, dataset_subset, split="train")
    ds = ds.shuffle(seed=seed)
    n = min(n_samples, len(ds))
    texts = []
    for i in range(n):
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": str(ds[i]["question"])},
                {"role": "assistant", "content": str(ds[i]["answer"])},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return texts


@torch.no_grad()
def _compute_val_loss(model, tokenizer, val_texts, max_seq_len=1024, batch_size=4):
    """Compute mean cross-entropy loss on validation texts."""
    model.eval()
    try:
        device = model.device
    except AttributeError:
        device = next(p.device for p in model.parameters())

    old_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    total_loss = 0.0
    n_batches = 0
    for i in range(0, len(val_texts), batch_size):
        batch = val_texts[i:i + batch_size]
        tokens = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_seq_len,
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += out.loss.item()
        n_batches += 1
    tokenizer.padding_side = old_pad_side
    return total_loss / max(n_batches, 1)


@click.command()
@click.option("--model", "model_id", default="google/gemma-2-2b-it")
@click.option("--dataset-name", default="4gate/StemQAMixture")
@click.option("--dataset-subset", default="biology")
@click.option("--n-calibration", default=128, help="Calibration samples for Wanda")
@click.option("--n-eval", default=100, help="Validation samples for loss computation")
@click.option("--max-seq-len", default=1024)
@click.option("--sparsity", default=0.5, help="Single sparsity level (if not --sweep)")
@click.option("--sparsity-levels", default=None, help="Comma-separated sparsity levels for sweep")
@click.option("--sweep", is_flag=True, help="Sweep over default sparsity levels")
@click.option("--eval", "do_eval", is_flag=True, help="Run LLM judge evaluation (slower)")
@click.option("--wandb-project", default="sae-scoping-baselines")
@click.option("--device", default="cuda:0")
def main(
    model_id, dataset_name, dataset_subset, n_calibration, n_eval,
    max_seq_len, sparsity, sparsity_levels, sweep, do_eval, wandb_project, device,
):
    device = torch.device(device)

    # Determine sparsity levels
    if sparsity_levels:
        levels = [float(s.strip()) for s in sparsity_levels.split(",")]
    elif sweep:
        levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    else:
        levels = [sparsity]

    print(f"Model: {model_id}")
    print(f"Sparsity levels: {levels}")
    print(f"Calibration: {n_calibration} samples from {dataset_name}/{dataset_subset}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager",
    )

    # Load calibration and validation data
    calib_texts = _load_calibration_texts(
        tokenizer, dataset_name, dataset_subset, n_calibration, seed=42,
    )
    val_texts = _load_calibration_texts(
        tokenizer, dataset_name, dataset_subset, n_eval, seed=123,
    )

    # Compute saliency once (reuse across sparsity levels)
    print("\n--- Computing Wanda saliency scores ---")
    saliency_map = compute_wanda_saliency(
        model, tokenizer, calib_texts, max_seq_len=max_seq_len,
    )
    print(f"Saliency map: {len(saliency_map)} parameters scored")

    # Save original weights for restoration between sparsity levels
    print("Saving original weights...")
    original_weights = save_original_weights(model)

    # Init wandb
    model_slug = model_id.replace("/", "--")
    run_name = f"wanda/{model_slug}/{dataset_subset}"
    wandb.init(
        project=wandb_project, name=run_name,
        config={
            "method": "wanda",
            "model": model_id,
            "dataset": f"{dataset_name}/{dataset_subset}",
            "n_calibration": n_calibration,
            "n_eval": n_eval,
            "sparsity_levels": levels,
        },
    )

    # Sweep
    results = []
    for sp in levels:
        print(f"\n{'='*60}")
        print(f"Sparsity: {sp:.0%}")
        print(f"{'='*60}")

        # Restore original weights
        restore_original_weights(model, original_weights)

        if sp > 0:
            # Compute per-row masks at this sparsity and apply
            keep_masks = compute_wanda_masks(saliency_map, sp)
            n_zeroed = apply_masks_to_model(model, keep_masks)
            n_total = sum(p.numel() for name, p in model.named_parameters() if name in keep_masks)
            actual_sparsity = n_zeroed / n_total if n_total > 0 else 0.0
            del keep_masks
        else:
            n_zeroed = 0
            n_total = sum(p.numel() for p in model.parameters())
            actual_sparsity = 0.0

        # Compute validation loss
        t0 = time.time()
        val_loss = _compute_val_loss(model, tokenizer, val_texts, max_seq_len=max_seq_len)
        loss_time = time.time() - t0

        result = {
            "sparsity": sp,
            "actual_sparsity": actual_sparsity,
            "n_zeroed": n_zeroed,
            "val_loss": val_loss,
            "loss_time_s": loss_time,
        }
        print(f"  val_loss={val_loss:.4f}  actual_sparsity={actual_sparsity:.4f}  ({loss_time:.1f}s)")

        wandb.log({
            "sparsity": sp,
            "actual_sparsity": actual_sparsity,
            "val_loss": val_loss,
            "n_zeroed": n_zeroed,
        })

        # Optionally run LLM judge eval
        if do_eval:
            try:
                from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval

                # Load eval questions
                eval_ds = load_dataset(dataset_name, dataset_subset, split="train")
                eval_ds = eval_ds.shuffle(seed=999)
                domain_questions = {dataset_subset: [str(q) for q in eval_ds["question"][:50]]}
                domain_answers = {dataset_subset: [str(a) for a in eval_ds["answer"][:50]]}

                evaluator = OneClickLLMJudgeScopingEval(
                    n_samples=50,
                    train_domain=dataset_subset,
                )
                scores, _ = evaluator.evaluate(
                    model, tokenizer, domain_questions,
                    domain_answers=domain_answers,
                )
                result.update(scores)
                wandb.log({**scores, "sparsity": sp})
                print(f"  LLM judge scores: {json.dumps(scores, indent=2)}")
            except Exception as e:
                print(f"  LLM judge eval failed: {e}")

        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for r in results:
        print(f"  sparsity={r['sparsity']:.0%}  val_loss={r['val_loss']:.4f}  actual={r['actual_sparsity']:.4f}")

    wandb.finish()
    del original_weights, saliency_map
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
