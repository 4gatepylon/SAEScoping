"""
SparseLLM pruning baseline experiment.

Usage:
  # Quick test:
  python run_sparse_llm.py --model google/gemma-2-2b-it --n-calibration 16 --sparsity 0.5

  # Full run with sweep:
  python run_sparse_llm.py --model google/gemma-2-2b-it --sweep

  # Custom sparsity levels:
  python run_sparse_llm.py --model google/gemma-2-2b-it --sparsity-levels 0.3,0.5,0.7
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

from sae_scoping.training.saliency.sparse_llm import (
    compute_sparse_llm_masks,
)
from sae_scoping.training.saliency.wanda import apply_masks_to_model
from sae_scoping.training.weight_pruning import (
    save_original_weights,
    restore_original_weights,
)


def _load_calibration_texts(tokenizer, dataset_name, dataset_subset, n_samples, seed=42):
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
@click.option("--n-calibration", default=64, help="Calibration samples (fewer for memory)")
@click.option("--n-eval", default=100, help="Validation samples")
@click.option("--max-seq-len", default=1024)
@click.option("--sparsity", default=0.5)
@click.option("--sparsity-levels", default=None, help="Comma-separated sparsity levels")
@click.option("--sweep", is_flag=True, help="Sweep default levels")
@click.option("--n-iterations", default=4, help="SparseLLM alternating opt iterations")
@click.option("--alpha", default=5.0)
@click.option("--beta", default=5.0)
@click.option("--gamma", default=5.0)
@click.option("--wandb-project", default="sae-scoping-baselines")
@click.option("--device", default="cuda:0")
def main(
    model_id, dataset_name, dataset_subset, n_calibration, n_eval,
    max_seq_len, sparsity, sparsity_levels, sweep, n_iterations,
    alpha, beta, gamma, wandb_project, device,
):
    device = torch.device(device)

    if sparsity_levels:
        levels = [float(s.strip()) for s in sparsity_levels.split(",")]
    elif sweep:
        levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    else:
        levels = [sparsity]

    print(f"Model: {model_id}")
    print(f"Sparsity levels: {levels}")
    print(f"SparseLLM params: n_iterations={n_iterations}, alpha={alpha}, beta={beta}, gamma={gamma}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager",
    )

    calib_texts = _load_calibration_texts(
        tokenizer, dataset_name, dataset_subset, n_calibration, seed=42,
    )
    val_texts = _load_calibration_texts(
        tokenizer, dataset_name, dataset_subset, n_eval, seed=123,
    )

    original_weights = save_original_weights(model)

    model_slug = model_id.replace("/", "--")
    run_name = f"sparse_llm/{model_slug}/{dataset_subset}"
    wandb.init(
        project=wandb_project, name=run_name,
        config={
            "method": "sparse_llm",
            "model": model_id,
            "dataset": f"{dataset_name}/{dataset_subset}",
            "n_calibration": n_calibration,
            "n_iterations": n_iterations,
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "sparsity_levels": levels,
        },
    )

    results = []
    for sp in levels:
        print(f"\n{'='*60}")
        print(f"Sparsity: {sp:.0%}")
        print(f"{'='*60}")

        restore_original_weights(model, original_weights)

        if sp > 0:
            t0 = time.time()
            masks = compute_sparse_llm_masks(
                model, tokenizer, calib_texts,
                sparsity=sp, n_iterations=n_iterations,
                alpha=alpha, beta=beta, gamma=gamma,
                max_seq_len=max_seq_len,
            )
            n_zeroed = apply_masks_to_model(model, masks)
            n_total = sum(p.numel() for name, p in model.named_parameters() if name in masks)
            actual_sparsity = n_zeroed / n_total if n_total > 0 else 0.0
            prune_time = time.time() - t0
            del masks
        else:
            n_zeroed = 0
            n_total = sum(p.numel() for p in model.parameters())
            actual_sparsity = 0.0
            prune_time = 0.0

        val_loss = _compute_val_loss(model, tokenizer, val_texts, max_seq_len=max_seq_len)

        result = {
            "sparsity": sp,
            "actual_sparsity": actual_sparsity,
            "n_zeroed": n_zeroed,
            "val_loss": val_loss,
            "prune_time_s": prune_time,
        }
        print(f"  val_loss={val_loss:.4f}  actual_sparsity={actual_sparsity:.4f}  prune_time={prune_time:.1f}s")

        wandb.log({
            "sparsity": sp,
            "actual_sparsity": actual_sparsity,
            "val_loss": val_loss,
            "n_zeroed": n_zeroed,
            "prune_time_s": prune_time,
        })
        results.append(result)
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("Summary:")
    for r in results:
        print(f"  sparsity={r['sparsity']:.0%}  val_loss={r['val_loss']:.4f}  actual={r['actual_sparsity']:.4f}")

    wandb.finish()
    del original_weights
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
