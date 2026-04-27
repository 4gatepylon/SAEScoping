"""
Quick WANDA pruning on gemma-2-2b-it.

Computes saliency, caches it, then prunes at several sparsity levels
and reports loss on a held-out eval set.

Usage:
    python experiments/deleteme_wanda.py
    python experiments/deleteme_wanda.py --device cpu
    python experiments/deleteme_wanda.py --sparsity 0.5
    python experiments/deleteme_wanda.py \
        --model-id google/gemma-2-9b-it \
        --device cuda:0 \
        --batch-size 2 \
        --max-seq-len 1024 \
        --n-calibration 1024 \
        --n-eval 256 \
"""

from __future__ import annotations

import copy
from pathlib import Path

import click
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_scoping.datasets.qa_datasets import load_qa_dataset, format_as_sft_text
from sae_scoping.training.saliency.wanda import (
    compute_wanda_saliency,
    compute_wanda_masks,
    apply_masks_to_model,
)

CACHE_DIR = Path(__file__).parent / ".cache" / "deleteme_wanda_cache"
SALIENCY_PATH = CACHE_DIR / "wanda_saliency.safetensors"

# TODO(Claude) why is this not already a utility method? Is this code duplicted everywhere?
@torch.no_grad()
def compute_loss(model, tokenizer, texts, max_seq_len, device, batch_size):
    assert isinstance(batch_size, int) and batch_size > 0, "Expected batch_size > 0."
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, len(texts), batch_size):
        text_batch = texts[i : i + batch_size]
        tokens = tokenizer(
            text_batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            padding=True,
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        labels = input_ids.clone()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        n_tokens = attention_mask.sum().item()
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens
    return total_loss / total_tokens


@click.command()
@click.option("--model-id", default="google/gemma-2-2b-it")
@click.option("--n-calibration", default=64)
@click.option("--n-eval", default=32)
@click.option("--batch-size", default=2, show_default=True)
@click.option("--max-seq-len", default=512)
@click.option("--sparsity", default=None, type=float, help="Single sparsity. Default: sweep [0.1..0.7]")
@click.option("--device", default=None, help="Force device (default: auto)")
@click.option("--no-cache", is_flag=True, help="Recompute saliency even if cached")
def main(model_id, n_calibration, n_eval, batch_size, max_seq_len, sparsity, device, no_cache):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Device: {device}")

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    print(f"Loading {model_id} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device, attn_implementation="eager",
    )
    model.eval()

    print("Loading StemQA biology dataset...")
    n_total = n_calibration + n_eval
    ds = load_qa_dataset("4gate/StemQAMixture", subset="biology", n=n_total, seed=42)
    all_texts = format_as_sft_text(ds, tokenizer)
    cal_texts = all_texts[:n_calibration]
    eval_texts = all_texts[n_calibration:]
    print(f"  calibration: {len(cal_texts)}, eval: {len(eval_texts)}")

    # --- Saliency (cached) ---
    if SALIENCY_PATH.exists() and not no_cache:
        from safetensors.torch import load_file
        print(f"Loading cached saliency from {SALIENCY_PATH}")
        saliency = load_file(str(SALIENCY_PATH))
    else:
        print("Computing WANDA saliency...")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        saliency = compute_wanda_saliency(
            model, tokenizer, cal_texts,
            max_seq_len=max_seq_len, batch_size=batch_size, save_path=SALIENCY_PATH,
        )

    # --- Baseline loss ---
    print("Computing baseline loss (unpruned)...")
    baseline_loss = compute_loss(
        model, tokenizer, eval_texts, max_seq_len, device, batch_size
    )
    print(f"  baseline loss: {baseline_loss:.4f}")

    # --- Prune & evaluate ---
    sparsities = [sparsity] if sparsity is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # TODO(hadriano) Not sure if this makes sense for our size model
    original_state = copy.deepcopy(model.state_dict())

    print(f"\n{'sparsity':>10} {'n_zeroed':>12} {'loss':>10} {'loss_delta':>12}")
    print("-" * 48)

    for s in sparsities:
        model.load_state_dict(original_state)
        masks = compute_wanda_masks(saliency, s)
        n_zeroed = apply_masks_to_model(model, masks)
        loss = compute_loss(model, tokenizer, eval_texts, max_seq_len, device, batch_size)
        delta = loss - baseline_loss
        print(f"{s:>10.1%} {n_zeroed:>12,} {loss:>10.4f} {delta:>+12.4f}")

    print(f"\nSaliency cached at: {SALIENCY_PATH}")


if __name__ == "__main__":
    main()

