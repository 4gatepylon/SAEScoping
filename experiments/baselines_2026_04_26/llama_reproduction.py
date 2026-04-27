"""Faithful reproduction of Wanda (Sun et al., 2023) on LLaMA-2-7B.

Uses our sae_scoping.training.saliency.wanda library with the paper's setup:
- Model: LLaMA-2-7B (meta-llama/Llama-2-7b-hf)
- Calibration: 128 samples from C4 (allenai/c4, streaming), seq_len=2048
- Evaluation: WikiText-2 test set perplexity
- Sparsities: 50%, 60%, 70%, 80% unstructured

Reference numbers from Wanda paper (Table 1):

  LLaMA-7B (v1) WikiText-2 ppl:
    Dense: 5.68 | Wanda: 7.26 / 10.98 / 26.30 / 128.33
    Magnitude: 15.56 / 163.89 / 12360 / 93661
    SparseGPT: 7.22 / 10.41 / 20.21 / 66.31

  LLaMA-2-7B WikiText-2 ppl:
    Dense: 5.47 | Wanda: 6.92 / 10.13 / 23.41 / 107.89

Since we default to LLaMA-2-7B, compare against the second row.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_scoping.training.saliency.wanda import (
    apply_masks_to_model,
    compute_wanda_masks,
    compute_wanda_saliency,
)

DEFAULT_MODEL_ID = "meta-llama/Llama-2-7b-hf"
N_CALIBRATION = 128
MAX_SEQ_LEN = 2048

PAPER_REFERENCE = {
    "meta-llama/Llama-2-7b-hf": {
        "dense": 5.47,
        "wanda": {0.5: 6.92, 0.6: 10.13, 0.7: 23.41, 0.8: 107.89},
    },
    "huggyllama/llama-7b": {
        "dense": 5.68,
        "wanda": {0.5: 7.26, 0.6: 10.98, 0.7: 26.30, 0.8: 128.33},
    },
}


def load_c4_calibration(
    tokenizer,
    n_samples: int = N_CALIBRATION,
    max_seq_len: int = MAX_SEQ_LEN,
) -> list[str]:
    """Load calibration data from C4, matching Wanda paper setup.

    Paper uses 128 sequences from C4's first shard, each at full context
    length. We stream from allenai/c4 and keep texts whose tokenization
    is at least max_seq_len tokens (our library truncates internally).
    """
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

    texts: list[str] = []
    for sample in dataset:
        text = sample["text"]
        n_tokens = len(tokenizer(text, truncation=False)["input_ids"])
        if n_tokens >= max_seq_len:
            texts.append(text)
        if len(texts) >= n_samples:
            break

    if len(texts) < n_samples:
        print(
            f"[warn] Only found {len(texts)} texts with >= {max_seq_len} tokens "
            f"(requested {n_samples}). Using all of them."
        )
    return texts


@torch.no_grad()
def eval_wikitext_ppl(
    model,
    tokenizer,
    max_seq_len: int = MAX_SEQ_LEN,
) -> float:
    """WikiText-2 test set perplexity (standard LM evaluation).

    Concatenates all test text, tokenizes as one sequence, processes in
    non-overlapping windows of max_seq_len, computes mean cross-entropy
    weighted by token count per window.
    """
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"]
    seq_len = input_ids.shape[1]

    try:
        device = model.device
    except AttributeError:
        device = next(model.parameters()).device

    total_nll = 0.0
    total_tokens = 0

    for begin in tqdm(range(0, seq_len, max_seq_len), desc="  eval ppl"):
        end = min(begin + max_seq_len, seq_len)
        chunk = input_ids[:, begin:end].to(device)
        outputs = model(input_ids=chunk, labels=chunk)
        n_tokens = chunk.shape[1] - 1
        if n_tokens > 0:
            total_nll += outputs.loss.float().item() * n_tokens
            total_tokens += n_tokens

    ppl = torch.exp(torch.tensor(total_nll / total_tokens)).item()
    return ppl


def _load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(model_id: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        attn_implementation="eager",
    )
    model.eval()
    return model


def _print_summary(results: dict, model_id: str) -> None:
    ref = PAPER_REFERENCE.get(model_id, {})
    header = f"{'Sparsity':<12} {'Our Wanda':>10}"
    if ref:
        header += f" {'Paper Wanda':>12}"
    print(f"\n{'=' * len(header)}")
    print(f"  Wanda Reproduction: {model_id}")
    print(f"  WikiText-2 Perplexity")
    print(f"{'=' * len(header)}")
    print(header)
    print("-" * len(header))

    if "dense" in results:
        line = f"{'Dense':<12} {results['dense']['ppl']:>10.2f}"
        if "dense" in ref:
            line += f" {ref['dense']:>12.2f}"
        print(line)

    for key in sorted(k for k in results if k.startswith("wanda_sp")):
        sp = float(key.replace("wanda_sp", ""))
        line = f"{sp:<12.0%} {results[key]['ppl']:>10.2f}"
        if "wanda" in ref and sp in ref["wanda"]:
            line += f" {ref['wanda'][sp]:>12.2f}"
        print(line)

    print("-" * len(header))


@click.command()
@click.option("--model-id", default=DEFAULT_MODEL_ID, show_default=True,
              help="HuggingFace model ID. Use huggyllama/llama-7b for LLaMA v1.")
@click.option("--sparsities", default="0.5,0.6,0.7,0.8", show_default=True,
              help="Comma-separated sparsity levels.")
@click.option("--n-calibration", default=N_CALIBRATION, show_default=True)
@click.option("--max-seq-len", default=MAX_SEQ_LEN, show_default=True)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--output", default="llama_wanda_results.json", show_default=True,
              help="Path to save JSON results.")
@click.option("--saliency-cache", default=None, type=click.Path(path_type=Path),
              help="Path to cache/load saliency map (.safetensors). Avoids recomputation.")
@click.option("--skip-dense", is_flag=True, default=False,
              help="Skip dense baseline evaluation (saves time if already measured).")
def main(model_id, sparsities, n_calibration, max_seq_len, device, output, saliency_cache, skip_dense):
    """Reproduce Wanda paper results on LLaMA-7B using our library."""
    sparsity_list = [float(s) for s in sparsities.split(",")]
    results = {}

    print(f"Loading {model_id}...")
    tokenizer = _load_tokenizer(model_id)
    model = _load_model(model_id, device)

    # --- Dense baseline ---
    if not skip_dense:
        print("\nEvaluating dense baseline...")
        dense_ppl = eval_wikitext_ppl(model, tokenizer, max_seq_len)
        results["dense"] = {"ppl": dense_ppl}
        print(f"  Dense ppl: {dense_ppl:.2f}")

    # --- Calibration ---
    print(f"\nLoading C4 calibration data ({n_calibration} samples)...")
    calibration_texts = load_c4_calibration(tokenizer, n_calibration, max_seq_len)
    print(f"  Loaded {len(calibration_texts)} calibration texts")

    # --- Saliency ---
    if saliency_cache and Path(saliency_cache).exists():
        print(f"\nLoading cached saliency from {saliency_cache}")
        saliency = load_file(str(saliency_cache))
    else:
        print("\nComputing Wanda saliency...")
        saliency = compute_wanda_saliency(
            model, tokenizer, calibration_texts, max_seq_len=max_seq_len,
        )
        if saliency_cache:
            Path(saliency_cache).parent.mkdir(parents=True, exist_ok=True)
            save_file(saliency, str(saliency_cache))
            print(f"  Cached saliency to {saliency_cache}")

    del model
    torch.cuda.empty_cache()

    # --- Prune + evaluate at each sparsity ---
    for sparsity in sparsity_list:
        print(f"\n{'=' * 60}")
        print(f"  Sparsity: {sparsity:.0%}")
        print(f"{'=' * 60}")

        model = _load_model(model_id, device)

        masks = compute_wanda_masks(saliency, sparsity)
        n_zeroed = apply_masks_to_model(model, masks)
        n_total = sum(p.numel() for name, p in model.named_parameters() if name in masks)
        actual_sparsity = n_zeroed / n_total if n_total > 0 else 0
        print(f"  Pruned {n_zeroed:,} / {n_total:,} ({actual_sparsity:.2%})")

        ppl = eval_wikitext_ppl(model, tokenizer, max_seq_len)
        results[f"wanda_sp{sparsity}"] = {
            "ppl": ppl,
            "n_zeroed": n_zeroed,
            "n_total": n_total,
            "actual_sparsity": actual_sparsity,
        }
        print(f"  Wanda ppl: {ppl:.2f}")

        del model, masks
        torch.cuda.empty_cache()

    # --- Summary ---
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output}")
    _print_summary(results, model_id)


if __name__ == "__main__":
    main()
