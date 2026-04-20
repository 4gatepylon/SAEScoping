"""
Unified sparsity sweep across all pruning methods and model sizes.

Logs train loss, test loss, and LLM judge utility at each sparsity level.
Designed to run on a single GPU.

Methods:
  - wanda:    |W| * ||X||_2, per-row pruning
  - random:   uniform random scores, global threshold
  - taylor:   |grad * weight| (requires pre-computed gradient map)
  - gradient: |grad| (requires pre-computed gradient map)
  - sparse_llm: iterative alternating optimization

Models (all instruction-tuned, with chat templates):
  - google/gemma-2-2b-it   (single A100 easily)
  - google/gemma-2-9b-it   (single A100 80GB)
  - google/gemma-3-12b-it  (single A100 80GB, bf16)

Usage:
  # Wanda sweep on gemma-2-2b-it:
  python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it

  # Random baseline on gemma-2-9b-it:
  python sweep_sparsity.py --method random --model google/gemma-2-9b-it

  # Taylor sweep (requires gradient map file):
  python sweep_sparsity.py --method taylor --model google/gemma-2-2b-it \
      --saliency-path ./biology/ema_grads.safetensors

  # SparseLLM on gemma-3-12b-it:
  python sweep_sparsity.py --method sparse_llm --model google/gemma-3-12b-it

  # Custom sparsity levels:
  python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it \
      --sparsity-levels 0.1,0.3,0.5,0.7,0.9

  # Skip LLM judge (faster, loss only):
  python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it --no-judge
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Optional

import click
import torch
import wandb
from datasets import load_dataset
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from sae_scoping.training.weight_pruning import (
    save_original_weights,
    restore_original_weights,
)

# Default coarse sweep levels
DEFAULT_SPARSITY_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split_texts(
    tokenizer: PreTrainedTokenizerBase,
    dataset_name: str,
    dataset_subset: str,
    n_train: int,
    n_test: int,
    n_calibration: int,
) -> tuple[list[str], list[str], list[str]]:
    """Load non-overlapping train/test/calibration splits from a StemQA dataset."""
    ds = load_dataset(dataset_name, dataset_subset, split="train")
    ds = ds.shuffle(seed=42)
    total_needed = n_train + n_test + n_calibration
    assert len(ds) >= total_needed, f"Dataset has {len(ds)} < {total_needed} needed"

    def format_texts(start, end):
        texts = []
        for i in range(start, end):
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

    calib_texts = format_texts(0, n_calibration)
    train_texts = format_texts(n_calibration, n_calibration + n_train)
    test_texts = format_texts(n_calibration + n_train, n_calibration + n_train + n_test)
    return train_texts, test_texts, calib_texts


def load_eval_questions(
    dataset_name: str,
    dataset_subset: str,
    n_samples: int,
    seed: int = 999,
) -> tuple[list[str], list[str]]:
    """Load questions and answers for LLM judge evaluation."""
    ds = load_dataset(dataset_name, dataset_subset, split="train")
    ds = ds.shuffle(seed=seed)
    n = min(n_samples, len(ds))
    questions = [str(ds[i]["question"]) for i in range(n)]
    answers = [str(ds[i]["answer"]) for i in range(n)]
    return questions, answers


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_loss(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_seq_len: int = 1024,
    batch_size: int = 4,
) -> float:
    """Mean cross-entropy loss over texts."""
    model.eval()
    try:
        device = model.device
    except AttributeError:
        device = next(p.device for p in model.parameters())

    old_pad = tokenizer.padding_side
    tokenizer.padding_side = "right"
    total_loss, n_batches = 0.0, 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tok = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_seq_len,
        )
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += out.loss.item()
        n_batches += 1
    tokenizer.padding_side = old_pad
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Pruning methods
# ---------------------------------------------------------------------------


def prune_with_method(
    method: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    sparsity: float,
    max_seq_len: int,
    saliency_path: Optional[str] = None,
    sparse_llm_iterations: int = 4,
    # Pre-computed saliency maps (cached between sparsity levels)
    _cached_wanda_saliency: Optional[dict] = None,
    _cached_saliency_map: Optional[dict] = None,
) -> tuple[int, Optional[dict]]:
    """Apply pruning method at given sparsity. Returns (n_zeroed, cached_saliency)."""

    if method == "wanda":
        from sae_scoping.training.saliency.wanda import (
            compute_wanda_saliency,
            compute_wanda_masks,
            apply_masks_to_model,
        )
        if _cached_wanda_saliency is None:
            saliency = compute_wanda_saliency(
                model, tokenizer, calibration_texts, max_seq_len=max_seq_len,
            )
        else:
            saliency = _cached_wanda_saliency
        masks = compute_wanda_masks(saliency, sparsity)
        n_zeroed = apply_masks_to_model(model, masks)
        return n_zeroed, saliency

    elif method == "random":
        from sae_scoping.training.saliency.random import make_random_map
        from sae_scoping.training.weight_pruning import compute_keep_masks, apply_keep_masks_streaming
        if _cached_saliency_map is None:
            saliency = make_random_map(model, seed=42)
        else:
            saliency = _cached_saliency_map
        # Global threshold pruning
        masks = {}
        import torch
        # Compute global threshold
        all_scores = torch.cat([s.flatten() for s in saliency.values()])
        threshold = torch.quantile(all_scores.float(), sparsity).item()
        for name, scores in saliency.items():
            masks[name] = (scores > threshold)
        n_zeroed = 0
        for name, param in model.named_parameters():
            if name not in masks:
                continue
            mask = masks[name].to(device=param.device, dtype=param.dtype)
            n_zeroed += int((mask == 0).sum().item())
            param.data.mul_(mask)
        return n_zeroed, saliency

    elif method in ("taylor", "gradient"):
        from sae_scoping.training.weight_pruning import compute_keep_masks, apply_keep_masks_streaming
        if saliency_path is None:
            raise click.UsageError(
                f"--saliency-path is required for method={method}. "
                "Run sae_scoping/training/saliency/grad.py first to generate gradient maps."
            )
        if _cached_saliency_map is None:
            raw_saliency = load_file(saliency_path)
            if method == "taylor":
                from sae_scoping.training.saliency.taylor import make_taylor_map
                saliency = make_taylor_map(raw_saliency, model)
            else:
                # gradient: just use |grad|
                saliency = {k: v.abs() for k, v in raw_saliency.items()}
        else:
            saliency = _cached_saliency_map
        # Global threshold
        all_scores = torch.cat([s.flatten().float() for s in saliency.values()])
        threshold = torch.quantile(all_scores, sparsity).item()
        n_zeroed = 0
        for name, param in model.named_parameters():
            if name not in saliency:
                continue
            mask = (saliency[name].to(param.device) > threshold).to(param.dtype)
            n_zeroed += int((mask == 0).sum().item())
            param.data.mul_(mask)
        return n_zeroed, saliency

    elif method == "sparse_llm":
        from sae_scoping.training.saliency.sparse_llm import compute_sparse_llm_masks
        from sae_scoping.training.saliency.wanda import apply_masks_to_model
        # SparseLLM computes masks directly (no reusable saliency map across sparsities)
        masks = compute_sparse_llm_masks(
            model, tokenizer, calibration_texts,
            sparsity=sparsity,
            n_iterations=sparse_llm_iterations,
            max_seq_len=max_seq_len,
        )
        n_zeroed = apply_masks_to_model(model, masks)
        return n_zeroed, None  # Can't cache — masks are sparsity-specific

    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--method", type=click.Choice(["wanda", "random", "taylor", "gradient", "sparse_llm"]), required=True)
@click.option("--model", "model_id", required=True, help="HuggingFace model ID (e.g. google/gemma-2-2b-it)")
@click.option("--dataset-name", default="4gate/StemQAMixture")
@click.option("--dataset-subset", default="biology")
@click.option("--sparsity-levels", default=None, help="Comma-separated sparsity levels (default: 0,0.1,...,0.9)")
@click.option("--n-calibration", default=128, help="Calibration samples for Wanda/SparseLLM")
@click.option("--n-train", default=200, help="Train set samples for loss")
@click.option("--n-test", default=200, help="Test set samples for loss")
@click.option("--n-judge-samples", default=50, help="Samples for LLM judge evaluation")
@click.option("--max-seq-len", default=1024)
@click.option("--saliency-path", default=None, help="Pre-computed gradient map for taylor/gradient methods")
@click.option("--sparse-llm-iterations", default=4, help="SparseLLM alternating opt iterations")
@click.option("--no-judge", is_flag=True, help="Skip LLM judge evaluation (faster)")
@click.option("--wandb-project", default="sae-scoping-baselines")
@click.option("--device", default=None, help="CUDA device (auto-detected if not set)")
def main(
    method, model_id, dataset_name, dataset_subset, sparsity_levels,
    n_calibration, n_train, n_test, n_judge_samples, max_seq_len,
    saliency_path, sparse_llm_iterations, no_judge, wandb_project, device,
):
    # Auto-detect free GPU
    if device is None:
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    # Parse sparsity levels
    if sparsity_levels:
        levels = sorted(float(s.strip()) for s in sparsity_levels.split(","))
    else:
        levels = DEFAULT_SPARSITY_LEVELS

    model_slug = model_id.replace("/", "--")
    print(f"Method: {method}")
    print(f"Model: {model_id}")
    print(f"Device: {device}")
    print(f"Sparsity levels: {levels}")
    print(f"Dataset: {dataset_name}/{dataset_subset}")

    # Load model and tokenizer
    print(f"\nLoading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager",
    )

    # Load data splits
    print("Loading data splits...")
    train_texts, test_texts, calib_texts = load_split_texts(
        tokenizer, dataset_name, dataset_subset,
        n_train=n_train, n_test=n_test, n_calibration=n_calibration,
    )
    print(f"  Calibration: {len(calib_texts)}, Train: {len(train_texts)}, Test: {len(test_texts)}")

    # Load judge eval questions (separate from train/test)
    judge_questions, judge_answers = None, None
    if not no_judge:
        judge_questions, judge_answers = load_eval_questions(
            dataset_name, dataset_subset, n_judge_samples, seed=777,
        )
        print(f"  Judge eval: {len(judge_questions)} questions")

    # Save original weights
    print("Saving original weights for restoration...")
    original_weights = save_original_weights(model)

    # Init wandb
    run_name = f"{method}/{model_slug}/{dataset_subset}"
    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "method": method,
            "model": model_id,
            "dataset": f"{dataset_name}/{dataset_subset}",
            "sparsity_levels": levels,
            "n_calibration": n_calibration,
            "n_train": n_train,
            "n_test": n_test,
            "n_judge_samples": n_judge_samples,
            "max_seq_len": max_seq_len,
            "saliency_path": saliency_path,
            "sparse_llm_iterations": sparse_llm_iterations,
        },
    )

    # Run sweep
    cached_saliency = None
    results = []

    for sp in levels:
        print(f"\n{'='*70}")
        print(f"  Sparsity: {sp:.0%}  |  Method: {method}  |  Model: {model_slug}")
        print(f"{'='*70}")

        # Restore original weights
        restore_original_weights(model, original_weights)

        n_zeroed = 0
        actual_sparsity = 0.0
        prune_time = 0.0

        if sp > 0:
            t0 = time.time()
            # Determine which cache arg to use
            cache_kwarg = {}
            if method == "wanda":
                cache_kwarg["_cached_wanda_saliency"] = cached_saliency
            elif method in ("random", "taylor", "gradient"):
                cache_kwarg["_cached_saliency_map"] = cached_saliency

            n_zeroed, new_cache = prune_with_method(
                method, model, tokenizer, calib_texts,
                sparsity=sp, max_seq_len=max_seq_len,
                saliency_path=saliency_path,
                sparse_llm_iterations=sparse_llm_iterations,
                **cache_kwarg,
            )
            if new_cache is not None:
                cached_saliency = new_cache
            prune_time = time.time() - t0

            n_total_prunable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            actual_sparsity = n_zeroed / n_total_prunable if n_total_prunable > 0 else 0.0

        # Compute train loss
        t0 = time.time()
        train_loss = compute_loss(model, tokenizer, train_texts, max_seq_len=max_seq_len)
        train_loss_time = time.time() - t0

        # Compute test loss
        t0 = time.time()
        test_loss = compute_loss(model, tokenizer, test_texts, max_seq_len=max_seq_len)
        test_loss_time = time.time() - t0

        result = {
            "sparsity": sp,
            "actual_sparsity": actual_sparsity,
            "n_zeroed": n_zeroed,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "prune_time_s": prune_time,
        }

        print(f"  train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  "
              f"actual_sparsity={actual_sparsity:.4f}  prune_time={prune_time:.1f}s")

        # LLM judge evaluation
        if not no_judge and judge_questions is not None:
            try:
                from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval
                evaluator = OneClickLLMJudgeScopingEval(
                    n_samples=n_judge_samples,
                    train_domain=dataset_subset,
                )
                domain_questions = {dataset_subset: judge_questions}
                domain_answers = {dataset_subset: judge_answers}
                scores, _ = evaluator.evaluate(
                    model, tokenizer, domain_questions,
                    domain_answers=domain_answers,
                )
                result.update(scores)
                print(f"  LLM judge: {json.dumps({k: f'{v:.3f}' for k, v in scores.items()})}")
            except Exception as e:
                print(f"  LLM judge failed: {e}")

        # Log to wandb
        wandb.log(result)
        results.append(result)
        torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*70}")
    print(f"Summary: {method} on {model_slug}/{dataset_subset}")
    print(f"{'='*70}")
    print(f"{'Sparsity':>10} {'Train Loss':>12} {'Test Loss':>12} {'Actual Sp':>10}")
    print("-" * 50)
    for r in results:
        print(f"{r['sparsity']:>10.0%} {r['train_loss']:>12.4f} {r['test_loss']:>12.4f} {r['actual_sparsity']:>10.4f}")

    wandb.finish()
    del original_weights, cached_saliency
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
