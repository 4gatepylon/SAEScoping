"""
Unified sparsity sweep across all pruning methods and model sizes.

Thin CLI script — all reusable logic lives in sae_scoping.

Usage:
  python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it
  python sweep_sparsity.py --method taylor --model google/gemma-2-9b-it
  python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it --no-cache
  python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it --no-judge
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

from sae_scoping.datasets.qa_datasets import load_nonoverlapping_splits
from sae_scoping.evaluation.loss import compute_loss
from sae_scoping.training.saliency.dispatch import METHODS, compute_saliency, masks_for_sparsity
from sae_scoping.training.saliency.wanda import apply_masks_to_model
from sae_scoping.training.weight_pruning import save_original_weights, restore_original_weights

DEFAULT_SPARSITY_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_CACHE_DIR = Path("./saliency_cache")


@click.command()
@click.option("--method", type=click.Choice(list(METHODS)), required=True)
@click.option("--model", "model_id", required=True, help="HuggingFace model ID")
@click.option("--dataset-name", default="4gate/StemQAMixture")
@click.option("--dataset-subset", default="biology")
@click.option("--sparsity-levels", default=None, help="Comma-separated (default: 0,0.1,...,0.9)")
@click.option("--n-calibration", default=128, help="Calibration samples")
@click.option("--n-train", default=200, help="Train samples for loss")
@click.option("--n-test", default=200, help="Test samples for loss")
@click.option("--n-judge-samples", default=50, help="Samples for LLM judge")
@click.option("--max-seq-len", default=1024)
@click.option("--cache-dir", type=click.Path(path_type=Path), default=DEFAULT_CACHE_DIR)
@click.option("--no-cache", is_flag=True, help="Disable caching (always recompute)")
@click.option("--no-judge", is_flag=True, help="Skip LLM judge evaluation")
@click.option("--wandb-project", default="sae-scoping-baselines")
@click.option("--device", default=None, help="CUDA device (auto-detected if not set)")
def main(
    method, model_id, dataset_name, dataset_subset, sparsity_levels,
    n_calibration, n_train, n_test, n_judge_samples, max_seq_len,
    cache_dir, no_cache, no_judge, wandb_project, device,
):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    levels = (
        sorted(float(s.strip()) for s in sparsity_levels.split(","))
        if sparsity_levels else DEFAULT_SPARSITY_LEVELS
    )

    model_slug = model_id.replace("/", "--")
    print(f"Method: {method} | Model: {model_id} | Device: {device}")
    print(f"Sparsity levels: {levels}")
    print(f"Cache: {'disabled' if no_cache else str(cache_dir)}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device, attn_implementation="eager",
    )

    # Load data
    calib_texts, train_texts, test_texts = load_nonoverlapping_splits(
        tokenizer, dataset_name, dataset_subset,
        n_calibration=n_calibration, n_train=n_train, n_test=n_test,
    )
    judge_questions, judge_answers = None, None
    if not no_judge:
        ds_judge = load_dataset(dataset_name, dataset_subset, split="train").shuffle(seed=777)
        n_j = min(n_judge_samples, len(ds_judge))
        judge_questions = [str(ds_judge[i]["question"]) for i in range(n_j)]
        judge_answers = [str(ds_judge[i]["answer"]) for i in range(n_j)]

    # Step 1: Compute saliency (cached, done once)
    print(f"\n--- Computing saliency for {method} ---")
    saliency_data = compute_saliency(
        method, model, tokenizer, calib_texts, max_seq_len,
        cache_dir, model_id, dataset_subset,
        no_cache=no_cache, dataset_name=dataset_name,
    )

    # Save original weights for sweep
    original_weights = save_original_weights(model)

    # Wandb
    wandb.init(
        project=wandb_project, name=f"{method}/{model_slug}/{dataset_subset}",
        config=dict(method=method, model=model_id, dataset=f"{dataset_name}/{dataset_subset}",
                    sparsity_levels=levels, n_calibration=n_calibration, cache_dir=str(cache_dir)),
    )

    # Step 2: Sweep sparsity levels
    results = []
    for sp in levels:
        print(f"\n{'='*70}")
        print(f"  {method} @ {sp:.0%} | {model_slug}")
        print(f"{'='*70}")

        restore_original_weights(model, original_weights)

        n_zeroed, actual_sparsity, prune_time = 0, 0.0, 0.0
        if sp > 0:
            t0 = time.time()
            masks = masks_for_sparsity(method, saliency_data, sp)
            n_zeroed = apply_masks_to_model(model, masks)
            n_total = sum(p.numel() for n, p in model.named_parameters() if n in masks)
            actual_sparsity = n_zeroed / n_total if n_total > 0 else 0.0
            prune_time = time.time() - t0

        train_loss = compute_loss(model, tokenizer, train_texts, max_seq_len)
        test_loss = compute_loss(model, tokenizer, test_texts, max_seq_len)

        result = dict(sparsity=sp, actual_sparsity=actual_sparsity, n_zeroed=n_zeroed,
                      train_loss=train_loss, test_loss=test_loss, prune_time_s=prune_time)
        print(f"  train={train_loss:.4f}  test={test_loss:.4f}  actual_sp={actual_sparsity:.4f}")

        if not no_judge and judge_questions:
            try:
                from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval
                evaluator = OneClickLLMJudgeScopingEval(n_samples=n_judge_samples, train_domain=dataset_subset)
                scores, _ = evaluator.evaluate(
                    model, tokenizer, {dataset_subset: judge_questions},
                    domain_answers={dataset_subset: judge_answers},
                )
                result.update(scores)
                print(f"  judge: {json.dumps({k: f'{v:.3f}' for k, v in scores.items()})}")
            except Exception as e:
                print(f"  judge failed: {e}")

        wandb.log(result)
        results.append(result)
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}\nSummary: {method} on {model_slug}/{dataset_subset}\n{'='*70}")
    print(f"{'Sparsity':>10} {'Train':>10} {'Test':>10} {'Actual':>10}")
    for r in results:
        print(f"{r['sparsity']:>10.0%} {r['train_loss']:>10.4f} {r['test_loss']:>10.4f} {r['actual_sparsity']:>10.4f}")

    wandb.finish()
    del original_weights
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
