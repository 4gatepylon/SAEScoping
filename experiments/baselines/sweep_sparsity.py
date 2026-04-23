"""
Unified sparsity sweep across all pruning methods and model sizes.

Logs train loss, test loss, and LLM judge utility at each sparsity level.
Designed to run on a single GPU.

All methods share the same interface:
  1. Compute method-specific saliency/shared data (cached to disk)
  2. For each sparsity level: derive masks from saliency, apply, evaluate

Methods:
  - wanda:      |W| * ||X||_2, per-row pruning (saliency cached)
  - random:     uniform random scores, global threshold (saliency cached)
  - taylor:     |grad * weight| (gradient map cached, auto-computed if missing)
  - gradient:   |grad| (gradient map cached, auto-computed if missing)
  - sparse_llm: iterative alternating optimization (shared data cached,
                 per-sparsity masks cached)

Caching:
  All intermediate artifacts are cached under --cache-dir (default: ./saliency_cache).
  Use --no-cache to force recomputation. Cache structure:
    {cache_dir}/{model_slug}/{subset}/wanda_saliency.safetensors
    {cache_dir}/{model_slug}/{subset}/random_saliency.safetensors
    {cache_dir}/{model_slug}/{subset}/ema_grads.safetensors
    {cache_dir}/{model_slug}/{subset}/taylor_saliency.safetensors
    {cache_dir}/{model_slug}/{subset}/gradient_saliency.safetensors
    {cache_dir}/{model_slug}/{subset}/sparse_llm_masks_{sparsity}.safetensors

Usage:
  python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it
  python sweep_sparsity.py --method taylor --model google/gemma-2-9b-it
  python sweep_sparsity.py --method sparse_llm --model google/gemma-2-2b-it
  python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it --no-cache
  python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it --cache-dir /network/drive/cache
  python sweep_sparsity.py --method wanda --model google/gemma-2-2b-it --no-judge
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Optional

import click
import torch
import wandb
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from sae_scoping.training.weight_pruning import save_original_weights, restore_original_weights
from sae_scoping.training.saliency.wanda import (
    compute_wanda_saliency, compute_wanda_masks, apply_masks_to_model,
)

DEFAULT_SPARSITY_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_CACHE_DIR = Path("./saliency_cache")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


# TODO(claude) priority:high: cache filename ignores n_calibration, max_seq_len, and
# dataset_name — changing any of these silently reuses a stale saliency artifact.
# Either hash these into the filename or store them alongside and verify on load.
def _cache_path(cache_dir: Path, model_id: str, subset: str, filename: str) -> Path:
    return cache_dir / model_id.replace("/", "--") / subset / filename


def _load_or_compute_safetensors(
    path: Path,
    compute_fn,
    no_cache: bool = False,
    label: str = "artifact",
) -> dict[str, torch.Tensor]:
    """Load from cache or compute and save."""
    if path.exists() and not no_cache:
        print(f"[cache] Loading cached {label} from {path}")
        return load_file(str(path))
    print(f"[cache] Computing {label}...")
    result = compute_fn()
    if not no_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(result, str(path))
        print(f"[cache] Saved {label} to {path}")
    return result


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
    """Load non-overlapping train/test/calibration splits."""
    # TODO(claude) priority:medium: load_dataset has no revision= pin — if the HF
    # dataset is re-uploaded, saliency cache becomes subtly stale (cache key
    # doesn't encode revision either; see H1 on _cache_path).
    ds = load_dataset(dataset_name, dataset_subset, split="train")
    ds = ds.shuffle(seed=42)
    total_needed = n_train + n_test + n_calibration
    # TODO(claude) priority:low: on failure this assert gives no actionable hint —
    # include subset name and suggest lowering n_train/n_test/n_calibration.
    assert len(ds) >= total_needed, f"Dataset has {len(ds)} < {total_needed} needed"

    def format_texts(start, end):
        return [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": str(ds[i]["question"])},
                 {"role": "assistant", "content": str(ds[i]["answer"])}],
                tokenize=False, add_generation_prompt=False,
            )
            for i in range(start, end)
        ]

    calib = format_texts(0, n_calibration)
    train = format_texts(n_calibration, n_calibration + n_train)
    test = format_texts(n_calibration + n_train, n_calibration + n_train + n_test)
    return train, test, calib


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------


@torch.no_grad()
# TODO(hadriano): expose batch_size as a CLI flag instead of hardcoding.
# TODO(claude) priority:high: this returns a per-batch mean, not a token-weighted
# mean. Batches with different non-pad token counts contribute equally, so the
# metric shifts when batch_size, max_seq_len, or pad distribution changes.
# Accumulate (sum of per-token CE) / (total non-pad tokens) instead.
def compute_loss(model, tokenizer, texts, max_seq_len=1024, batch_size=2):
    model.eval()
    try:
        device = model.device
    except AttributeError:
        device = next(p.device for p in model.parameters())
    old_pad = tokenizer.padding_side
    tokenizer.padding_side = "right"
    total, n = 0.0, 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tok = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)
        ids = tok["input_ids"].to(device)
        mask = tok["attention_mask"].to(device)
        labels = ids.clone()
        labels[mask == 0] = -100
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        total += out.loss.item()
        n += 1
    tokenizer.padding_side = old_pad
    # TODO(claude) priority:high: empty texts returns 0.0 silently (looks like a
    # valid loss); raise or return NaN/None so upstream can notice.
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Unified saliency computation + masking
# ---------------------------------------------------------------------------


def compute_saliency(
    method: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    calibration_texts: list[str],
    max_seq_len: int,
    cache_dir: Path,
    model_id: str,
    dataset_name: str,
    dataset_subset: str,
    no_cache: bool,
    sparse_llm_iterations: int = 4,
) -> Any:
    """Compute and cache the method-specific saliency data.

    Returns:
      - For wanda/random/taylor/gradient: dict[str, Tensor] saliency map
      - For sparse_llm: SparseLLMSharedData object
    """
    if method == "wanda":
        path = _cache_path(cache_dir, model_id, dataset_subset, "wanda_saliency.safetensors")
        return _load_or_compute_safetensors(
            path, lambda: compute_wanda_saliency(model, tokenizer, calibration_texts, max_seq_len=max_seq_len),
            no_cache=no_cache, label="Wanda saliency",
        )

    elif method == "random":
        from sae_scoping.training.saliency.random import make_random_map
        # TODO(claude) priority:high: seed is hardcoded to 42 and the cache
        # filename doesn't encode it; if a --seed flag is added later, multiple
        # seeds will collide on the same cache file. Encode seed into filename.
        path = _cache_path(cache_dir, model_id, dataset_subset, "random_saliency.safetensors")
        return _load_or_compute_safetensors(
            path, lambda: make_random_map(model, seed=42),
            no_cache=no_cache, label="random saliency",
        )

    elif method in ("taylor", "gradient"):
        # Step 1: ensure gradient map exists
        grad_path = _cache_path(cache_dir, model_id, dataset_subset, "ema_grads.safetensors")
        raw_grads = _load_or_compute_safetensors(
            grad_path, lambda: _compute_ema_grads(model, tokenizer, dataset_name, dataset_subset),
            no_cache=no_cache, label="EMA gradient map",
        )
        # Step 2: derive method-specific saliency
        if method == "taylor":
            from sae_scoping.training.saliency.taylor import make_taylor_map
            filename = "taylor_saliency.safetensors"
            saliency_path = _cache_path(cache_dir, model_id, dataset_subset, filename)
            return _load_or_compute_safetensors(
                saliency_path, lambda: make_taylor_map(raw_grads, model),
                no_cache=no_cache, label="Taylor saliency",
            )
        else:
            filename = "gradient_saliency.safetensors"
            saliency_path = _cache_path(cache_dir, model_id, dataset_subset, filename)
            return _load_or_compute_safetensors(
                saliency_path, lambda: {k: v.abs() for k, v in raw_grads.items()},
                no_cache=no_cache, label="gradient saliency",
            )

    elif method == "sparse_llm":
        from sae_scoping.training.saliency.sparse_llm import precompute_shared_data
        # Shared data is not easily serializable (contains large tensors + dataclass).
        # We don't cache it to disk — but it's computed once and reused across sparsities.
        # Per-sparsity masks ARE cached to disk (see masks_for_sparsity below).
        return precompute_shared_data(model, tokenizer, calibration_texts, max_seq_len=max_seq_len)

    else:
        raise ValueError(f"Unknown method: {method}")


def _compute_ema_grads(model, tokenizer, dataset_name, dataset_subset):
    """Compute EMA gradient map using GradCollectTrainer."""
    from sae_scoping.datasets.qa_datasets import load_qa_dataset, format_as_sft_dataset
    from sae_scoping.training.saliency.grad import GradCollectTrainer
    from trl import SFTConfig

    qa_dataset = load_qa_dataset(dataset_name, dataset_subset, split="train", n=4096, seed=42)
    sft_dataset = format_as_sft_dataset(qa_dataset, tokenizer)
    trainer = GradCollectTrainer(
        model=model, beta=0.95, abs_grad=False,
        processing_class=tokenizer, train_dataset=sft_dataset,
        args=SFTConfig(
            output_dir="./deleteme_grad_collect", num_train_epochs=1,
            per_device_train_batch_size=2, gradient_accumulation_steps=1,
            bf16=True, max_grad_norm=None, learning_rate=1e-4,
            save_strategy="no", report_to="none", max_length=1024,
            dataset_text_field="text",
        ),
    )
    trainer.train()
    return trainer.ema_grads()


def masks_for_sparsity(
    method: str,
    saliency_data: Any,
    model: PreTrainedModel,
    sparsity: float,
    cache_dir: Path,
    model_id: str,
    dataset_subset: str,
    no_cache: bool,
    sparse_llm_iterations: int = 4,
) -> dict[str, torch.Tensor]:
    """Compute keep masks for a specific sparsity from pre-computed saliency data.

    For wanda: per-row thresholding.
    For random/taylor/gradient: global thresholding.
    For sparse_llm: iterative optimization (cached per sparsity level).
    """
    if method == "wanda":
        return compute_wanda_masks(saliency_data, sparsity)

    elif method in ("random", "taylor", "gradient"):
        # Global threshold
        # TODO(claude) priority:high: concatenates all scores into a single CPU
        # tensor (~36 GB float32 at 9B-params, worse at 12B) and older torch has
        # a 16M-element cap on quantile(). weight_pruning.py::compute_keep_masks
        # already solves this via a 10M-element random sample — use the same
        # trick here or import that helper.
        all_scores = torch.cat([s.flatten().float() for s in saliency_data.values()])
        threshold = torch.quantile(all_scores, sparsity).item()
        return {name: (scores > threshold) for name, scores in saliency_data.items()}

    elif method == "sparse_llm":
        from sae_scoping.training.saliency.sparse_llm import compute_sparse_llm_masks
        # Cache per-sparsity masks
        sp_str = f"{sparsity:.4f}".replace(".", "p")
        mask_path = _cache_path(cache_dir, model_id, dataset_subset, f"sparse_llm_masks_{sp_str}.safetensors")
        return _load_or_compute_safetensors(
            mask_path,
            lambda: compute_sparse_llm_masks(
                saliency_data, model, sparsity=sparsity, n_iterations=sparse_llm_iterations,
            ),
            no_cache=no_cache, label=f"SparseLLM masks @ {sparsity:.0%}",
        )

    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--method", type=click.Choice(["wanda", "random", "taylor", "gradient", "sparse_llm"]), required=True)
@click.option("--model", "model_id", required=True, help="HuggingFace model ID")
@click.option("--dataset-name", default="4gate/StemQAMixture")
@click.option("--dataset-subset", default="biology")
@click.option("--sparsity-levels", default=None, help="Comma-separated (default: 0,0.1,...,0.9)")
@click.option("--n-calibration", default=128, help="Calibration samples")
@click.option("--n-train", default=200, help="Train samples for loss")
@click.option("--n-test", default=200, help="Test samples for loss")
@click.option("--n-judge-samples", default=50, help="Samples for LLM judge")
@click.option("--max-seq-len", default=1024)
@click.option("--sparse-llm-iterations", default=4)
@click.option("--cache-dir", type=click.Path(path_type=Path), default=DEFAULT_CACHE_DIR,
              help="Cache directory for saliency maps and masks")
@click.option("--no-cache", is_flag=True, help="Disable caching (always recompute)")
@click.option("--no-judge", is_flag=True, help="Skip LLM judge evaluation")
@click.option("--wandb-project", default="sae-scoping-baselines")
@click.option("--device", default=None, help="CUDA device (auto-detected if not set)")
def main(
    method, model_id, dataset_name, dataset_subset, sparsity_levels,
    n_calibration, n_train, n_test, n_judge_samples, max_seq_len,
    sparse_llm_iterations, cache_dir, no_cache, no_judge, wandb_project, device,
):
    # TODO(claude) priority:medium: no torch.manual_seed / np.random.seed called
    # anywhere in this script. Forward-only Wanda is deterministic, but if
    # do_sample is ever enabled in the judge generation, or any RNG-using
    # saliency is added, runs will drift. Set seeds explicitly near the top.
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
    train_texts, test_texts, calib_texts = load_split_texts(
        tokenizer, dataset_name, dataset_subset, n_train, n_test, n_calibration,
    )
    judge_questions, judge_answers = None, None
    if not no_judge:
        # TODO(claude) priority:high: judge pool is drawn from the same subset's
        # train split with a different seed (777), but is NOT excluded from the
        # seed-42 calibration/train/test splits above. At n_calibration=10k with
        # subset size ~10-11k the judge set is almost entirely inside calibration
        # — "judge" is not held-out utility. Subtract the calib/train/test row
        # indices before sampling, or use a separate held-out split.
        ds_judge = load_dataset(dataset_name, dataset_subset, split="train").shuffle(seed=777)
        n_j = min(n_judge_samples, len(ds_judge))
        judge_questions = [str(ds_judge[i]["question"]) for i in range(n_j)]
        judge_answers = [str(ds_judge[i]["answer"]) for i in range(n_j)]

    # Step 1: Compute saliency (cached, done once)
    print(f"\n--- Computing saliency for {method} ---")
    saliency_data = compute_saliency(
        method, model, tokenizer, calib_texts, max_seq_len,
        cache_dir, model_id, dataset_name, dataset_subset, no_cache,
        sparse_llm_iterations=sparse_llm_iterations,
    )

    # Save original weights for sweep
    original_weights = save_original_weights(model)

    # Wandb
    # TODO(claude) priority:medium: wandb.init is unconditional — if the machine
    # isn't logged in, a detached/nohup run will block on interactive auth
    # silently. Either respect WANDB_MODE=offline by default, or fail-fast check
    # wandb.api.api_key before reaching here.
    # TODO(claude) priority:low: config omits n_train, n_test, n_judge_samples,
    # max_seq_len, dataset_name, sparse_llm_iterations, no_judge, and compute_loss
    # batch_size. Varying any of these across runs makes W&B-side comparison
    # ambiguous.
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
            masks = masks_for_sparsity(
                method, saliency_data, model, sp,
                cache_dir, model_id, dataset_subset, no_cache,
                sparse_llm_iterations=sparse_llm_iterations,
            )
            n_zeroed = apply_masks_to_model(model, masks)
            # TODO(claude) priority:medium: denominator counts every param whose
            # name is in masks. Works today (Wanda only masks 2D Linear weights),
            # but if anyone adds embed_tokens or norm params to masks later the
            # denominator shifts silently. Filter explicitly to 2D weights.
            n_total = sum(p.numel() for n, p in model.named_parameters() if n in masks)
            actual_sparsity = n_zeroed / n_total if n_total > 0 else 0.0
            prune_time = time.time() - t0

        train_loss = compute_loss(model, tokenizer, train_texts, max_seq_len)
        test_loss = compute_loss(model, tokenizer, test_texts, max_seq_len)

        result = dict(sparsity=sp, actual_sparsity=actual_sparsity, n_zeroed=n_zeroed,
                      train_loss=train_loss, test_loss=test_loss, prune_time_s=prune_time)
        # TODO(claude) priority:low: no subset/method tag in the print — when two
        # subprocess logs get merged/tailed it's hard to attribute a line.
        print(f"  train={train_loss:.4f}  test={test_loss:.4f}  actual_sp={actual_sparsity:.4f}")

        if not no_judge and judge_questions:
            # TODO(claude) priority:medium: evaluator is re-instantiated every
            # sparsity — templates reload and n_max_openai_requests=1800 guard
            # resets, so cumulative cost across a 10-level sweep can be 18k calls.
            # Hoist the evaluator outside the loop.
            # TODO(claude) priority:medium: any judge exception becomes a single
            # printed line; the row is logged without judge keys and the sweep
            # completes "successfully" with half the metrics silently missing.
            # Record a judge_failed counter and raise if it exceeds a threshold.
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
        # TODO(claude) priority:medium: empty_cache only runs after both
        # compute_loss calls and the judge; fragmentation could OOM mid-sweep on
        # a contended GPU. Consider calling between train-loss and test-loss too.
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}\nSummary: {method} on {model_slug}/{dataset_subset}\n{'='*70}")
    print(f"{'Sparsity':>10} {'Train':>10} {'Test':>10} {'Actual':>10}")
    for r in results:
        print(f"{r['sparsity']:>10.0%} {r['train_loss']:>10.4f} {r['test_loss']:>10.4f} {r['actual_sparsity']:>10.4f}")

    wandb.finish()
    del original_weights
    # TODO(claude) priority:low: consider running gc.collect / empty_cache between
    # sparsity levels too, not just at end — helps marginal memory recovery.
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
