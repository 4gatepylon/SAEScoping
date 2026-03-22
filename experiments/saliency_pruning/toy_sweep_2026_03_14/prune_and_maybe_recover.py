"""
prune_and_maybe_recover.py

Prune a model using a saliency map, then optionally recover quality
via SFT (supervised fine-tuning) on a recovery dataset.

Pipeline:
    1. Load saliency map and prune the model in-place.
    2. Evaluate the pruned model using a configurable metric
       (validation loss or LLM judge score).
    3. If the metric already meets the threshold, skip recovery.
    4. Otherwise run recovery SFT with an early-stopping callback
       that periodically evaluates the model and stops training
       when the metric crosses the threshold.

The early-stopping callback does NOT modify model weights — it only
evaluates and decides when to stop. Weight zeroing happens once
before recovery begins.

CLI usage:
    python prune_and_maybe_recover.py \\
        --saliency-path biology/ema_grads.safetensors \\
        --sparsity 0.5 \\
        --metric-type loss \\
        --threshold 2.5 \\
        --max-steps 500

    python prune_and_maybe_recover.py \\
        --saliency-path biology/ema_grads.safetensors \\
        --sparsity 0.5 \\
        --metric-type judge \\
        --threshold 0.7 \\
        --max-steps 200 \\
        --eval-every 50
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
import pydantic
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl import SFTConfig, SFTTrainer

from dataset_utils import (
    format_as_0turn,
    format_as_sft_dataset,
    format_as_sft_text,
    load_qa_dataset,
)
from prune import prune_model
from utils import RecoveryCallback, evaluate_model, is_metric_passing, resolve_threshold


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


class PruneAndRecoverResult(pydantic.BaseModel):
    sparsity: float
    n_weights_zeroed: int
    metric_type: str
    metric_before_recovery: float
    metric_after_recovery: float
    recovery_steps: int
    recovery_stopped_early: bool


# RecoveryEarlyStoppingCallback is the unified RecoveryCallback (no give-up rules).
RecoveryEarlyStoppingCallback = RecoveryCallback


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def prune_and_maybe_recover(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    saliency_path: str | Path,
    sparsity: float,
    dataset_evaluation: Dataset,
    saliency_type: str = "gradient",
    param_regex: Optional[str] = None,
    metric_type: str = "loss",
    threshold: float = -1.0,
    threshold_mode: str = "absolute",
    dataset_recovery: Optional[Dataset] = None,
    max_steps: int = 500,
    eval_every: int = 50,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-5,
    max_seq_len: int = 1024,
    max_new_tokens: int = 256,
    output_dir: str = "./recovery_output",
) -> PruneAndRecoverResult:
    """
    Prune a model, evaluate, and optionally recover via SFT.

    Args:
        model: Model to prune and recover. Modified in-place.
        tokenizer: Tokenizer matching the model.
        saliency_path: Path to .safetensors saliency map.
        sparsity: Fraction of scored weights to zero (0.0-1.0).
        dataset_evaluation: Dataset for evaluation (must have question/answer).
        saliency_type: "gradient" or "taylor".
        param_regex: Optional regex filter for which params to prune.
        metric_type: "loss" or "judge". Controls evaluation metric.
        threshold: Quality threshold. For loss: stop recovery when loss <= threshold.
            For judge: stop when score >= threshold. If negative, skip recovery.
        threshold_mode: "absolute" (use threshold as-is) or "fraction" (multiply
            threshold by the pre-prune baseline metric, e.g. 1.10 means 10% above
            baseline). Ignored when threshold < 0.
        dataset_recovery: Dataset for recovery SFT. Required if threshold >= 0.
        max_steps: Maximum recovery SFT steps.
        eval_every: Evaluate metric every N steps during recovery.
        batch_size: Batch size for training and evaluation.
        learning_rate: Learning rate for recovery SFT.
        max_seq_len: Maximum sequence length.
        max_new_tokens: Max tokens for generation (judge metric only).
        output_dir: Directory for trainer outputs (checkpoints, etc.).

    Returns:
        PruneAndRecoverResult with metrics and recovery info.
    """
    eval_texts = format_as_sft_text(dataset_evaluation, tokenizer)
    eval_conversations = format_as_0turn(dataset_evaluation)
    metric_label = "loss" if metric_type == "loss" else "judge_score"
    # TODO(Adriano) we should output the data to some location to make sure
    # that there are no bugs here

    # Resolve relative threshold before pruning (needs unpruned model)
    if threshold_mode == "fraction" and threshold >= 0.0:
        # TODO(Adriano) we want to make sure the evaluate loss amount is the
        # same as the one that we use to do early-stopping (I'm not 100% sure this
        # is the case lol; that's due to past experience seeing orders of magnitude)
        baseline = evaluate_model(
            model, tokenizer, metric_type, eval_texts, eval_conversations,
            batch_size, max_seq_len, max_new_tokens,
        )
        threshold = resolve_threshold(threshold, "fraction", baseline)
        print(
            f"Fraction threshold: baseline {metric_label}={baseline:.4f}, "
            f"effective threshold={threshold:.4f}"
        )

    # Step 1: Prune
    n_zeroed = prune_model(
        model, saliency_path, sparsity,
        saliency_type=saliency_type, param_regex=param_regex,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Pruned {n_zeroed:,}/{total_params:,} weights ({n_zeroed/total_params:.2%})")

    # Step 2: Evaluate before recovery
    metric_before = evaluate_model(
        model, tokenizer, metric_type, eval_texts, eval_conversations,
        batch_size, max_seq_len, max_new_tokens,
    )
    print(f"Post-prune {metric_label}: {metric_before:.4f}")

    # Step 3: Check if recovery is needed
    skip_recovery = threshold < 0.0 or is_metric_passing(metric_before, metric_type, threshold)
    if skip_recovery and threshold >= 0.0:
        print("Model already meets threshold. Skipping recovery.")

    if skip_recovery:
        return PruneAndRecoverResult(
            sparsity=sparsity,
            n_weights_zeroed=n_zeroed,
            metric_type=metric_type,
            metric_before_recovery=metric_before,
            metric_after_recovery=metric_before,
            recovery_steps=0,
            recovery_stopped_early=False,
        )

    # Step 4: Recovery SFT
    if dataset_recovery is None:
        raise ValueError("dataset_recovery is required when threshold >= 0")

    # Enable gradients for training
    # TODO(Adriano) consider only doing half the weights or something like that
    for p in model.parameters():
        p.requires_grad = True

    sft_dataset = format_as_sft_dataset(dataset_recovery, tokenizer)

    callback = RecoveryEarlyStoppingCallback(
        eval_every=eval_every,
        threshold=threshold,
        metric_type=metric_type,
        tokenizer=tokenizer,
        eval_texts=eval_texts,
        eval_conversations=eval_conversations,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
    )

    # TODO(Adriano) we MIGHT want to be able to do PeFT
    # TODO(Adriano) seperately, we will want to be able to do PeFT or SFT using
    # projected gradient descent.
    training_args = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="no",
        report_to="none",
        max_length=max_seq_len,
        dataset_text_field="text",
        logging_steps=10,
        optim="adamw_bnb_8bit",
    )

    # TODO(Adriano) we want to be able to do multi-gpu ideally to do this faster
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=sft_dataset,
        args=training_args,
        callbacks=[callback],
    )

    print(f"Starting recovery SFT (max_steps={max_steps}, eval_every={eval_every})...")
    trainer.train()
    recovery_steps = trainer.state.global_step

    # Step 5: Final evaluation
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    metric_after = evaluate_model(
        model, tokenizer, metric_type, eval_texts, eval_conversations,
        batch_size, max_seq_len, max_new_tokens,
    )
    print(f"Post-recovery {metric_label}: {metric_after:.4f} (was {metric_before:.4f})")

    stopped_early = (
        callback.last_metric is not None
        and is_metric_passing(callback.last_metric, metric_type, threshold)
    )

    return PruneAndRecoverResult(
        sparsity=sparsity,
        n_weights_zeroed=n_zeroed,
        metric_type=metric_type,
        metric_before_recovery=metric_before,
        metric_after_recovery=metric_after,
        recovery_steps=recovery_steps,
        recovery_stopped_early=stopped_early,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--saliency-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to .safetensors saliency map.",
)
@click.option("--model-id", type=str, default="google/gemma-2-9b-it", show_default=True)
@click.option("--sparsity", type=float, required=True, help="Fraction to prune (0.0-1.0).")
@click.option(
    "--saliency-type",
    type=click.Choice(["gradient", "taylor"]),
    default="gradient",
    show_default=True,
)
@click.option("--param-regex", type=str, default=None)
@click.option(
    "--metric-type",
    type=click.Choice(["loss", "judge"]),
    default="loss",
    show_default=True,
    help="Metric for evaluation: loss (cheap) or judge (expensive).",
)
@click.option(
    "--threshold",
    type=float,
    default=-1.0,
    show_default=True,
    help="Recovery threshold. For loss: stop when <= threshold. "
         "For judge: stop when >= threshold. Negative = skip recovery.",
)
@click.option(
    "--threshold-mode",
    type=click.Choice(["absolute", "fraction"]),
    default="absolute",
    show_default=True,
    help="absolute: use --threshold as-is. "
         "fraction: multiply --threshold by the pre-prune baseline metric "
         "(e.g. --threshold 1.10 means 10% above baseline).",
)
@click.option("--dataset-name", type=str, default="4gate/StemQAMixture", show_default=True)
@click.option("--dataset-subset", type=str, default="biology", show_default=True)
@click.option("--n-eval", type=int, default=128, show_default=True)
@click.option("--n-recovery", type=int, default=512, show_default=True)
@click.option("--max-steps", type=int, default=500, show_default=True)
@click.option("--eval-every", type=int, default=50, show_default=True)
@click.option("--batch-size", type=int, default=4, show_default=True)
@click.option("--gradient-accumulation-steps", type=int, default=1, show_default=True)
@click.option("--learning-rate", type=float, default=2e-5, show_default=True)
@click.option("--max-seq-len", type=int, default=1024, show_default=True)
@click.option("--max-new-tokens", type=int, default=256, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--output-dir",
    type=str,
    default="./recovery_output",
    show_default=True,
)
@click.option(
    "--output-json",
    type=click.Path(path_type=Path),
    default=None,
    help="Write result JSON to this path.",
)
@click.option(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
)
def main(
    saliency_path: Path,
    model_id: str,
    sparsity: float,
    saliency_type: str,
    param_regex: Optional[str],
    metric_type: str,
    threshold: float,
    threshold_mode: str,
    dataset_name: str,
    dataset_subset: str,
    n_eval: int,
    n_recovery: int,
    max_steps: int,
    eval_every: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_seq_len: int,
    max_new_tokens: int,
    seed: int,
    output_dir: str,
    output_json: Optional[Path],
    device: str,
) -> None:
    """Prune a model and optionally recover quality via SFT."""
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    chat_template_path = Path(__file__).parent / "prompts" / "gemma2_chat_template_system_prompt.j2"
    if chat_template_path.exists():
        tokenizer.chat_template = chat_template_path.read_text()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    dataset_eval = load_qa_dataset(
        dataset_name, dataset_subset, split="validation", n=n_eval, seed=seed,
    )
    dataset_rec = None
    if threshold >= 0.0:
        dataset_rec = load_qa_dataset(
            dataset_name, dataset_subset, split="train", n=n_recovery, seed=seed,
        )

    result = prune_and_maybe_recover(
        model=model,
        tokenizer=tokenizer,
        saliency_path=saliency_path,
        sparsity=sparsity,
        dataset_evaluation=dataset_eval,
        saliency_type=saliency_type,
        param_regex=param_regex,
        metric_type=metric_type,
        threshold=threshold,
        threshold_mode=threshold_mode,
        dataset_recovery=dataset_rec,
        max_steps=max_steps,
        eval_every=eval_every,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        output_dir=output_dir,
    )

    result_json = result.model_dump_json(indent=2)
    print(f"\nResult:\n{result_json}")
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(result_json)
        print(f"Wrote result to {output_json}")


if __name__ == "__main__":
    main()
