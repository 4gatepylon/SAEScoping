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
evaluates and decides when to stop. Weight zeroing runs at the start of
each prune phase; with ``n_iterations > 1`` the model is pruned again
after each recovery round (see ``n_iterations``).

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

Relevant older code:
    - Sparse trainer: https://github.com/4gatepylon/Deprecated-ScopeBench/blob/0c30cda68f0a0712c00864e8ab92a28e2994389e/l1_training/l1_sparse_trainer.py#L1
    - ??
"""

from __future__ import annotations

import json
import os
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
from pgd_trainer import PGDSFTTrainer
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
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    use_pgd: bool = True,
    n_iterations: int = 1,
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
        wandb_project: WandB project name. If None, WandB logging is disabled.
        wandb_run_name: WandB run name. Ignored when wandb_project is None.
        use_pgd: If True (default), use projected gradient descent during
            recovery SFT so that pruned (zeroed) weights remain at zero
            throughout training.  The keep-masks are derived directly from
            the saliency computation and passed to :class:`PGDSFTTrainer`.
            Set to False to use a standard SFTTrainer (pruned weights may
            drift back toward non-zero values).
        n_iterations: Number of prune phases to run. Each iteration:
            prune → evaluate → recover (unless threshold < 0 or metric already
            passes).  For ``saliency_type="taylor"``, masks are recomputed from
            **current** weights each call to :func:`prune.prune_model`, so
            later passes can zero additional weights.  For ``"gradient"``, the
            keep-mask is fixed by the saliency file; re-applying it is
            idempotent on weights, but recovery still runs each iteration when
            enabled.  Future work: optional on-the-fly saliency (e.g.
            gradients_map) and disk caching of maps per iteration.

    Returns:
        PruneAndRecoverResult with metrics and recovery info.
        ``n_weights_zeroed`` is from the **last** prune pass;
        ``recovery_steps`` is the **sum** of SFT steps across all recovery
        phases; ``metric_before_recovery`` / ``metric_after_recovery`` refer
        to the **final** iteration (or the early-exit iteration if training
        stops because the metric already passes after a prune).
    """
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
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

    total_recovery_steps = 0
    last_n_zeroed = 0
    last_metric_before = 0.0
    last_metric_after = 0.0
    last_stopped_early = False

    out_root = Path(output_dir)

    for it in range(n_iterations):
        print(f"\n=== Prune/recover iteration {it + 1}/{n_iterations} ===")

        prune_result = prune_model(
            model, saliency_path, sparsity,
            saliency_type=saliency_type, param_regex=param_regex,
            return_masks=use_pgd,
        )
        if use_pgd:
            n_zeroed, pgd_masks = prune_result  # type: ignore[misc]
        else:
            n_zeroed = prune_result  # type: ignore[assignment]
            pgd_masks = None
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Pruned {n_zeroed:,}/{total_params:,} weights ({n_zeroed/total_params:.2%})")
        last_n_zeroed = n_zeroed

        metric_before = evaluate_model(
            model, tokenizer, metric_type, eval_texts, eval_conversations,
            batch_size, max_seq_len, max_new_tokens,
        )
        print(f"Post-prune {metric_label}: {metric_before:.4f}")
        last_metric_before = metric_before

        skip_recovery = threshold < 0.0 or is_metric_passing(
            metric_before, metric_type, threshold,
        )
        if skip_recovery and threshold >= 0.0:
            print("Model already meets threshold. Skipping recovery.")
            last_metric_after = metric_before
            return PruneAndRecoverResult(
                sparsity=sparsity,
                n_weights_zeroed=last_n_zeroed,
                metric_type=metric_type,
                metric_before_recovery=last_metric_before,
                metric_after_recovery=last_metric_after,
                recovery_steps=total_recovery_steps,
                recovery_stopped_early=False,
            )

        if skip_recovery and threshold < 0.0:
            last_metric_after = metric_before
            if it < n_iterations - 1:
                continue
            return PruneAndRecoverResult(
                sparsity=sparsity,
                n_weights_zeroed=last_n_zeroed,
                metric_type=metric_type,
                metric_before_recovery=last_metric_before,
                metric_after_recovery=last_metric_after,
                recovery_steps=0,
                recovery_stopped_early=False,
            )

        if dataset_recovery is None:
            raise ValueError("dataset_recovery is required when threshold >= 0")

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

        iter_output_dir = str(out_root / f"iter_{it:03d}")
        if wandb_project is not None:
            os.environ["WANDB_PROJECT"] = wandb_project
        iter_run_name = (
            f"{wandb_run_name}_iter{it + 1}" if wandb_run_name is not None else None
        )
        use_cuda = torch.cuda.is_available() and model.device.type == "cuda"
        training_args = SFTConfig(
            output_dir=iter_output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            bf16=use_cuda,
            save_strategy="no",
            report_to="wandb" if wandb_project is not None else "none",
            run_name=iter_run_name,
            max_length=max_seq_len,
            dataset_text_field="text",
            logging_steps=10,
            optim="adamw_bnb_8bit" if use_cuda else "adamw_torch",
            no_cuda=not use_cuda,
        )

        if use_pgd and pgd_masks:
            trainer: SFTTrainer = PGDSFTTrainer(
                masks=pgd_masks,
                model=model,
                processing_class=tokenizer,
                train_dataset=sft_dataset,
                args=training_args,
                callbacks=[callback],
            )
            print(
                f"Using PGDSFTTrainer ({len(pgd_masks)} masked parameters, "
                f"sparsity enforced throughout recovery)"
            )
        else:
            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=sft_dataset,
                args=training_args,
                callbacks=[callback],
            )

        print(
            f"Starting recovery SFT iter {it + 1} "
            f"(max_steps={max_steps}, eval_every={eval_every})..."
        )
        trainer.train()
        total_recovery_steps += trainer.state.global_step

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        metric_after = evaluate_model(
            model, tokenizer, metric_type, eval_texts, eval_conversations,
            batch_size, max_seq_len, max_new_tokens,
        )
        print(f"Post-recovery {metric_label}: {metric_after:.4f} (was {metric_before:.4f})")
        last_metric_after = metric_after
        last_stopped_early = (
            callback.last_metric is not None
            and is_metric_passing(callback.last_metric, metric_type, threshold)
        )

        if it == n_iterations - 1:
            return PruneAndRecoverResult(
                sparsity=sparsity,
                n_weights_zeroed=last_n_zeroed,
                metric_type=metric_type,
                metric_before_recovery=last_metric_before,
                metric_after_recovery=last_metric_after,
                recovery_steps=total_recovery_steps,
                recovery_stopped_early=last_stopped_early,
            )

    raise AssertionError("bug: prune/recover loop exited without returning")


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
@click.option(
    "--n-iterations",
    type=int,
    default=1,
    show_default=True,
    help="Number of prune→(optional recover) rounds. Taylor saliency recomputes "
         "masks from current weights each round; gradient uses the same file each time.",
)
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
@click.option(
    "--wandb-project",
    type=str,
    default=None,
    help="WandB project name. Omit to disable WandB logging.",
)
@click.option(
    "--wandb-run-name",
    type=str,
    default=None,
    help="WandB run name. Only used when --wandb-project is set.",
)
@click.option(
    "--save-final-model/--no-save-final-model",
    default=False,
    help="Save the final pruned (and possibly recovered) model weights to "
         "<output-dir>/final_model/ after the run completes.",
)
@click.option(
    "--pgd/--no-pgd",
    default=True,
    show_default=True,
    help="Use projected gradient descent during recovery SFT so that pruned "
         "weights remain exactly zero throughout training (default: on). "
         "Pass --no-pgd to use a standard SFTTrainer instead.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-run even if --output-json already exists. Without this flag "
         "the script exits early if the output file is present.",
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
    n_iterations: int,
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
    wandb_project: Optional[str],
    wandb_run_name: Optional[str],
    save_final_model: bool,
    pgd: bool,
    force: bool,
) -> None:
    """Prune a model and optionally recover quality via SFT."""
    # Output-caching guard: skip if result already exists and --force not given.
    if output_json is not None and output_json.exists() and not force:
        print(f"Skipping: output already exists at {output_json} (pass --force to rerun)")
        return

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
        n_iterations=n_iterations,
        eval_every=eval_every,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        output_dir=output_dir,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        use_pgd=pgd,
    )

    result_json = result.model_dump_json(indent=2)
    print(f"\nResult:\n{result_json}")
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(result_json)
        print(f"Wrote result to {output_json}")

    if save_final_model:
        final_model_dir = Path(output_dir) / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        print(f"Saved final model weights to {final_model_dir}")


if __name__ == "__main__":
    main()
