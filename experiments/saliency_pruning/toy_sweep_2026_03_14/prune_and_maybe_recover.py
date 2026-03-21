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
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

from dataset_utils import (
    format_as_0turn,
    format_as_sft_dataset,
    format_as_sft_text,
    load_qa_dataset,
)
from model_generator import HFGenerator
from prune import prune_model
from utils import compute_validation_loss, generate_and_grade


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


# ---------------------------------------------------------------------------
# Early-stopping callback
# ---------------------------------------------------------------------------


class RecoveryEarlyStoppingCallback(TrainerCallback):
    """
    Periodically evaluate the model during recovery SFT and signal
    early stopping when the metric crosses a threshold.

    This callback does NOT modify model weights. It only reads the
    model state, evaluates, and sets trainer_control.should_training_stop.
    """

    def __init__(
        self,
        eval_every: int,
        threshold: float,
        metric_type: str,
        tokenizer: PreTrainedTokenizerBase,
        eval_texts: list[str],
        eval_conversations: list[list[dict]],
        batch_size: int = 4,
        max_seq_len: int = 1024,
        max_new_tokens: int = 256,
    ):
        self.eval_every = eval_every
        self.threshold = threshold
        self.metric_type = metric_type
        self.tokenizer = tokenizer
        self.eval_texts = eval_texts
        self.eval_conversations = eval_conversations
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.last_metric: Optional[float] = None
        self._generator: Optional[HFGenerator] = None

    def _evaluate_metric(self, model: PreTrainedModel) -> float:
        if self.metric_type == "loss":
            loss = compute_validation_loss(
                model, self.tokenizer, self.eval_texts,
                batch_size=self.batch_size, max_seq_len=self.max_seq_len,
            )
            return loss
        else:
            if self._generator is None:
                self._generator = HFGenerator(model, self.tokenizer)
            graded = generate_and_grade(
                self._generator, self.tokenizer,
                self.eval_conversations,
                batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
            )
            return graded.overall_mean_score

    def _should_stop(self, metric: float) -> bool:
        if self.metric_type == "loss":
            return metric <= self.threshold
        else:
            return metric >= self.threshold

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel = None,
        **kwargs,
    ) -> TrainerControl:
        if state.global_step % self.eval_every != 0:
            return control
        if model is None:
            return control

        metric = self._evaluate_metric(model)
        self.last_metric = metric
        metric_label = "loss" if self.metric_type == "loss" else "judge_score"
        print(
            f"  [Recovery step {state.global_step}] "
            f"{metric_label}={metric:.4f} (threshold={self.threshold})"
        )
        if self._should_stop(metric):
            print(f"  Recovery threshold met at step {state.global_step}. Stopping.")
            control.should_training_stop = True
        return control


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
    dataset_recovery: Optional[Dataset] = None,
    max_steps: int = 500,
    eval_every: int = 50,
    batch_size: int = 4,
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
    # Step 1: Prune
    n_zeroed = prune_model(
        model, saliency_path, sparsity,
        saliency_type=saliency_type, param_regex=param_regex,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Pruned {n_zeroed:,}/{total_params:,} weights ({n_zeroed/total_params:.2%})")

    # Step 2: Evaluate before recovery
    eval_texts = format_as_sft_text(dataset_evaluation, tokenizer)
    eval_conversations = format_as_0turn(dataset_evaluation)

    metric_before = _evaluate(
        model, tokenizer, metric_type, eval_texts, eval_conversations,
        batch_size, max_seq_len, max_new_tokens,
    )
    metric_label = "loss" if metric_type == "loss" else "judge_score"
    print(f"Post-prune {metric_label}: {metric_before:.4f}")

    # Step 3: Check if recovery is needed
    skip_recovery = threshold < 0.0
    if not skip_recovery:
        already_good = (
            (metric_type == "loss" and metric_before <= threshold) or
            (metric_type == "judge" and metric_before >= threshold)
        )
        if already_good:
            print("Model already meets threshold. Skipping recovery.")
            skip_recovery = True

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

    training_args = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="no",
        report_to="none",
        max_seq_length=max_seq_len,
        dataset_text_field="text",
        logging_steps=10,
    )

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

    metric_after = _evaluate(
        model, tokenizer, metric_type, eval_texts, eval_conversations,
        batch_size, max_seq_len, max_new_tokens,
    )
    print(f"Post-recovery {metric_label}: {metric_after:.4f} (was {metric_before:.4f})")

    stopped_early = callback.last_metric is not None and (
        (metric_type == "loss" and callback.last_metric <= threshold) or
        (metric_type == "judge" and callback.last_metric >= threshold)
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


def _evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    metric_type: str,
    eval_texts: list[str],
    eval_conversations: list[list[dict]],
    batch_size: int,
    max_seq_len: int,
    max_new_tokens: int,
) -> float:
    """Evaluate the model using the configured metric type."""
    if metric_type == "loss":
        return compute_validation_loss(
            model, tokenizer, eval_texts,
            batch_size=batch_size, max_seq_len=max_seq_len,
        )
    elif metric_type == "judge":
        generator = HFGenerator(model, tokenizer)
        graded = generate_and_grade(
            generator, tokenizer, eval_conversations,
            batch_size=batch_size, max_new_tokens=max_new_tokens,
        )
        return graded.overall_mean_score
    else:
        raise ValueError(f"Unknown metric_type '{metric_type}'. Choose 'loss' or 'judge'.")


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
@click.option("--dataset-name", type=str, default="4gate/StemQAMixture", show_default=True)
@click.option("--dataset-subset", type=str, default="biology", show_default=True)
@click.option("--n-eval", type=int, default=128, show_default=True)
@click.option("--n-recovery", type=int, default=512, show_default=True)
@click.option("--max-steps", type=int, default=500, show_default=True)
@click.option("--eval-every", type=int, default=50, show_default=True)
@click.option("--batch-size", type=int, default=4, show_default=True)
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
    dataset_name: str,
    dataset_subset: str,
    n_eval: int,
    n_recovery: int,
    max_steps: int,
    eval_every: int,
    batch_size: int,
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
        dataset_recovery=dataset_rec,
        max_steps=max_steps,
        eval_every=eval_every,
        batch_size=batch_size,
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
