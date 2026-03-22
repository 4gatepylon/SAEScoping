"""
prune_and_maybe_recover_sweep.py

Binary search over sparsity levels to find the maximum sparsity where the
model still meets a quality threshold after pruning + optional recovery SFT.

For each binary search step:
    1. Reload model from model_name_or_path (pruning destroys weights in-place).
    2. Prune via prune_model.
    3. Evaluate metric before recovery.
    4. If threshold >= 0 and model doesn't already pass: run recovery SFT with
       SweepRecoveryCallback (handles early-stop AND give-up rules).
    5. Mark step as success/fail. Advance binary search bounds.
    6. Cache the top num_cache successful checkpoints by (sparsity, metric).

Metric types
------------
loss  : validation cross-entropy (lower=better). Passes when loss  <= threshold.
judge : LLM judge score 0-1    (higher=better). Passes when score >= threshold.

Recommended workflow
--------------------
1. Run with --metric-type loss for a fast, cheap range-finding pass.
2. Narrow k-min / k-max, then re-run with --metric-type judge for accuracy.

CLI usage:
    python prune_and_maybe_recover_sweep.py \\
        --saliency-path biology/ema_grads.safetensors \\
        --metric-type loss --threshold 2.5 \\
        --k-min 0.1 --k-max 0.9 \\
        --max-steps-sweep 8 --max-steps-recovery 200 \\
        --output-dir ./sweep_output
"""

from __future__ import annotations

import datetime
import gc
import json
from pathlib import Path
from typing import Optional

import click
import pydantic
import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl import SFTConfig, SFTTrainer

from dataset_utils import format_as_0turn, format_as_sft_dataset, format_as_sft_text, load_qa_dataset
from prune import prune_model
from utils import GiveUpThreshold, RecoveryCallback, evaluate_model, is_metric_better, is_metric_passing


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_ID = "google/gemma-2-9b-it"
_DEFAULT_DATASET = "4gate/StemQAMixture"
_DEFAULT_SUBSET = "biology"
_CHAT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "gemma2_chat_template_system_prompt.j2"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


# GiveUpThreshold is defined in utils.py (re-exported here for import compatibility).
# SweepRecoveryCallback is an alias for the unified RecoveryCallback in utils.py.
SweepRecoveryCallback = RecoveryCallback


class SweepStepResult(pydantic.BaseModel):
    step_index: int
    sparsity: float
    n_weights_zeroed: int
    metric_type: str
    metric_before_recovery: float
    metric_after_recovery: float
    recovery_steps: int
    recovery_stopped_early: bool
    gave_up: bool
    is_success: bool


class SweepResult(pydantic.BaseModel):
    best_sparsity: Optional[float]
    best_metric: Optional[float]
    metric_type: str
    threshold: float
    k_min: float
    k_max: float
    steps: list[SweepStepResult]
    cached_checkpoint_dirs: list[str]


# ---------------------------------------------------------------------------
# Checkpoint cache
# ---------------------------------------------------------------------------


def _checkpoint_sort_key(
    entry: tuple[float, float, str],
    metric_type: str,
) -> tuple[float, float]:
    """Sort key: higher sparsity first; better metric breaks ties."""
    sparsity, metric, _ = entry
    metric_for_sort = -metric if metric_type == "loss" else metric
    return (sparsity, metric_for_sort)


class CheckpointCache:
    """
    Fixed-capacity cache of the best successful sweep checkpoints.

    Priority: higher sparsity first; better metric (lower loss / higher judge) breaks ties.
    """

    def __init__(self, capacity: int, metric_type: str) -> None:
        self.capacity = capacity
        self.metric_type = metric_type
        self._entries: list[tuple[float, float, str]] = []

    def try_add(self, sparsity: float, metric: float, output_dir: str) -> bool:
        """Add entry if it improves the cache. Returns True if accepted."""
        new_entry = (sparsity, metric, output_dir)
        self._entries.append(new_entry)
        self._entries.sort(
            key=lambda e: _checkpoint_sort_key(e, self.metric_type),
            reverse=True,
        )
        if len(self._entries) > self.capacity:
            self._entries = self._entries[: self.capacity]
        return new_entry in self._entries

    def checkpoint_dirs(self) -> list[str]:
        return [e[2] for e in self._entries]


# ---------------------------------------------------------------------------
# Single sweep step
# ---------------------------------------------------------------------------


def run_sweep_step(
    step_index: int,
    model_name_or_path: str,
    tokenizer: PreTrainedTokenizerBase,
    saliency_path: str | Path,
    sparsity: float,
    dataset_evaluation: Dataset,
    saliency_type: str,
    param_regex: Optional[str],
    metric_type: str,
    threshold: float,
    dataset_recovery: Optional[Dataset],
    max_steps_recovery: int,
    eval_every: int,
    batch_size: int,
    learning_rate: float,
    max_seq_len: int,
    max_new_tokens: int,
    give_up_thresholds: list[GiveUpThreshold],
    step_output_dir: str,
    device: str,
) -> SweepStepResult:
    """
    Run one binary search step: load a fresh model, prune, then optionally recover.

    The model is loaded fresh from model_name_or_path each call because pruning
    modifies weights in-place; the same model object cannot be reused across steps.
    The model is explicitly deleted and CUDA cache cleared before returning.
    """
    print(f"\n{'='*70}")
    print(f"[Sweep step {step_index}] sparsity={sparsity:.4f}")
    print(f"{'='*70}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    n_zeroed = prune_model(
        model, saliency_path, sparsity,
        saliency_type=saliency_type, param_regex=param_regex,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Pruned {n_zeroed:,}/{total_params:,} weights ({n_zeroed/total_params:.2%})")

    eval_texts = format_as_sft_text(dataset_evaluation, tokenizer)
    eval_conversations = format_as_0turn(dataset_evaluation)

    metric_before = evaluate_model(
        model, tokenizer, metric_type, eval_texts, eval_conversations,
        batch_size, max_seq_len, max_new_tokens,
    )
    metric_label = "loss" if metric_type == "loss" else "judge"
    print(f"Post-prune {metric_label}: {metric_before:.4f}")

    skip_recovery = threshold < 0.0 or is_metric_passing(metric_before, metric_type, threshold)

    if skip_recovery:
        is_success = threshold < 0.0 or is_metric_passing(metric_before, metric_type, threshold)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return SweepStepResult(
            step_index=step_index,
            sparsity=sparsity,
            n_weights_zeroed=n_zeroed,
            metric_type=metric_type,
            metric_before_recovery=metric_before,
            metric_after_recovery=metric_before,
            recovery_steps=0,
            recovery_stopped_early=False,
            gave_up=False,
            is_success=is_success,
        )

    if dataset_recovery is None:
        raise ValueError("dataset_recovery is required when threshold >= 0")

    for p in model.parameters():
        p.requires_grad = True

    sft_dataset = format_as_sft_dataset(dataset_recovery, tokenizer)

    callback = SweepRecoveryCallback(
        eval_every=eval_every,
        threshold=threshold,
        metric_type=metric_type,
        tokenizer=tokenizer,
        eval_texts=eval_texts,
        eval_conversations=eval_conversations,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        give_up_thresholds=give_up_thresholds,
    )

    training_args = SFTConfig(
        output_dir=step_output_dir,
        max_steps=max_steps_recovery,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        bf16=True,
        save_strategy="no",
        report_to="none",
        max_length=max_seq_len,
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

    print(f"Starting recovery SFT (max_steps={max_steps_recovery}, eval_every={eval_every})...")
    trainer.train()
    recovery_steps = trainer.state.global_step

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
    is_success = is_metric_passing(metric_after, metric_type, threshold) and not callback.gave_up

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return SweepStepResult(
        step_index=step_index,
        sparsity=sparsity,
        n_weights_zeroed=n_zeroed,
        metric_type=metric_type,
        metric_before_recovery=metric_before,
        metric_after_recovery=metric_after,
        recovery_steps=recovery_steps,
        recovery_stopped_early=stopped_early,
        gave_up=callback.gave_up,
        is_success=is_success,
    )


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def prune_and_maybe_recover_sweep(
    model_name_or_path: str,
    saliency_path: str | Path,
    dataset_evaluation: Dataset,
    k_min: float = 0.0,
    k_max: float = 1.0,
    saliency_type: str = "gradient",
    param_regex: Optional[str] = None,
    metric_type: str = "loss",
    threshold: float = -1.0,
    dataset_recovery: Optional[Dataset] = None,
    max_steps_recovery: int = 500,
    eval_every: int = 50,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_seq_len: int = 1024,
    max_new_tokens: int = 256,
    max_steps_sweep: int = 8,
    give_up_thresholds: Optional[list[GiveUpThreshold]] = None,
    num_cache: int = 3,
    output_dir: str = "./sweep_output",
    seed: int = 42,
    device: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    tokenizer_id: Optional[str] = None,
) -> SweepResult:
    """
    Binary-search sweep over sparsity levels to maximise pruning while meeting
    a quality threshold.

    Args:
        model_name_or_path: HF model ID or local path. Reloaded fresh each step.
        saliency_path: Path to .safetensors saliency map.
        dataset_evaluation: Evaluation dataset (must have question/answer columns).
        k_min: Minimum sparsity (lower bound for binary search).
        k_max: Maximum sparsity (upper bound for binary search).
        saliency_type: "gradient" or "taylor".
        param_regex: Optional regex to filter which params to prune.
        metric_type: "loss" or "judge".
        threshold: Quality threshold. Negative = no bar (all steps succeed).
        dataset_recovery: Recovery SFT dataset. Required when threshold >= 0.
        max_steps_recovery: Max SFT steps per sweep step.
        eval_every: Eval metric every N recovery steps.
        batch_size: Batch size for training and evaluation.
        learning_rate: Recovery SFT learning rate.
        max_seq_len: Max sequence length.
        max_new_tokens: Max tokens for generation (judge metric only).
        max_steps_sweep: Number of binary search halvings.
        give_up_thresholds: Give-up rules; see GiveUpThreshold.
        num_cache: Keep top-N successful checkpoints.
        output_dir: Root dir; per-step outputs go to <output_dir>/step_NNN/.
        seed: Random seed for dataset loading.
        device: "cuda" / "cpu". Defaults to CUDA if available.
        wandb_project: WandB project name. No logging if None.
        wandb_run_name: WandB run name; auto-generated if None.
        tokenizer_id: Tokenizer ID if different from model_name_or_path.

    Returns:
        SweepResult with best sparsity found and full per-step history.
    """
    effective_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    effective_give_up = give_up_thresholds or []
    tok_id = tokenizer_id or model_name_or_path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    if _CHAT_TEMPLATE_PATH.exists():
        tokenizer.chat_template = _CHAT_TEMPLATE_PATH.read_text()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if wandb_project:
        run_name = wandb_run_name or (
            f"{datetime.date.today().isoformat()}_sweep_{metric_type}"
        )
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "model": model_name_or_path,
                "saliency_path": str(saliency_path),
                "k_min": k_min,
                "k_max": k_max,
                "metric_type": metric_type,
                "threshold": threshold,
                "max_steps_sweep": max_steps_sweep,
                "max_steps_recovery": max_steps_recovery,
            },
        )

    cache = CheckpointCache(capacity=num_cache, metric_type=metric_type)
    steps: list[SweepStepResult] = []
    lo, hi = k_min, k_max
    best_sparsity: Optional[float] = None
    best_metric: Optional[float] = None

    for step_i in range(max_steps_sweep):
        mid = (lo + hi) / 2.0
        step_output_dir = str(Path(output_dir) / f"step_{step_i:03d}")

        step_result = run_sweep_step(
            step_index=step_i,
            model_name_or_path=model_name_or_path,
            tokenizer=tokenizer,
            saliency_path=saliency_path,
            sparsity=mid,
            dataset_evaluation=dataset_evaluation,
            saliency_type=saliency_type,
            param_regex=param_regex,
            metric_type=metric_type,
            threshold=threshold,
            dataset_recovery=dataset_recovery,
            max_steps_recovery=max_steps_recovery,
            eval_every=eval_every,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_seq_len=max_seq_len,
            max_new_tokens=max_new_tokens,
            give_up_thresholds=effective_give_up,
            step_output_dir=step_output_dir,
            device=effective_device,
        )
        steps.append(step_result)

        if step_result.is_success:
            lo = mid
            if best_sparsity is None or mid > best_sparsity:
                best_sparsity = mid
                best_metric = step_result.metric_after_recovery
            elif mid == best_sparsity and is_metric_better(
                step_result.metric_after_recovery, best_metric, metric_type
            ):
                best_metric = step_result.metric_after_recovery
            cache.try_add(mid, step_result.metric_after_recovery, step_output_dir)
        else:
            hi = mid

        step_json_path = Path(output_dir) / f"step_{step_i:03d}_result.json"
        step_json_path.write_text(step_result.model_dump_json(indent=2))

        if wandb_project:
            wandb.log({
                "step": step_i,
                "sparsity": mid,
                "metric_before": step_result.metric_before_recovery,
                "metric_after": step_result.metric_after_recovery,
                "recovery_steps": step_result.recovery_steps,
                "is_success": int(step_result.is_success),
                "gave_up": int(step_result.gave_up),
                "lo": lo,
                "hi": hi,
            })

        metric_label = "loss" if metric_type == "loss" else "judge"
        status = "✅ success" if step_result.is_success else "❌ fail"
        print(
            f"\n[Sweep step {step_i}] {status}  sparsity={mid:.4f}  "
            f"{metric_label}_after={step_result.metric_after_recovery:.4f}  "
            f"range=[{lo:.4f}, {hi:.4f}]"
        )

        if hi - lo < 1e-6:
            print("Binary search converged (hi - lo < 1e-6). Stopping.")
            break

    result = SweepResult(
        best_sparsity=best_sparsity,
        best_metric=best_metric,
        metric_type=metric_type,
        threshold=threshold,
        k_min=k_min,
        k_max=k_max,
        steps=steps,
        cached_checkpoint_dirs=cache.checkpoint_dirs(),
    )

    result_path = Path(output_dir) / "sweep_result.json"
    result_path.write_text(result.model_dump_json(indent=2))
    print(f"\nSweep complete. Best sparsity: {best_sparsity}. Result -> {result_path}")

    if wandb_project:
        wandb.finish()

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--model-id", type=str, default=_DEFAULT_MODEL_ID, show_default=True)
@click.option(
    "--saliency-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to .safetensors saliency map.",
)
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
)
@click.option(
    "--threshold",
    type=float,
    default=-1.0,
    show_default=True,
    help="Quality threshold. Negative = no quality bar (every step succeeds).",
)
@click.option("--k-min", type=float, default=0.0, show_default=True)
@click.option("--k-max", type=float, default=1.0, show_default=True)
@click.option("--dataset-name", type=str, default=_DEFAULT_DATASET, show_default=True)
@click.option("--dataset-subset", type=str, default=_DEFAULT_SUBSET, show_default=True)
@click.option("--n-eval", type=int, default=128, show_default=True)
@click.option("--n-recovery", type=int, default=512, show_default=True)
@click.option("--max-steps-sweep", type=int, default=8, show_default=True)
@click.option("--max-steps-recovery", type=int, default=200, show_default=True)
@click.option("--eval-every", type=int, default=50, show_default=True)
@click.option("--batch-size", type=int, default=4, show_default=True)
@click.option("--learning-rate", type=float, default=2e-5, show_default=True)
@click.option("--max-seq-len", type=int, default=1024, show_default=True)
@click.option("--max-new-tokens", type=int, default=256, show_default=True)
@click.option(
    "--give-up-rules",
    type=str,
    default=None,
    help="JSON list of give-up rules, e.g. '[{\"steps\":50,\"threshold\":2.0}]'.",
)
@click.option("--num-cache", type=int, default=3, show_default=True)
@click.option("--output-dir", type=str, default="./sweep_output", show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--wandb-project", type=str, default=None)
@click.option("--wandb-run-name", type=str, default=None)
@click.option("--device", type=str, default=None)
def main(
    model_id: str,
    saliency_path: Path,
    saliency_type: str,
    param_regex: Optional[str],
    metric_type: str,
    threshold: float,
    k_min: float,
    k_max: float,
    dataset_name: str,
    dataset_subset: str,
    n_eval: int,
    n_recovery: int,
    max_steps_sweep: int,
    max_steps_recovery: int,
    eval_every: int,
    batch_size: int,
    learning_rate: float,
    max_seq_len: int,
    max_new_tokens: int,
    give_up_rules: Optional[str],
    num_cache: int,
    output_dir: str,
    seed: int,
    wandb_project: Optional[str],
    wandb_run_name: Optional[str],
    device: Optional[str],
) -> None:
    """Binary-search sweep over sparsity levels for maximum pruning with quality."""
    give_up_thresholds: list[GiveUpThreshold] = []
    if give_up_rules:
        give_up_thresholds = [GiveUpThreshold(**r) for r in json.loads(give_up_rules)]

    dataset_eval = load_qa_dataset(
        dataset_name, dataset_subset, split="validation", n=n_eval, seed=seed,
    )
    dataset_rec: Optional[Dataset] = None
    if threshold >= 0.0:
        dataset_rec = load_qa_dataset(
            dataset_name, dataset_subset, split="train", n=n_recovery, seed=seed,
        )

    result = prune_and_maybe_recover_sweep(
        model_name_or_path=model_id,
        saliency_path=saliency_path,
        dataset_evaluation=dataset_eval,
        k_min=k_min,
        k_max=k_max,
        saliency_type=saliency_type,
        param_regex=param_regex,
        metric_type=metric_type,
        threshold=threshold,
        dataset_recovery=dataset_rec,
        max_steps_recovery=max_steps_recovery,
        eval_every=eval_every,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        max_steps_sweep=max_steps_sweep,
        give_up_thresholds=give_up_thresholds,
        num_cache=num_cache,
        output_dir=output_dir,
        seed=seed,
        device=device,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )

    print(f"\nBest sparsity: {result.best_sparsity}")
    print(f"\nResult:\n{result.model_dump_json(indent=2)}")


if __name__ == "__main__":
    main()
