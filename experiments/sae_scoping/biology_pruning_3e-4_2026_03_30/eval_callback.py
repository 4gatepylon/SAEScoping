"""
eval_callback.py

Extensible TrainerCallback for utility evaluation during training.
Loss is handled by SFTTrainer's built-in eval; this callback is for metrics
that require generation (e.g. LLM judge).

Wandb step alignment note:
  This callback logs to "utility_eval/<metric_name>" with a separate
  "trainer/global_step" key. SFTTrainer logs its own metrics under
  "train/*". These use different wandb step counters, so when viewing
  both on the same chart, set the x-axis to "trainer/global_step"
  explicitly rather than relying on wandb's default step.

To add a new metric:
  1. Write a function matching MetricFn signature
  2. Register it via register_metric().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import torch
import wandb
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from evaluation.generic_judges import grade_chats
from inference.model_generator import HFGenerator

_OOM_EXIT_FILENAME = "exit_reason.json"


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

# Signature: (model, tokenizer, **config_kwargs) -> float
MetricFn = Callable[..., float]

_METRIC_REGISTRY: dict[str, MetricFn] = {}


def register_metric(name: str, fn: MetricFn) -> None:
    _METRIC_REGISTRY[name] = fn


def _get_metric_fn(name: str) -> MetricFn:
    if name not in _METRIC_REGISTRY:
        raise KeyError(
            f"Unknown metric '{name}'. Available: {sorted(_METRIC_REGISTRY.keys())}"
        )
    return _METRIC_REGISTRY[name]


# ---------------------------------------------------------------------------
# Built-in metric: LLM judge (requires generation + grading)
# ---------------------------------------------------------------------------


def _metric_judge(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    eval_conversations: list[list[dict]],
    batch_size: int = 4,
    max_new_tokens: int = 256,
    **_kwargs,
) -> float:
    """Generate responses then grade with LLM judges. Returns overall mean score."""
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        generator = HFGenerator(model, tokenizer)
        generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
        completed = generator.generate(
            eval_conversations, batch_size=batch_size, generation_kwargs=generation_kwargs,
        )
        graded = grade_chats(completed)
        return graded.overall_mean_score
    finally:
        tokenizer.padding_side = original_padding_side


register_metric("judge", _metric_judge)


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


class UtilityEvalCallback(TrainerCallback):
    """
    Periodically evaluate model utility during training and log to wandb.
    Validation loss is already handled by SFTTrainer's built-in eval loop;
    this callback is for generation-based metrics (e.g. LLM judge).

    On CUDA OOM during eval: saves a checkpoint to output_dir, writes an
    exit_reason.json, and stops training gracefully.

    Args:
        eval_every: Run evaluation every N training steps (optimizer steps,
            not micro-batches — with accum=8, step 1 = 8 forward passes).
        metric_name: Name of registered metric (default: "judge").
        tokenizer: Tokenizer for the model.
        eval_conversations: 0-turn OpenAI conversations for generation+grading.
        batch_size: Batch size for generation.
        max_new_tokens: Max generation tokens.
        log_to_wandb: Whether to log metrics to wandb.
    """

    def __init__(
        self,
        eval_every: int,
        metric_name: str = "judge",
        tokenizer: PreTrainedTokenizerBase | None = None,
        eval_conversations: list[list[dict]] | None = None,
        batch_size: int = 4,
        max_new_tokens: int = 256,
        log_to_wandb: bool = True,
    ) -> None:
        self.eval_every = eval_every
        self.metric_name = metric_name
        self.tokenizer = tokenizer
        self.eval_conversations = eval_conversations or []
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.log_to_wandb = log_to_wandb
        self._metric_fn = _get_metric_fn(metric_name)
        self.metric_history: list[tuple[int, float]] = []

    def _compute_metric(self, model: PreTrainedModel) -> float:
        model.eval()
        with torch.no_grad():
            return self._metric_fn(
                model=model,
                tokenizer=self.tokenizer,
                eval_conversations=self.eval_conversations,
                batch_size=self.batch_size,
                max_new_tokens=self.max_new_tokens,
            )

    def _handle_oom(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
    ) -> TrainerControl:
        """On OOM: save checkpoint, write exit reason, stop training."""
        print("CUDA OOM during utility eval. Saving checkpoint and stopping.")
        torch.cuda.empty_cache()
        oom_ckpt_dir = Path(args.output_dir) / f"oom_checkpoint-{state.global_step}"
        model.save_pretrained(str(oom_ckpt_dir))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(oom_ckpt_dir))
        exit_reason = {
            "reason": "OOM",
            "step": state.global_step,
            "context": "utility_eval_callback",
            "metric_history": self.metric_history,
        }
        exit_file = Path(args.output_dir) / _OOM_EXIT_FILENAME
        exit_file.parent.mkdir(parents=True, exist_ok=True)
        exit_file.write_text(json.dumps(exit_reason, indent=2))
        print(f"Checkpoint saved to {oom_ckpt_dir}")
        print(f"Exit reason written to {exit_file}")
        control.should_training_stop = True
        return control

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[PreTrainedModel] = None,
        **kwargs,
    ) -> TrainerControl:
        if model is None or state.global_step == 0:
            return control
        if state.global_step % self.eval_every != 0:
            return control

        try:
            metric = self._compute_metric(model)
        except torch.cuda.OutOfMemoryError:
            return self._handle_oom(args, state, control, model)

        self.metric_history.append((state.global_step, metric))
        print(
            f"  [UtilityEval step {state.global_step}] "
            f"{self.metric_name}={metric:.4f}"
        )

        if self.log_to_wandb and wandb.run is not None:
            wandb.log({
                f"utility_eval/{self.metric_name}": metric,
                "trainer/global_step": state.global_step,
            })

        model.train()
        return control


if __name__ == "__main__":
    # Test: PYTHONPATH=. python eval_callback.py
    # Runs a tiny SFT training loop with the judge-based UtilityEvalCallback,
    # logs to a disposable wandb project, and prompts you to verify the plot.
    # Requires OPENAI_API_KEY set (for LLM judge via litellm).
    import click
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    wandb_project = "deleteme-eval-callback-test"
    wandb_run_name = "eval-callback-smoke-test"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    num_steps = 50
    utility_eval_every = 25
    hf_eval_every = 10

    import os
    os.environ["WANDB_PROJECT"] = wandb_project

    print("=== Loading tiny model + tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("=== Building dummy train + eval datasets ===")
    dummy_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"What is {i}+{i}?"},
             {"role": "assistant", "content": str(i + i)}],
            tokenize=False,
        )
        for i in range(100)
    ]
    train_ds = Dataset.from_dict({"text": dummy_texts[:80]})
    eval_ds = Dataset.from_dict({"text": dummy_texts[80:]})

    eval_conversations = [
        [{"role": "user", "content": f"What is {i}*{i}?"}]
        for i in range(5)
    ]

    print(f"=== Creating UtilityEvalCallback (judge, every {utility_eval_every} steps) ===")
    cb = UtilityEvalCallback(
        eval_every=utility_eval_every,
        metric_name="judge",
        tokenizer=tokenizer,
        eval_conversations=eval_conversations,
        batch_size=4,
        max_new_tokens=64,
        log_to_wandb=True,
    )

    print(f"=== Training for {num_steps} steps (HF eval loss every {hf_eval_every}) ===")
    sft_config = SFTConfig(
        output_dir="./deleteme_eval_cb_test",
        max_steps=num_steps,
        per_device_train_batch_size=2,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=hf_eval_every,
        save_strategy="no",
        report_to="wandb",
        run_name=wandb_run_name,
        bf16=True,
        max_length=128,
        gradient_checkpointing=False,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        callbacks=[cb],
    )
    trainer.train()

    print("\n=== Metric history ===")
    for step, val in cb.metric_history:
        print(f"  step {step}: judge={val:.4f}")

    print(f"\nCheck wandb project '{wandb_project}' run '{wandb_run_name}'.")
    print(f"You should see utility_eval/judge logged every {utility_eval_every} steps.")

    if wandb.run is not None:
        run_url = wandb.run.get_url()
        print(f"Run URL: {run_url}")
        wandb.finish()

    if click.confirm("Does the wandb plot look OK? Delete the test run?"):
        import subprocess
        subprocess.run(
            ["wandb", "run", "delete", f"{wandb_project}/{wandb_run_name}"],
            check=False,
        )
        print("Attempted deletion. You may need to delete manually from the wandb UI.")
    else:
        print(f"Keeping run. Delete later: wandb UI -> project '{wandb_project}'")

    import shutil
    shutil.rmtree("./deleteme_eval_cb_test", ignore_errors=True)
    print("[OK] eval_callback.py integration test done")
