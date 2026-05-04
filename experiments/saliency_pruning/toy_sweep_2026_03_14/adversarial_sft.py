"""
adversarial_sft.py

SFT a model on an out-of-distribution domain with periodic chat evaluation
(generate + LLM-judge grading).  Works identically for regular HuggingFace
models and locally-saved pruned models — both are loaded via
``AutoModelForCausalLM.from_pretrained``.

Use case: test whether a pruned-and-recovered model can be adversarially
fine-tuned to regain out-of-domain capabilities (tamper resistance).

Pipeline:
    1. Load model + tokenizer (HF id or local path).
    2. Load OOD train set for SFT and OOD validation set for chat eval.
    3. Run SFT with a ``ChatEvalCallback`` that periodically generates
       responses on the validation set, grades them with LLM judges, and
       logs per-judge scores to WandB.
    4. Write a JSON result with the full chat-eval history over training.

CLI usage:
    python adversarial_sft.py \\
        --model-id google/gemma-2-9b-it \\
        --dataset-subset physics \\
        --max-steps 4000 \\
        --chat-eval-every 250 \\
        --n-chat-eval 40 \\
        --output-json result.json

    python adversarial_sft.py \\
        --model-id ./recovery_taylor_s025_phase2/final_model \\
        --dataset-subset physics \\
        --max-steps 4000 \\
        --output-json result_pgd_s025.json
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
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
    load_qa_dataset,
)
from grade_chats import grade_chats
from model_generator import HFGenerator


# ---------------------------------------------------------------------------
# Result schemas
# ---------------------------------------------------------------------------


class ChatEvalResult(pydantic.BaseModel):
    """Single chat-evaluation snapshot at a given training step."""

    step: int
    judge_name2mean_scores: dict[str, float]
    overall_mean_score: float


class AdversarialSFTResult(pydantic.BaseModel):
    """Full result of an adversarial SFT run."""

    model_id: str
    dataset_subset: str
    max_steps: int
    total_steps: int
    chat_eval_history: list[ChatEvalResult]


# ---------------------------------------------------------------------------
# Chat-evaluation callback
# ---------------------------------------------------------------------------


class ChatEvalCallback(TrainerCallback):
    """Periodically generate responses and grade them with LLM judges.

    This callback does **not** stop training — it is purely for logging.
    Every ``eval_every`` optimiser steps it:

    1. Puts the model in eval mode.
    2. Generates responses for ``eval_conversations`` (greedy decoding).
    3. Grades them with the default judges (answering, factual_helpful, precise).
    4. Saves the raw generated chats to ``chats_dir/chats_step_NNNN.json``.
    5. Appends a scores line to ``scores_jsonl_path`` (crash-safe).
    6. Logs per-judge and overall scores to WandB (if active).
    7. Appends a :class:`ChatEvalResult` to ``self.history``.
    8. Restores the model to train mode.

    Args:
        eval_every: Run chat evaluation every N training steps.
        tokenizer: Tokenizer for generation (padding_side temporarily set
            to ``"left"`` during generation).
        eval_conversations: 0-turn OpenAI conversations (user questions
            only) to generate on.
        batch_size: Batch size for generation.
        max_new_tokens: Max new tokens per generated response.
        scores_jsonl_path: Path to a JSONL file where each eval step
            appends one line with scores.  Crash-safe: each line is
            flushed immediately after writing.
        chats_dir: Directory where raw generated chats are saved, one
            JSON file per eval step (``chats_step_0250.json``, etc.).
    """

    def __init__(
        self,
        eval_every: int,
        tokenizer: PreTrainedTokenizerBase,
        eval_conversations: list[list[dict]],
        batch_size: int = 20,
        max_new_tokens: int = 256,
        scores_jsonl_path: Optional[Path] = None,
        chats_dir: Optional[Path] = None,
    ) -> None:
        self.eval_every = eval_every
        self.tokenizer = tokenizer
        self.eval_conversations = eval_conversations
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.scores_jsonl_path = scores_jsonl_path
        self.chats_dir = chats_dir
        self.history: list[ChatEvalResult] = []

        # Create directories eagerly so permission errors surface early.
        if self.scores_jsonl_path is not None:
            self.scores_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        if self.chats_dir is not None:
            self.chats_dir.mkdir(parents=True, exist_ok=True)

    def _generate_chats(
        self, model: PreTrainedModel,
    ) -> list[list[dict]]:
        """Generate responses (GPU work). Separated from grading so chats
        can be persisted to disk before the judge API is called."""
        model.eval()
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        generator = HFGenerator(model, self.tokenizer)
        generation_kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        completed = generator.generate(
            self.eval_conversations,
            batch_size=self.batch_size,
            generation_kwargs=generation_kwargs,
        )

        self.tokenizer.padding_side = original_padding_side
        model.train()
        return completed

    def _grade_chats(self, chats: list[list[dict]], step: int) -> ChatEvalResult:
        """Grade already-generated chats with LLM judges."""
        graded = grade_chats(chats)
        return ChatEvalResult(
            step=step,
            judge_name2mean_scores=graded.judge_name2mean_scores,
            overall_mean_score=graded.overall_mean_score,
        )

    def _flush_scores(self, result: ChatEvalResult) -> None:
        """Append one JSON line to the scores JSONL file."""
        if self.scores_jsonl_path is None:
            return
        line = result.model_dump_json() + "\n"
        with open(self.scores_jsonl_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
        print(f"  Appended scores to {self.scores_jsonl_path}")

    def _save_chats(self, step: int, chats: list[list[dict]]) -> None:
        """Save raw generated chats to a per-step JSON file."""
        if self.chats_dir is None:
            return
        chats_path = self.chats_dir / f"chats_step_{step:04d}.json"
        chats_path.write_text(json.dumps(chats, indent=2), encoding="utf-8")
        print(f"  Saved {len(chats)} chats to {chats_path}")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Optional[PreTrainedModel] = None,
        **kwargs,
    ) -> TrainerControl:
        if state.global_step % self.eval_every != 0 or model is None:
            return control

        # 1. Generate (expensive GPU work)
        completed_chats = self._generate_chats(model)

        # 2. Save chats BEFORE grading — if the judge API fails we still
        #    have the generated responses on disk.
        self._save_chats(state.global_step, completed_chats)

        # 3. Grade with LLM judges (API calls, may fail)
        result = self._grade_chats(completed_chats, state.global_step)
        self.history.append(result)

        # 4. Persist scores
        self._flush_scores(result)

        scores_str = ", ".join(
            f"{k}={v:.3f}" for k, v in result.judge_name2mean_scores.items()
        )
        print(
            f"  [Chat eval step {state.global_step}] "
            f"overall={result.overall_mean_score:.4f} ({scores_str})"
        )

        # Log to wandb if active
        if state.is_world_process_zero:
            try:
                import wandb

                if wandb.run is not None:
                    log_dict = {
                        "chat_eval/overall_mean_score": result.overall_mean_score,
                    }
                    for k, v in result.judge_name2mean_scores.items():
                        log_dict[f"chat_eval/{k}"] = v
                    wandb.log(log_dict, step=state.global_step)
            except ImportError:
                pass

        return control


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def adversarial_sft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset_train: Dataset,
    eval_conversations: list[list[dict]],
    model_id: str = "unknown",
    dataset_subset: str = "unknown",
    max_steps: int = 4000,
    chat_eval_every: int = 250,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    learning_rate: float = 1e-5,
    max_seq_len: int = 1024,
    max_new_tokens: int = 256,
    chat_eval_batch_size: int = 20,
    output_dir: str = "./adversarial_output",
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
) -> AdversarialSFTResult:
    """SFT a model on a dataset with periodic chat evaluation.

    Works for both regular and pruned models — both are loaded identically
    via ``AutoModelForCausalLM.from_pretrained``.

    Args:
        model: Model to fine-tune. Modified in-place.
        tokenizer: Tokenizer matching the model.
        dataset_train: Training dataset (must have question/answer columns).
        eval_conversations: 0-turn OpenAI conversations for chat eval.
        model_id: Model identifier (for result metadata).
        dataset_subset: Dataset subset name (for result metadata).
        max_steps: Maximum SFT training steps.
        chat_eval_every: Run chat evaluation every N steps.
        batch_size: Per-device training batch size.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Learning rate for AdamW.
        max_seq_len: Maximum sequence length for tokenization.
        max_new_tokens: Max tokens for generation during chat eval.
        chat_eval_batch_size: Batch size for generation during chat eval.
        output_dir: Directory for trainer outputs.
        wandb_project: WandB project name. If None, WandB is disabled.
        wandb_run_name: WandB run name.

    Returns:
        AdversarialSFTResult with model info and chat eval history.
    """
    for p in model.parameters():
        p.requires_grad = True

    sft_dataset = format_as_sft_dataset(dataset_train, tokenizer)

    out_root = Path(output_dir)
    scores_jsonl_path = out_root / "chat_eval_scores.jsonl"
    chats_dir = out_root / "chat_eval_chats"

    # Clear stale logs from a previous run so appended JSONL lines and
    # leftover chat files don't mix with fresh data.
    if scores_jsonl_path.exists():
        scores_jsonl_path.unlink()
        print(f"Removed stale {scores_jsonl_path}")
    if chats_dir.exists():
        shutil.rmtree(chats_dir)
        print(f"Removed stale {chats_dir}")

    callback = ChatEvalCallback(
        eval_every=chat_eval_every,
        tokenizer=tokenizer,
        eval_conversations=eval_conversations,
        batch_size=chat_eval_batch_size,
        max_new_tokens=max_new_tokens,
        scores_jsonl_path=scores_jsonl_path,
        chats_dir=chats_dir,
    )

    if wandb_project is not None:
        os.environ["WANDB_PROJECT"] = wandb_project

    use_cuda = torch.cuda.is_available() and model.device.type == "cuda"
    training_args = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=use_cuda,
        save_strategy="no",
        report_to="wandb" if wandb_project is not None else "none",
        run_name=wandb_run_name,
        max_length=max_seq_len,
        dataset_text_field="text",
        logging_steps=10,
        optim="adamw_bnb_8bit" if use_cuda else "adamw_torch",
        no_cuda=not use_cuda,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=sft_dataset,
        args=training_args,
        callbacks=[callback],
    )

    print(
        f"Starting adversarial SFT on '{dataset_subset}' "
        f"(max_steps={max_steps}, chat_eval_every={chat_eval_every})..."
    )
    trainer.train()

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return AdversarialSFTResult(
        model_id=model_id,
        dataset_subset=dataset_subset,
        max_steps=max_steps,
        total_steps=trainer.state.global_step,
        chat_eval_history=callback.history,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_CHAT_TEMPLATE_PATH = (
    Path(__file__).parent / "prompts" / "gemma2_chat_template_system_prompt.j2"
)


@click.command()
@click.option(
    "--model-id",
    type=str,
    default="google/gemma-2-9b-it",
    show_default=True,
    help="HuggingFace model ID or local path to a saved model.",
)
@click.option(
    "--dataset-name",
    type=str,
    default="4gate/StemQAMixture",
    show_default=True,
)
@click.option(
    "--dataset-subset",
    type=str,
    default="physics",
    show_default=True,
    help="Dataset subset for OOD SFT (e.g. physics, math, chemistry).",
)
@click.option("--n-train", type=int, default=500, show_default=True)
@click.option("--n-chat-eval", type=int, default=40, show_default=True)
@click.option("--max-steps", type=int, default=4000, show_default=True)
@click.option("--chat-eval-every", type=int, default=250, show_default=True)
@click.option("--batch-size", type=int, default=1, show_default=True)
@click.option(
    "--gradient-accumulation-steps", type=int, default=16, show_default=True
)
@click.option("--learning-rate", type=float, default=1e-5, show_default=True)
@click.option("--max-seq-len", type=int, default=1024, show_default=True)
@click.option("--max-new-tokens", type=int, default=256, show_default=True)
@click.option("--chat-eval-batch-size", type=int, default=20, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--output-dir",
    type=str,
    default="./adversarial_output",
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
    "--force",
    is_flag=True,
    default=False,
    help="Re-run even if --output-json already exists.",
)
def main(
    model_id: str,
    dataset_name: str,
    dataset_subset: str,
    n_train: int,
    n_chat_eval: int,
    max_steps: int,
    chat_eval_every: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    max_seq_len: int,
    max_new_tokens: int,
    chat_eval_batch_size: int,
    seed: int,
    output_dir: str,
    output_json: Optional[Path],
    device: str,
    wandb_project: Optional[str],
    wandb_run_name: Optional[str],
    force: bool,
) -> None:
    """SFT a model on an OOD domain with periodic chat evaluation."""
    # Output-caching guard
    if output_json is not None and output_json.exists() and not force:
        print(
            f"Skipping: output already exists at {output_json} "
            f"(pass --force to rerun)"
        )
        return

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if _CHAT_TEMPLATE_PATH.exists():
        tokenizer.chat_template = _CHAT_TEMPLATE_PATH.read_text()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    dataset_train = load_qa_dataset(
        dataset_name, dataset_subset, split="train", n=n_train, seed=seed,
    )
    dataset_eval = load_qa_dataset(
        dataset_name, dataset_subset, split="validation", n=n_chat_eval, seed=seed,
    )
    eval_conversations = format_as_0turn(dataset_eval)

    print(
        f"Loaded {len(dataset_train)} train rows, "
        f"{len(dataset_eval)} chat-eval rows from {dataset_name}/{dataset_subset}"
    )

    result = adversarial_sft(
        model=model,
        tokenizer=tokenizer,
        dataset_train=dataset_train,
        eval_conversations=eval_conversations,
        model_id=model_id,
        dataset_subset=dataset_subset,
        max_steps=max_steps,
        chat_eval_every=chat_eval_every,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        chat_eval_batch_size=chat_eval_batch_size,
        output_dir=output_dir,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )

    result_json = result.model_dump_json(indent=2)
    print(f"\nResult:\n{result_json}")
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to temp file then rename, so a crash mid-write
        # never leaves a half-written result JSON on disk.
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=output_json.parent, suffix=".tmp", prefix=output_json.stem,
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(result_json)
            os.replace(tmp_path, output_json)
        except BaseException:
            os.unlink(tmp_path)
            raise
        print(f"Wrote result to {output_json}")


if __name__ == "__main__":
    main()
