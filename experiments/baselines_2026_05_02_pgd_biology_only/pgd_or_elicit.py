"""PGD recovery / elicitation training step.

Inputs:
    --step-spec: path to a pre-compiled StepSpec YAML (PGDStep or ElicitStep)
    --no-wandb: disable W&B logging (optional override)
    Environment: $SAESCOPING_ARTIFACTS_LOCATION

Outputs (pgd mode, under artifacts):
    pgd_checkpoints/{model}/{scope_domain}/{sparsity}/checkpoint-{step}/
    pgd_judge_logs/{model}/{scope_domain}/{sparsity}/step_metadata.jsonl
    pgd_judge_logs/{model}/{scope_domain}/{sparsity}/judgements.jsonl
    pgd_judge_logs/{model}/{scope_domain}/{sparsity}/inference.jsonl
    pgd_judge_logs/{model}/{scope_domain}/{sparsity}/scores.json

Outputs (elicit mode, under artifacts):
    elicitation_judge_logs/{model}/{scope_domain}/{elicit_domain}/{sparsity}/step_metadata.jsonl
    elicitation_judge_logs/{model}/{scope_domain}/{elicit_domain}/{sparsity}/judgements.jsonl
    elicitation_judge_logs/{model}/{scope_domain}/{elicit_domain}/{sparsity}/inference.jsonl
    elicitation_judge_logs/{model}/{scope_domain}/{elicit_domain}/{sparsity}/scores.json
    (optionally) elicitation_checkpoints/... if save_elicitation_checkpoints=True

Side effects:
    Logs training metrics + LLM judge scores to W&B.

Idempotency:
    pgd: skips if checkpoint dir is non-empty.
    elicit: skips if step_metadata.jsonl exists in judge logs dir.

Failure mode:
    Partial checkpoints may remain; next run detects completed state via outputs.

TODO(hadriano) not reviewed, might just not work
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import click
import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from trl import SFTConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interface import ElicitStep, ModelConfig, PGDStep, StepSpec, _slash_safe

from sae_scoping.training.pgd_trainer import (
    PGDSFTTrainer,
    filter_masks_by_min_layer_idx,
    freeze_early_side_params,
)
from sae_scoping.training.saliency.wanda import apply_masks_to_model, compute_wanda_masks
from sae_scoping.utils.model_loading import load_model_and_tokenizer


# ── Dry-run ───────────────────────────────────────────────────────────────


def _dry_run_pgd_or_elicit(
    spec: StepSpec,
    checkpoint_dir: Path,
    judge_logs_dir: Path,
) -> None:
    """Dry-run: load model to CPU, save checkpoint without training."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    step = spec.step
    mc = spec.model_cfg
    mode = step.type

    save_ckpt = isinstance(step, PGDStep) or (isinstance(step, ElicitStep) and spec.save_elicitation_checkpoints)
    if save_ckpt:
        print(f"[pgd_or_elicit][dry-run] Loading {mc.model_id} to CPU...")
        model = AutoModelForCausalLM.from_pretrained(mc.model_id, torch_dtype=torch.bfloat16, device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained(mc.model_id)
        ckpt_path = checkpoint_dir / "checkpoint-0"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_path))
        tokenizer.save_pretrained(str(ckpt_path))
        print(f"[pgd_or_elicit][dry-run] Saved stub {mode} checkpoint → {ckpt_path}")
        del model

    # Stub judge logs so downstream sees this step as complete
    stub_meta = {"step_id": "dry_run", "mode": mode, "dry_run": True}
    with open(judge_logs_dir / "step_metadata.jsonl", "w") as f:
        f.write(json.dumps(stub_meta) + "\n")

    meta = {
        "mode": mode,
        "model_id": mc.model_id,
        "scope_domain": step.scope_domain,
        "sparsity": step.sparsity,
        "elicitation_domain": step.elicitation_domain if isinstance(step, ElicitStep) else None,
        "dry_run": True,
    }
    with open(judge_logs_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[pgd_or_elicit][dry-run] Done: {mode} for {mc.model_id}")


# ── Helpers ───────────────────────────────────────────────────────────────


def _compute_eval_loss(model, tokenizer, texts: list[str], max_seq_len: int, batch_size: int) -> float:
    """Compute mean cross-entropy loss on a list of texts."""
    model.eval()
    device = next(model.parameters()).device
    total_loss, n = 0.0, 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_seq_len, padding=True)
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device), labels=enc["input_ids"].to(device))
        total_loss += out.loss.item()
        n += 1
    model.train()
    return total_loss / max(n, 1)


def _count_sparsity(model) -> tuple[float, float]:
    """Return (overall_sparsity, nn_linear_sparsity)."""
    total, zeros, lin_total, lin_zeros = 0, 0, 0, 0
    for name, param in model.named_parameters():
        n = param.numel()
        z = int((param.data == 0).sum().item())
        total += n
        zeros += z
        if ".weight" in name and param.dim() == 2:
            lin_total += n
            lin_zeros += z
    return zeros / max(total, 1), lin_zeros / max(lin_total, 1)


def _resolve_artifacts_root(spec: StepSpec) -> Path:
    base = os.environ.get("SAESCOPING_ARTIFACTS_LOCATION")
    if not base:
        raise click.ClickException("SAESCOPING_ARTIFACTS_LOCATION not set.")
    root = Path(base) / spec.artifacts_subdir
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_sft_config(mc: ModelConfig, output_dir: str) -> SFTConfig:
    """Build TRL SFTConfig from the pre-merged model config sft dict."""
    sft_dict = dict(mc.sft)
    batch_size = sft_dict.pop("train_batch_size", 1)
    sft_dict.setdefault("eval_strategy", "steps")
    sft_dict.setdefault("eval_steps", mc.wrapper.eval_every_steps)
    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=mc.wrapper.eval_batch_size,
        **sft_dict,
    )


def _load_pruned_model(
    model_id: str,
    saliency_path: str,
    sparsity: float,
    min_layer_idx: int | None,
    device: str,
):
    """Load base model, apply saliency mask, optionally freeze early layers.

    Returns (model, tokenizer, masks) ready for PGD training.
    """
    model, tokenizer = load_model_and_tokenizer(model_id, device=device)
    saliency_map = load_file(saliency_path)
    masks = compute_wanda_masks(saliency_map, sparsity)
    apply_masks_to_model(model, masks)

    if min_layer_idx is not None:
        masks = filter_masks_by_min_layer_idx(masks, min_layer_idx)
        frozen_names, n_frozen = freeze_early_side_params(model, min_layer_idx)
        print(f"[pgd_or_elicit] Froze {n_frozen} early-side tensors (layers ≤ {min_layer_idx})")

    return model, tokenizer, masks


def _load_pgd_checkpoint(
    spec: StepSpec,
    device: str,
):
    """Load the best PGD checkpoint for elicitation.

    Performs checkpoint selection: picks the checkpoint with best OOD score
    on the elicitation domain.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    step = spec.step
    assert isinstance(step, ElicitStep)
    artifacts_root = _resolve_artifacts_root(spec)
    ckpt_dir = artifacts_root / step.pgd_checkpoint_dir

    checkpoints = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )

    if not checkpoints:
        raise click.ClickException(f"No checkpoints found in {ckpt_dir}")

    if len(checkpoints) == 1:
        best_ckpt = checkpoints[0]
        print(f"[pgd_or_elicit] Single checkpoint available: {best_ckpt.name}")
    else:
        best_ckpt = _select_best_checkpoint(
            checkpoints,
            ckpt_dir,
            artifacts_root,
            step.scope_domain,
            step.elicitation_domain,
            step.sparsity,
        )
        print(f"[pgd_or_elicit] Selected checkpoint: {best_ckpt.name} (best OOD for {step.elicitation_domain})")

    model = AutoModelForCausalLM.from_pretrained(str(best_ckpt), torch_dtype=torch.bfloat16, device_map=device, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(str(best_ckpt))

    from sae_scoping.training.pgd_trainer import build_pgd_masks_from_model

    masks = build_pgd_masks_from_model(model)

    min_layer_idx = spec.model_cfg.wrapper.min_layer_idx
    if min_layer_idx is not None:
        masks = filter_masks_by_min_layer_idx(masks, min_layer_idx)
        freeze_early_side_params(model, min_layer_idx)

    return model, tokenizer, masks


def _select_best_checkpoint(
    checkpoints: list[Path],
    ckpt_dir: Path,
    artifacts_root: Path,
    scope_domain: str,
    elicitation_domain: str,
    sparsity: float,
) -> Path:
    """Select the PGD checkpoint with best OOD score for the elicitation domain.

    Reads step_metadata.jsonl from pgd_judge_logs, finds the eval entry
    closest-after each checkpoint's step, reads the quality score for the
    target elicitation domain.
    """
    # Derive model_safe from directory structure:
    # pgd_checkpoints/{model_safe}/{scope_domain}/{sparsity}/
    parts = ckpt_dir.relative_to(artifacts_root).parts
    model_safe_from_path = parts[1] if len(parts) >= 2 else "unknown"

    judge_logs_dir = artifacts_root / "pgd_judge_logs" / model_safe_from_path / scope_domain / str(sparsity)
    metadata_path = judge_logs_dir / "step_metadata.jsonl"

    if not metadata_path.exists():
        print(f"[pgd_or_elicit] WARNING: no step_metadata.jsonl at {metadata_path}, using last checkpoint")
        return checkpoints[-1]

    # Parse step_metadata.jsonl
    entries = []
    with open(metadata_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        return checkpoints[-1]

    # For each checkpoint, find the closest-after eval entry
    best_score = -float("inf")
    best_ckpt = checkpoints[-1]

    for ckpt in checkpoints:
        ckpt_step = int(ckpt.name.split("-")[1])
        # Find closest entry with train_step >= ckpt_step
        candidates = [e for e in entries if e.get("train_step", 0) >= ckpt_step]
        if not candidates:
            continue
        closest = min(candidates, key=lambda e: e["train_step"])

        # Read the quality score for the elicitation domain (strip scope label)
        llm_judge = closest.get("llm_judge", {})
        score = _extract_domain_score(llm_judge, elicitation_domain)
        if score > best_score:
            best_score = score
            best_ckpt = ckpt

    return best_ckpt


def _extract_domain_score(llm_judge: dict, domain: str) -> float:
    """Extract the quality/utility score for a domain, stripping scope labels."""
    for key, value in llm_judge.items():
        # Keys look like "llm_judge/{domain}/{scope}/{judge_name}"
        parts = key.split("/")
        if len(parts) >= 4 and parts[1] == domain and parts[3] in ("quality", "utility", "overall"):
            if isinstance(value, (int, float)):
                return float(value)
    # Fallback: average all scores for this domain
    total, count = 0.0, 0
    for key, value in llm_judge.items():
        parts = key.split("/")
        if len(parts) >= 2 and parts[1] == domain and isinstance(value, (int, float)):
            total += float(value)
            count += 1
    return total / max(count, 1)


# ── Eval callback ─────────────────────────────────────────────────────────


class _JsonlSink:
    """Append-only JSONL file writer."""

    def __init__(self, path: Path):
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, row: dict) -> None:
        with open(self._path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")


class _StepInjectingSink:
    """Wraps a sink to prepend train_step to every row."""

    def __init__(self, inner: _JsonlSink):
        self._inner = inner
        self.current_step: int = 0

    def __call__(self, row: dict) -> None:
        self._inner({"train_step": self.current_step, **row})


class RecoveryEvalCallback(TrainerCallback):
    """Periodic evaluation callback with LLM judge + early stopping."""

    def __init__(
        self,
        model,
        tokenizer,
        spec: StepSpec,
        judge_logs_dir: Path,
        vanilla_scores: dict | None,
        wandb_run=None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._spec = spec
        self._mc = spec.model_cfg
        step = spec.step
        self._scope_domain = step.scope_domain
        self._mode = step.type
        self._elicitation_domain = step.elicitation_domain if isinstance(step, ElicitStep) else None
        self._vanilla_scores = vanilla_scores or {}
        self._wandb_run = wandb_run
        self._eval_every = self._mc.wrapper.eval_every_steps

        # Sinks
        self._metadata_sink = _JsonlSink(judge_logs_dir / "step_metadata.jsonl")
        self._judgement_sink = _StepInjectingSink(_JsonlSink(judge_logs_dir / "judgements.jsonl"))
        self._inference_sink = _StepInjectingSink(_JsonlSink(judge_logs_dir / "inference.jsonl"))
        self._scores_path = judge_logs_dir / "scores.json"

        # Load eval data
        self._eval_data = self._load_eval_data()
        self._last_scores: dict = {}

    def _load_eval_data(self) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        judge_cfg = self._spec.llm_judge
        all_domains = self._spec.scope_domains
        questions: dict[str, list[str]] = {}
        answers: dict[str, list[str]] = {}
        for domain in all_domains:
            ds = load_dataset(self._spec.dataset_name, domain, split=judge_cfg.split)
            n = min(judge_cfg.n_samples, len(ds))
            ds_sub = ds.select(range(n))
            questions[domain] = [str(r["question"]) for r in ds_sub]
            answers[domain] = [str(r["answer"]) for r in ds_sub]
        return questions, answers

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        step = state.global_step
        if step % self._eval_every != 0:
            return

        print(f"\n[eval] Step {step}: running evaluation...")
        self._judgement_sink.current_step = step
        self._inference_sink.current_step = step

        # Compute loss
        from sae_scoping.training.pgd_trainer import build_pgd_masks_from_model

        questions, answers = self._eval_data
        model = self._model
        tokenizer = self._tokenizer

        # Run LLM judge if enabled
        scores_dict: dict[str, Any] = {}
        if self._spec.llm_judge.enabled:
            from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval

            judge_cfg = self._spec.llm_judge
            evaluator = OneClickLLMJudgeScopingEval(
                train_domain=self._scope_domain,
                judge_model=judge_cfg.judge_model,
                n_samples=judge_cfg.n_samples,
                generation_kwargs={"do_sample": False, "max_new_tokens": self._mc.wrapper.max_seq_len},
            )
            scores, _ = evaluator.evaluate(
                model,
                tokenizer,
                domain_questions=questions,
                domain_answers=answers,
                judgement_sink=self._judgement_sink,
                inference_sink=self._inference_sink,
            )
            scores_dict = scores

        # Compute loss on a small subset
        loss_texts = []
        for domain_qs in questions.values():
            loss_texts.extend(domain_qs[:10])
        loss = _compute_eval_loss(model, tokenizer, loss_texts[:40], self._mc.wrapper.max_seq_len, self._mc.wrapper.eval_batch_size)
        model_sp, linear_sp = _count_sparsity(model)

        metadata_entry = {
            "train_step": step,
            "loss": loss,
            "nn_linear_sparsity": linear_sp,
            "model_sparsity": model_sp,
            "llm_judge": scores_dict,
        }
        self._metadata_sink(metadata_entry)
        self._last_scores = scores_dict

        # Save latest scores
        with open(self._scores_path, "w") as f:
            json.dump(scores_dict, f, indent=2)

        # W&B logging
        if self._wandb_run is not None:
            log_dict = {f"{self._mode}/loss": loss, f"{self._mode}/linear_sparsity": linear_sp}
            for k, v in scores_dict.items():
                if isinstance(v, (int, float)):
                    log_dict[f"{self._mode}/{k}"] = v
            self._wandb_run.log(log_dict, step=step)

        # Early stopping check
        if self._should_early_stop(scores_dict, step):
            print(f"[eval] Early stopping triggered at step {step}")
            control.should_training_stop = True

    def _should_early_stop(self, scores: dict, step: int) -> bool:
        """Check if scores meet early-stopping thresholds."""
        if not self._vanilla_scores or not scores:
            return False

        if self._mode == "pgd":
            return self._check_pgd_early_stop(scores)
        else:
            return self._check_elicit_early_stop(scores)

    def _check_pgd_early_stop(self, scores: dict) -> bool:
        """PGD stops if relevance AND fluency >= 100% of vanilla."""
        min_rel_frac = self._mc.wrapper.pgd_min_relevance_frac
        min_flu_frac = self._mc.wrapper.pgd_min_fluency_frac

        for key, value in scores.items():
            if not isinstance(value, (int, float)):
                continue
            parts = key.split("/")
            if len(parts) < 4:
                continue
            domain, judge = parts[1], parts[3]
            # Find matching vanilla score (strip scope)
            vanilla_val = self._find_vanilla_score(domain, judge)
            if vanilla_val is None or vanilla_val == 0:
                continue

            frac = value / vanilla_val
            if "relevance" in judge and domain == self._scope_domain:
                if frac >= min_rel_frac:
                    continue
                return False
            if "fluency" in judge and domain == self._scope_domain:
                if frac >= min_flu_frac:
                    continue
                return False

        # All in-scope relevance and fluency meet threshold
        return True

    def _check_elicit_early_stop(self, scores: dict) -> bool:
        """Elicitation stops if OOD domain score >= 90% of vanilla."""
        min_frac = self._mc.wrapper.elicit_min_score_frac
        if self._elicitation_domain is None:
            return False

        for key, value in scores.items():
            if not isinstance(value, (int, float)):
                continue
            parts = key.split("/")
            if len(parts) < 4:
                continue
            domain, judge = parts[1], parts[3]
            if domain != self._elicitation_domain:
                continue
            vanilla_val = self._find_vanilla_score(domain, judge)
            if vanilla_val is None or vanilla_val == 0:
                continue
            frac = value / vanilla_val
            if frac >= min_frac:
                return True

        return False

    def _find_vanilla_score(self, domain: str, judge: str) -> float | None:
        """Find vanilla baseline score matching (domain, judge), ignoring scope."""
        for key, value in self._vanilla_scores.items():
            if not isinstance(value, (int, float)):
                continue
            parts = key.split("/")
            if len(parts) >= 4 and parts[1] == domain and parts[3] == judge:
                return float(value)
        return None


# ── Training logic ────────────────────────────────────────────────────────


def _format_example(example):
    example["text"] = f"Question: {example['question']}\nAnswer: {example['answer']}"
    return example


def _prepare_train_dataset(spec: StepSpec, domain: str):
    """Load and prepare the training dataset for SFT."""
    ds = load_dataset(spec.dataset_name, domain, split="train")
    n = min(spec.n_train, len(ds))
    return ds.select(range(n)).map(_format_example)


def _prepare_eval_dataset(spec: StepSpec, domain: str):
    """Load validation split for TRL's built-in eval loop (val loss)."""
    ds = load_dataset(spec.dataset_name, domain, split="validation")
    n = min(spec.n_eval, len(ds))
    return ds.select(range(n)).map(_format_example)


def _run_training(
    model,
    tokenizer,
    masks: dict[str, torch.Tensor],
    sft_config: SFTConfig,
    train_dataset,
    eval_dataset,
    eval_callback: RecoveryEvalCallback,
) -> None:
    """Run PGD training with the eval callback."""
    trainer = PGDSFTTrainer(
        masks=masks,
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[eval_callback],
    )
    trainer.train()


# ── CLI ───────────────────────────────────────────────────────────────────


@click.command()
@click.option("--step-spec", required=True, type=click.Path(exists=True))
@click.option("--no-wandb", is_flag=True, default=False)
def main(step_spec: str, no_wandb: bool) -> None:
    """Run PGD recovery or elicitation training.

    CONTRACT: exit code must be 0 iff the step fully succeeded.
    The scheduler treats any non-zero exit as FAILED and skips all dependents.
    """
    spec = StepSpec.from_yaml(step_spec)
    if spec.dry_run:
        print("[pgd_or_elicit] *** DRY RUN MODE ***")
    step = spec.step
    assert isinstance(step, (PGDStep, ElicitStep))
    mc = spec.model_cfg
    mode = step.type
    scope_domain = step.scope_domain
    sparsity = step.sparsity
    model_safe = _slash_safe(mc.model_id)
    artifacts_root = _resolve_artifacts_root(spec)
    elicitation_domain = step.elicitation_domain if isinstance(step, ElicitStep) else None

    # Determine output paths
    if isinstance(step, PGDStep):
        checkpoint_dir = artifacts_root / step.checkpoint_dir
        judge_logs_dir = artifacts_root / "pgd_judge_logs" / model_safe / scope_domain / str(sparsity)
    else:
        checkpoint_dir = artifacts_root / step.checkpoint_dir
        judge_logs_dir = artifacts_root / "elicitation_judge_logs" / model_safe / scope_domain / step.elicitation_domain / str(sparsity)

    # Idempotency check
    # TODO(hadriano) lots of duplicated code from the AI here :/
    if isinstance(step, PGDStep):
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            if not spec.no_cache:
                print(f"[pgd_or_elicit] Output exists, skipping: {checkpoint_dir}")
                return
    else:
        if (judge_logs_dir / "step_metadata.jsonl").exists():
            if not spec.no_cache:
                print(f"[pgd_or_elicit] Output exists, skipping: {judge_logs_dir}")
                return

    judge_logs_dir.mkdir(parents=True, exist_ok=True)

    if spec.dry_run:
        _dry_run_pgd_or_elicit(spec, checkpoint_dir, judge_logs_dir)
        return

    raise NotImplementedError("W&B setup not implemented")  # TODO(hadriano) toggle on the rest

    # # W&B setup
    # wandb_run = None
    # if spec.wandb.enabled and not no_wandb:
    #     import wandb

    #     os.environ["WANDB_DIR"] = str(artifacts_root / "wandb")
    #     run_name = f"{mode}__{mc.model_id.split('/')[-1]}__{scope_domain}__sp{sparsity}"
    #     if elicitation_domain:
    #         run_name += f"__{elicitation_domain}"
    #     wandb_run = wandb.init(
    #         project=spec.wandb.project,
    #         name=run_name,
    #         config={
    #             "model_id": mc.model_id,
    #             "scope_domain": scope_domain,
    #             "sparsity": sparsity,
    #             "mode": mode,
    #             "elicitation_domain": elicitation_domain,
    #         },
    #     )

    # print(f"[pgd_or_elicit] Mode: {mode}")
    # print(f"[pgd_or_elicit] Model: {mc.model_id}")
    # print(f"[pgd_or_elicit] Scope: {scope_domain}, Sparsity: {sparsity}")
    # if elicitation_domain:
    #     print(f"[pgd_or_elicit] Elicitation domain: {elicitation_domain}")

    # # Load model
    # if isinstance(step, PGDStep):
    #     saliency_path = str(artifacts_root / step.saliency_path)
    #     model, tokenizer, masks = _load_pruned_model(mc.model_id, saliency_path, sparsity, mc.wrapper.min_layer_idx, spec.device)
    # else:
    #     model, tokenizer, masks = _load_pgd_checkpoint(spec, spec.device)

    # # Load vanilla scores for early stopping
    # vanilla_scores_path = artifacts_root / "saliency_maps" / model_safe / scope_domain / "vanilla_scores.json"
    # vanilla_scores = None
    # if vanilla_scores_path.exists():
    #     with open(vanilla_scores_path) as f:
    #         vanilla_scores = json.load(f)
    #     print(f"[pgd_or_elicit] Loaded vanilla scores from {vanilla_scores_path}")

    # # Determine training domain and output dir for SFT
    # train_domain = scope_domain if isinstance(step, PGDStep) else elicitation_domain
    # save_checkpoints = True if isinstance(step, PGDStep) else spec.save_elicitation_checkpoints
    # sft_output = str(checkpoint_dir) if save_checkpoints else str(judge_logs_dir / "trl_output")

    # if save_checkpoints:
    #     checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # # Build SFT config (sft_overrides already merged at compile time)
    # sft_config = _build_sft_config(mc, sft_output)
    # if not save_checkpoints:
    #     sft_config.save_strategy = "no"

    # # Prepare dataset
    # train_dataset = _prepare_train_dataset(spec, train_domain)

    # # Create eval callback
    # eval_callback = RecoveryEvalCallback(
    #     model=model,
    #     tokenizer=tokenizer,
    #     spec=spec,
    #     judge_logs_dir=judge_logs_dir,
    #     vanilla_scores=vanilla_scores,
    #     wandb_run=wandb_run,
    # )

    # # Train
    # print(f"[pgd_or_elicit] Starting {mode} training...")
    # _run_training(model, tokenizer, masks, sft_config, train_dataset, eval_callback)

    # # Save final metadata
    # meta = {
    #     "mode": mode,
    #     "model_id": mc.model_id,
    #     "scope_domain": scope_domain,
    #     "sparsity": sparsity,
    #     "elicitation_domain": elicitation_domain,
    # }
    # with open(judge_logs_dir / "metadata.json", "w") as f:
    #     json.dump(meta, f, indent=2)

    # if wandb_run is not None:
    #     wandb_run.finish()

    # print(f"[pgd_or_elicit] Done: {mode} for {mc.model_id} / {scope_domain} / sp{sparsity}")


if __name__ == "__main__":
    main()
