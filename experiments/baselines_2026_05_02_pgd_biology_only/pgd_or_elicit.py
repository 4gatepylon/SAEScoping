"""PGD recovery / elicitation training step.

Callee script invoked by the scheduler for PGD and elicit steps.
Reads a pre-compiled StepSpec YAML; writes checkpoints + judge logs
under $SAESCOPING_ARTIFACTS_LOCATION.  Idempotent via COMPLETED markers.

CONTRACT: exit code 0 iff the step fully succeeded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))

from callbacks import RecoveryEvalCallback, StepInjectingSink, VanillaFracEarlyStoppingCallback
from interface import ElicitStep, ModelConfig, PGDStep, StepSpec, _slash_safe
from utils import maybe_init_wandb, resolve_artifacts_root

from sae_scoping.datasets.qa_datasets import format_as_sft_dataset, format_as_sft_text
from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval
from sae_scoping.evaluation.utils import JsonlSink
from sae_scoping.training.pgd_trainer import (
    PGDSFTTrainer,
    build_pgd_masks_from_model,
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

    if save_ckpt:
        (checkpoint_dir / "COMPLETED").write_text(f"step_id={mode}\n")
    (judge_logs_dir / "COMPLETED").write_text(f"step_id={mode}\n")

    print(f"[pgd_or_elicit][dry-run] Done: {mode} for {mc.model_id}")


# ── Helpers ───────────────────────────────────────────────────────────────


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
    step = spec.step
    assert isinstance(step, ElicitStep)
    artifacts_root = resolve_artifacts_root(spec)
    ckpt_dir = artifacts_root / step.pgd_checkpoint_dir

    checkpoints = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )

    if not checkpoints:
        raise click.ClickException(f"No checkpoints found in {ckpt_dir}")

    # TODO: select checkpoint with best OOD score on the elicitation domain
    # by joining checkpoint steps against step_metadata.jsonl eval entries.
    # For now, just pick the last (highest-step) checkpoint.
    best_ckpt = checkpoints[-1]
    print(f"[pgd_or_elicit] Using checkpoint: {best_ckpt.name} (last of {len(checkpoints)})")

    model = AutoModelForCausalLM.from_pretrained(str(best_ckpt), torch_dtype=torch.bfloat16, device_map=device, attn_implementation="eager")
    tokenizer = AutoTokenizer.from_pretrained(str(best_ckpt))
    masks = build_pgd_masks_from_model(model)

    min_layer_idx = spec.model_cfg.wrapper.min_layer_idx
    if min_layer_idx is not None:
        masks = filter_masks_by_min_layer_idx(masks, min_layer_idx)
        freeze_early_side_params(model, min_layer_idx)

    return model, tokenizer, masks


# ── Training logic ────────────────────────────────────────────────────────


def _prepare_dataset(spec: StepSpec, domain: str, tokenizer, split: str, max_n: int):
    """Load a dataset split, truncate to max_n, and format for SFT."""
    ds = load_dataset(spec.dataset_name, domain, split=split)
    return format_as_sft_dataset(ds.select(range(min(max_n, len(ds)))), tokenizer)


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
    artifacts_root = resolve_artifacts_root(spec)
    elicitation_domain = step.elicitation_domain if isinstance(step, ElicitStep) else None

    # Determine output paths
    checkpoint_dir = artifacts_root / step.checkpoint_dir
    if isinstance(step, PGDStep):
        judge_logs_dir = artifacts_root / "pgd_judge_logs" / model_safe / scope_domain / str(sparsity)
    else:
        judge_logs_dir = artifacts_root / "elicitation_judge_logs" / model_safe / scope_domain / step.elicitation_domain / str(sparsity)

    # Idempotency: skip only if COMPLETED marker exists (written last by previous run)
    if isinstance(step, PGDStep):
        has_completed = (checkpoint_dir / "COMPLETED").exists()
    else:
        has_completed = (judge_logs_dir / "COMPLETED").exists()
    if has_completed and not spec.no_cache:
        print(f"[pgd_or_elicit] COMPLETED marker found, skipping: {mode}")
        return

    judge_logs_dir.mkdir(parents=True, exist_ok=True)

    if spec.dry_run:
        _dry_run_pgd_or_elicit(spec, checkpoint_dir, judge_logs_dir)
        return

    # W&B setup
    run_name = f"{mode}__{mc.model_id.split('/')[-1]}__{scope_domain}__sp{sparsity}"
    if elicitation_domain:
        run_name += f"__{elicitation_domain}"
    wandb_run = maybe_init_wandb(
        spec,
        artifacts_root,
        name=run_name,
        no_wandb=no_wandb,
        config={"model_id": mc.model_id, "scope_domain": scope_domain, "sparsity": sparsity, "mode": mode, "elicitation_domain": elicitation_domain},
    )

    print(f"[pgd_or_elicit] Mode: {mode}")
    print(f"[pgd_or_elicit] Model: {mc.model_id}")
    print(f"[pgd_or_elicit] Scope: {scope_domain}, Sparsity: {sparsity}")
    if elicitation_domain:
        print(f"[pgd_or_elicit] Elicitation domain: {elicitation_domain}")

    # Load model
    if isinstance(step, PGDStep):
        saliency_path = str(artifacts_root / step.saliency_path)
        model, tokenizer, masks = _load_pruned_model(mc.model_id, saliency_path, sparsity, mc.wrapper.min_layer_idx, spec.device)
    else:
        model, tokenizer, masks = _load_pgd_checkpoint(spec, spec.device)

    # Load vanilla scores for early stopping
    vanilla_scores_path = artifacts_root / "saliency_maps" / model_safe / scope_domain / "vanilla_scores.json"
    vanilla_scores = None
    if vanilla_scores_path.exists():
        with open(vanilla_scores_path) as f:
            vanilla_scores = json.load(f)
        print(f"[pgd_or_elicit] Loaded vanilla scores from {vanilla_scores_path}")

    # Determine training domain and output dir for SFT
    train_domain = scope_domain if isinstance(step, PGDStep) else elicitation_domain
    save_checkpoints = True if isinstance(step, PGDStep) else spec.save_elicitation_checkpoints
    sft_output = str(checkpoint_dir) if save_checkpoints else str(judge_logs_dir / "trl_output")

    if save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build SFT config (sft_overrides already merged at compile time)
    sft_config = _build_sft_config(mc, sft_output)
    if not save_checkpoints:
        sft_config.save_strategy = "no"

    # Prepare datasets
    train_dataset = _prepare_dataset(spec, train_domain, tokenizer, "train", spec.n_train)
    eval_dataset = _prepare_dataset(spec, train_domain, tokenizer, "validation", spec.n_eval)

    # ── Eval data + sinks ───────────────────────────────────────────────
    judge_cfg = spec.llm_judge
    eval_questions: dict[str, list[str]] = {}
    eval_answers: dict[str, list[str]] = {}
    loss_texts: list[str] = []
    for domain in spec.scope_domains:
        ds = load_dataset(spec.dataset_name, domain, split=judge_cfg.split)
        ds_sub = ds.select(range(min(judge_cfg.n_samples, len(ds))))
        eval_questions[domain] = [str(r["question"]) for r in ds_sub]
        eval_answers[domain] = [str(r["answer"]) for r in ds_sub]
        loss_texts.extend(format_as_sft_text(ds_sub, tokenizer))

    metadata_sink = JsonlSink(judge_logs_dir / "step_metadata.jsonl")
    judgement_sink = StepInjectingSink(JsonlSink(judge_logs_dir / "judgements.jsonl"))
    inference_sink = StepInjectingSink(JsonlSink(judge_logs_dir / "inference.jsonl"))

    evaluator = None
    if judge_cfg.enabled:
        evaluator = OneClickLLMJudgeScopingEval(
            train_domain=scope_domain,
            judge_model=judge_cfg.judge_model,
            n_samples=judge_cfg.n_samples,
            generation_kwargs={"do_sample": False, "max_new_tokens": mc.wrapper.max_seq_len},
        )

    # ── Callbacks ─────────────────────────────────────────────────────
    eval_callback = RecoveryEvalCallback(
        model=model,
        tokenizer=tokenizer,
        eval_every_steps=mc.wrapper.eval_every_steps,
        max_seq_len=mc.wrapper.max_seq_len,
        eval_batch_size=mc.wrapper.eval_batch_size,
        mode=mode,
        domain_questions=eval_questions,
        domain_answers=eval_answers,
        loss_texts=loss_texts,
        metadata_sink=metadata_sink,
        judgement_sink=judgement_sink,
        inference_sink=inference_sink,
        scores_path=judge_logs_dir / "scores.json",
        evaluator=evaluator,
        wandb_run=wandb_run,
    )
    callbacks = [eval_callback]
    if vanilla_scores:
        callbacks.append(
            VanillaFracEarlyStoppingCallback(
                eval_callback=eval_callback,
                vanilla_scores=vanilla_scores,
                mode=mode,
                scope_domain=scope_domain,
                eval_every_steps=mc.wrapper.eval_every_steps,
                elicitation_domain=elicitation_domain,
                pgd_min_relevance_frac=mc.wrapper.pgd_min_relevance_frac,
                pgd_min_fluency_frac=mc.wrapper.pgd_min_fluency_frac,
                pgd_min_similarity_frac=mc.wrapper.pgd_min_similarity_frac,
                elicit_min_score_frac=mc.wrapper.elicit_min_score_frac,
            )
        )

    # ── Train ─────────────────────────────────────────────────────────
    print(f"[pgd_or_elicit] Starting {mode} training...")
    trainer = PGDSFTTrainer(
        masks=masks,
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()

    # Save final metadata
    meta = {
        "mode": mode,
        "model_id": mc.model_id,
        "scope_domain": scope_domain,
        "sparsity": sparsity,
        "elicitation_domain": elicitation_domain,
    }
    with open(judge_logs_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    if wandb_run is not None:
        wandb_run.finish()

    # Completion marker — written last so the scheduler can trust it
    if isinstance(step, PGDStep):
        (checkpoint_dir / "COMPLETED").write_text(f"step_id={step.type}\n")
    (judge_logs_dir / "COMPLETED").write_text(f"step_id={step.type}\n")

    print(f"[pgd_or_elicit] Done: {mode} for {mc.model_id} / {scope_domain} / sp{sparsity}")


if __name__ == "__main__":
    main()
