# TODO(Claude) this is dated---you will want to refactor.
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import click
import torch
from datasets import Dataset
from trl import SFTConfig

from sae_scoping.datasets.qa_datasets import (
    format_as_sft_text,
    load_nonoverlapping_splits,
    load_qa_dataset,
)
from sae_scoping.evaluation.loss import compute_loss, count_zeros
from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval
from sae_scoping.evaluation.utils import JsonlSink
from sae_scoping.training.pgd_trainer import (
    PGDSFTTrainer,
    filter_masks_by_min_layer_idx,
    freeze_early_side_params,
)
from sae_scoping.training.saliency.wanda import (
    apply_masks_to_model,
    compute_wanda_masks,
    compute_wanda_saliency,
)
from sae_scoping.utils.artifacts import (
    build_run_metadata,
    make_run_dir,
    make_run_id,
    make_step_dir,
    resolve_artifacts_root,
)
from sae_scoping.utils.cache import cache_path, load_or_compute_safetensors
from sae_scoping.utils.model_loading import load_model_and_tokenizer
from sae_scoping.utils.sweep_config import SweepConfig
from sae_scoping.utils.wandb_utils import resolve_wandb_settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_pgd_recovery(
    *,
    model: Any,
    tokenizer: Any,
    masks: dict[str, torch.Tensor],
    train_texts: list[str],
    eval_texts: list[str],
    sparsity: float,
    baseline_loss: float,
    cfg: SweepConfig,
    recovery_dir: Path,
    evaluator: Optional[OneClickLLMJudgeScopingEval],
    wandb_run: Optional[Any],
) -> dict[str, float]:
    """Run PGD recovery and stream metrics to `recovery_dir/`.

    Returns the last LLM-judge scores dict (empty when judge is off).
    The evaluator already carries its own domain questions/answers from
    __init__, so no need to pass them separately.
    """
    from experiments.baselines_2026_04_29.sweep_wanda import RecoveryEvalCallback

    pgd_cfg = cfg.pgd
    sft_dataset = Dataset.from_dict({"text": train_texts})
    use_cuda = cfg.operational.device.startswith("cuda")

    sft_config = SFTConfig(
        output_dir=str(recovery_dir / "trl_output"),
        learning_rate=pgd_cfg.learning_rate,
        num_train_epochs=pgd_cfg.num_train_epochs,
        max_steps=pgd_cfg.max_steps,
        per_device_train_batch_size=pgd_cfg.train_batch_size,
        gradient_accumulation_steps=pgd_cfg.gradient_accumulation_steps,
        warmup_ratio=pgd_cfg.warmup_ratio,
        logging_steps=pgd_cfg.logging_steps,
        max_length=cfg.calibration.max_seq_len,
        bf16=use_cuda,
        report_to=pgd_cfg.report_to,
        save_strategy=pgd_cfg.save_strategy,
        save_steps=pgd_cfg.save_steps,
        save_total_limit=pgd_cfg.save_total_limit,
        optim=pgd_cfg.optim,
        gradient_checkpointing=pgd_cfg.gradient_checkpointing,
        use_cpu=not use_cuda,
    )

    with (
        JsonlSink(recovery_dir / "step_metadata.jsonl") as step_sink,
        JsonlSink(recovery_dir / "judgements.jsonl") as j_sink,
        JsonlSink(recovery_dir / "inference.jsonl") as i_sink,
    ):
        callback = RecoveryEvalCallback(
            sparsity=sparsity,
            eval_every_steps=pgd_cfg.eval_every_steps,
            eval_texts=eval_texts,
            max_seq_len=cfg.calibration.max_seq_len,
            batch_size=cfg.calibration.batch_size,
            tokenizer=tokenizer,
            baseline_loss=baseline_loss,
            step_metadata_sink=step_sink,
            evaluator=evaluator,
            domain_questions=evaluator._default_domain_questions if evaluator is not None else None,
            domain_answers=evaluator._default_domain_answers if evaluator is not None else None,
            judgement_sink=j_sink if evaluator is not None else None,
            inference_sink=i_sink if evaluator is not None else None,
            wandb_run=wandb_run,
            min_layer_idx=pgd_cfg.min_layer_idx,
        )

        # TODO(claude) we should just ** the YAML arguments section for PGD Trainer. The exception is that we will write in the masks
        trainer = PGDSFTTrainer(
            masks=masks,
            validate_sparsity=pgd_cfg.validate_sparsity,
            model=model,
            args=sft_config,
            train_dataset=sft_dataset,
            processing_class=tokenizer,
            callbacks=[callback],
        )
        trainer.train()
        final_scores = callback.last_scores

        if pgd_cfg.save_final_model:
            final_model_dir = recovery_dir / "final_model"
            print(f"[pgd] saving final model to {final_model_dir}", flush=True)
            trainer.save_model(str(final_model_dir))

    if final_scores:
        (recovery_dir / "scores.json").write_text(
            json.dumps(final_scores, indent=2, default=str),
            encoding="utf-8",
        )
    return final_scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--config", type=click.Path(exists=True), required=True, help="YAML config (SweepConfig schema).")
@click.option("--model-id", default=None, help="Override model_id.")
@click.option("--dataset-subset", default=None, help="Override dataset_subset.")
@click.option("--batch-size", type=int, default=None, help="Override calibration.batch_size.")
@click.option("--device", default=None, help="Override operational.device.")
@click.option("--nn-linear-sparsity", "-s", default=None, help="Comma-separated sparsities, e.g. 0.3,0.5,0.7.")
@click.option("--enable-llm-judge", is_flag=True, default=False, help="Force LLM judge on.")
@click.option("--enable-wandb", is_flag=True, default=False, help="Force W&B on.")
@click.option("--enable-pgd", is_flag=True, default=False, help="Force PGD recovery on.")
@click.option("--pgd-train-batch-size", type=int, default=None, help="Override pgd.train_batch_size.")
@click.option("--gradient-accumulation-steps", "pgd_grad_accum", type=int, default=None, help="Override pgd.gradient_accumulation_steps.")
@click.option("--early-stop-delta", type=float, default=None, help="Stop sweeping if loss delta exceeds this threshold.")
def main(
    config: str,
    model_id: Optional[str],
    dataset_subset: Optional[str],
    batch_size: Optional[int],
    device: Optional[str],
    nn_linear_sparsity: Optional[str],
    enable_llm_judge: bool,
    enable_wandb: bool,
    enable_pgd: bool,
    pgd_train_batch_size: Optional[int],
    pgd_grad_accum: Optional[int],
    early_stop_delta: Optional[float],
) -> None:
    """Wanda prune + (optional) PGD recovery sweep on StemQA."""

    # ── 1. Config ────────────────────────────────────────────────────────
    cfg = SweepConfig.from_yaml(config)
    if model_id is not None:
        cfg.model_id = model_id
    if dataset_subset is not None:
        cfg.dataset_subset = dataset_subset
    if batch_size is not None:
        cfg.calibration.batch_size = batch_size
    if device is not None:
        cfg.operational.device = device
    if nn_linear_sparsity is not None:
        cfg.sweep.nn_linear_sparsities = [float(x) for x in nn_linear_sparsity.split(",")]
    if enable_llm_judge:
        cfg.operational.llm_judge.enabled = True
    if enable_wandb:
        cfg.operational.wandb.enabled = True
    if enable_pgd:
        cfg.pgd.enabled = True
    if pgd_train_batch_size is not None:
        cfg.pgd.train_batch_size = pgd_train_batch_size
    if pgd_grad_accum is not None:
        cfg.pgd.gradient_accumulation_steps = pgd_grad_accum

    sparsities = cfg.sweep.nn_linear_sparsities
    print(f"Sweep sparsities: {[f'{s:.1%}' for s in sparsities]}")
    if cfg.pgd.enabled:
        print(f"[pgd] enabled: lr={cfg.pgd.learning_rate}, max_steps={cfg.pgd.max_steps}")
    if early_stop_delta is not None:
        print(f"[early-stop] will stop if loss delta > {early_stop_delta:.4f}")

    # ── 2. Artifacts + W&B ───────────────────────────────────────────────
    artifacts_root = resolve_artifacts_root(cfg.operational.artifacts_dir)
    run_id = make_run_id()
    run_dir = make_run_dir(artifacts_root, run_id)
    print(f"[artifacts] {run_dir}")

    judge_domains = cfg.operational.llm_judge.domains or [cfg.dataset_subset]
    run_metadata = build_run_metadata(
        cfg.model_dump(),
        run_id=run_id,
        script=Path(__file__),
        artifacts_dir_resolved=str(artifacts_root),
        judge_domains_resolved=judge_domains if cfg.operational.llm_judge.enabled else None,
        early_stop_delta=early_stop_delta,
    )
    (run_dir / "metadata.json").write_text(
        json.dumps(run_metadata, indent=2, default=str),
        encoding="utf-8",
    )

    wandb_run = None
    if cfg.operational.wandb.enabled:
        import wandb

        wandb_run = wandb.init(
            **resolve_wandb_settings(
                project=cfg.operational.wandb.project,
                entity=cfg.operational.wandb.entity,
                mode=cfg.operational.wandb.mode,
                name=cfg.operational.wandb.name or os.environ.get("WANDB_NAME") or run_id,
                tags=cfg.operational.wandb.tags,
            ),
            config=run_metadata,
        )
        wandb.define_metric("nn_linear_sparsity")
        wandb.define_metric("sweep/*", step_metric="nn_linear_sparsity")
        wandb.define_metric("recovery/train_step")
        wandb.define_metric("recovery/*", step_metric="recovery/train_step")
        print(f"[wandb] {wandb_run.name} ({wandb_run.url})")

    # ── 3. Load model + data ─────────────────────────────────────────────
    print(f"Loading model: {cfg.model_id}")
    model, tokenizer = load_model_and_tokenizer(cfg.model_id, device=cfg.operational.device)

    if cfg.pgd.enabled:
        calib_texts, train_texts, eval_texts = load_nonoverlapping_splits(
            tokenizer,
            dataset_name=cfg.dataset_name,
            subset=cfg.dataset_subset,
            n_calibration=cfg.calibration.n_calibration,
            n_train=cfg.pgd.n_train,
            n_test=cfg.sweep.n_eval,
            seed=42,
        )
    else:
        n_total = cfg.calibration.n_calibration + cfg.sweep.n_eval
        ds = load_qa_dataset(cfg.dataset_name, cfg.dataset_subset, n=n_total, seed=42)
        all_texts = format_as_sft_text(ds, tokenizer)
        calib_texts = all_texts[: cfg.calibration.n_calibration]
        eval_texts = all_texts[cfg.calibration.n_calibration :]
        train_texts = []

    # ── 4. LLM judge setup ───────────────────────────────────────────────
    evaluator: Optional[OneClickLLMJudgeScopingEval] = None
    if cfg.operational.llm_judge.enabled:
        evaluator = OneClickLLMJudgeScopingEval(
            n_samples=cfg.operational.llm_judge.n_samples,
            judge_model=cfg.operational.llm_judge.judge_model,
            train_domain=cfg.dataset_subset,
            domain_datasets={
                d: load_qa_dataset(
                    cfg.dataset_name,
                    d,
                    split=cfg.operational.llm_judge.split,
                    n=cfg.operational.llm_judge.n_samples,
                )
                for d in judge_domains
            },
        )

    # ── 5. Baseline ──────────────────────────────────────────────────────
    print("\n=== Baseline ===")
    baseline_loss = compute_loss(
        model,
        tokenizer,
        eval_texts,
        max_seq_len=cfg.calibration.max_seq_len,
        batch_size=cfg.calibration.batch_size,
    )
    zeros_before, total_params = count_zeros(model)
    print(f"  Loss: {baseline_loss:.4f}  Sparsity: {zeros_before / total_params:.2%}")

    baseline = {"loss": baseline_loss, "model_sparsity": zeros_before / total_params}
    (run_dir / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.log({f"sweep/baseline/{k}": float(v) for k, v in baseline.items()})

    # ── 6. Wanda saliency (cached) ───────────────────────────────────────
    cache_dir = Path(cfg.operational.cache_dir) if cfg.operational.cache_dir else artifacts_root / "cache"
    saliency_file = cache_path(cache_dir, cfg.model_id, cfg.dataset_subset, "wanda_saliency.safetensors")
    saliency_map = load_or_compute_safetensors(
        path=saliency_file,
        compute_fn=lambda: compute_wanda_saliency(
            model,
            tokenizer,
            calib_texts,
            max_seq_len=cfg.calibration.max_seq_len,
            batch_size=cfg.calibration.batch_size,
        ),
        no_cache=cfg.operational.no_cache,
        label="Wanda saliency",
    )
    linear_total = sum(t.numel() for t in saliency_map.values())

    # ── 7. Sweep ─────────────────────────────────────────────────────────
    results: list[tuple[float, float, float]] = []
    for i, sparsity in enumerate(sparsities):
        masks = compute_wanda_masks(saliency_map, sparsity)
        apply_masks_to_model(model, masks)

        pruned_loss = compute_loss(
            model,
            tokenizer,
            eval_texts,
            max_seq_len=cfg.calibration.max_seq_len,
            batch_size=cfg.calibration.batch_size,
        )
        zeros_after, _ = count_zeros(model)
        delta = pruned_loss - baseline_loss
        results.append((sparsity, pruned_loss, delta))

        step_dir = make_step_dir(run_dir, i)
        sweep_dir = step_dir / "sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        linear_zeros = sum(int((~m).sum().item()) for m in masks.values())
        step_meta = {
            "step_idx": i,
            "nn_linear_sparsity": sparsity,
            "loss": pruned_loss,
            "loss_delta_vs_baseline": delta,
            "linear_sparsity": linear_zeros / linear_total,
            "model_sparsity": zeros_after / total_params,
        }
        (sweep_dir / "step_metadata.json").write_text(json.dumps(step_meta, indent=2), encoding="utf-8")

        print(f"\n=== Sparsity {sparsity:.1%} (step {i}) ===")
        print(f"  Loss: {pruned_loss:.4f} (delta: {delta:+.4f})")
        print(f"  Linear sparsity: {linear_zeros / linear_total:.2%}  Model sparsity: {zeros_after / total_params:.2%}")

        sweep_scores: dict[str, float] = {}
        if evaluator is not None:
            with (
                JsonlSink(sweep_dir / "judgements.jsonl") as j_sink,
                JsonlSink(sweep_dir / "inference.jsonl") as i_sink,
            ):
                sweep_scores, _ = evaluator.evaluate(
                    model,
                    tokenizer,
                    judgement_sink=j_sink,
                    inference_sink=i_sink,
                )
            (sweep_dir / "scores.json").write_text(json.dumps(sweep_scores, indent=2), encoding="utf-8")
            for k, v in sorted(sweep_scores.items()):
                print(f"  {k}: {v:.3f}")

        if wandb_run is not None:
            log_dict: dict[str, float] = {
                "nn_linear_sparsity": sparsity,
                "sweep/loss": pruned_loss,
                "sweep/loss_delta_vs_baseline": delta,
                "sweep/model_sparsity": zeros_after / total_params,
            }
            for k, v in sweep_scores.items():
                log_dict[f"sweep/{k}"] = float(v)
            wandb_run.log(log_dict)

        # ── PGD recovery ─────────────────────────────────────────────────
        if cfg.pgd.enabled:
            recovery_dir = step_dir / "recovery"
            recovery_dir.mkdir(parents=True, exist_ok=True)
            pgd_masks = masks
            if cfg.pgd.min_layer_idx is not None:
                pgd_masks = filter_masks_by_min_layer_idx(masks, cfg.pgd.min_layer_idx)
                if not pgd_masks:
                    raise click.ClickException(f"pgd.min_layer_idx={cfg.pgd.min_layer_idx} filtered out every mask.")
                freeze_early_side_params(model, cfg.pgd.min_layer_idx)
            print(f"\n  [pgd] Starting recovery at sparsity {sparsity:.1%}")
            _run_pgd_recovery(
                model=model,
                tokenizer=tokenizer,
                masks=pgd_masks,
                train_texts=train_texts,
                eval_texts=eval_texts,
                sparsity=sparsity,
                baseline_loss=baseline_loss,
                cfg=cfg,
                recovery_dir=recovery_dir,
                evaluator=evaluator,
                wandb_run=wandb_run,
            )

        # ── Early stop ───────────────────────────────────────────────────
        if early_stop_delta is not None and delta > early_stop_delta:
            print(f"\n[early-stop] delta {delta:.4f} > threshold {early_stop_delta:.4f}, stopping.")
            break

    # ── 8. Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Summary: {cfg.model_id} on {cfg.dataset_subset} (run_id={run_id})")
    print(f"{'Sparsity':>10} {'Loss':>10} {'Delta':>10}")
    for sparsity, loss, delta in results:
        print(f"{sparsity:>10.1%} {loss:>10.4f} {delta:>+10.4f}")
    print(f"\nArtifacts: {run_dir}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
