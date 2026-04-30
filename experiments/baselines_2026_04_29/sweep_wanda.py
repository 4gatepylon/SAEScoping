"""Run Wanda pruning sweep on a model and report loss vs sparsity.

All settings live in a hierarchical pydantic-yaml config (see
sae_scoping.utils.sweep_config.SweepConfig). A YAML file passed via
--config supplies the bulk of the configuration; defaults are baked into
the pydantic schema. CLI flags exist only for the most-likely-to-change
overrides (--device, -s, --artifacts-dir, --enable-llm-judge,
--enable-wandb, --no-cache). Anything you'd otherwise want to tweak goes
in the YAML.

Per-run artifacts are written under
    $artifacts_root/outputs/{run_id}/
        metadata.json                # full resolved config + git sha + start time
        step_NNN/
            step_metadata.json       # sparsity, loss, zeros counts
            judgements.jsonl         # streamed by JsonlSink (LLM judge enabled)
            inference.jsonl          # streamed by JsonlSink (LLM judge enabled)
            scores.json              # final llm_judge/{domain}/{scope}/... dict

For the crash/durability/thread-safety guarantees of the streaming JSONL
files, see `JsonlSink` in sae_scoping.evaluation.utils.
"""

import json
from pathlib import Path
from typing import Optional

import click
import torch

from sae_scoping.datasets.qa_datasets import format_as_sft_text, load_qa_dataset
from sae_scoping.evaluation.loss import compute_loss, count_zeros
from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval
from sae_scoping.evaluation.utils import JsonlSink
from sae_scoping.training.saliency.validators import MaskSubsetValidator
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
from sae_scoping.utils.click_utils import parse_comma_separated_floats
from sae_scoping.utils.model_loading import load_model_and_tokenizer
from sae_scoping.utils.sweep_config import SweepConfig
from sae_scoping.utils.wandb_utils import resolve_wandb_settings


def _load_judge_domains(
    dataset_name: str,
    domains: list[str],
    split: str,
    n: int,
    seed: int = 42,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Load `n` (question, answer) rows per domain from `split`."""
    domain_questions: dict[str, list[str]] = {}
    domain_answers: dict[str, list[str]] = {}
    for d in domains:
        ds = load_qa_dataset(dataset_name, d, split=split, n=n, seed=seed)
        domain_questions[d] = list(ds["question"])
        domain_answers[d] = list(ds["answer"])
    return domain_questions, domain_answers


def _apply_cli_overrides(
    cfg: SweepConfig,
    *,
    model_id: Optional[str],
    n_calibration: Optional[int],
    n_eval: Optional[int],
    max_seq_len: Optional[int],
    batch_size: Optional[int],
    nn_linear_sparsity: Optional[str],
    device: Optional[str],
    artifacts_dir: Optional[str],
    no_cache: bool,
    enable_llm_judge: bool,
    enable_wandb: bool,
) -> None:
    """Mutate `cfg` in place, applying any non-None CLI overrides.

    Boolean flags (no_cache / enable_llm_judge / enable_wandb) are one-way:
    passing the flag forces the corresponding cfg field to True. They cannot
    flip a YAML-set True back to False — use the YAML file for that.
    """
    if model_id is not None:
        cfg.model_id = model_id
    if n_calibration is not None:
        cfg.calibration.n_calibration = n_calibration
    if n_eval is not None:
        cfg.sweep.n_eval = n_eval
    if max_seq_len is not None:
        cfg.calibration.max_seq_len = max_seq_len
    if batch_size is not None:
        cfg.calibration.batch_size = batch_size
    if nn_linear_sparsity is not None:
        cfg.sweep.nn_linear_sparsities = parse_comma_separated_floats(nn_linear_sparsity)
    if device is not None:
        cfg.operational.device = device
    if artifacts_dir is not None:
        cfg.operational.artifacts_dir = artifacts_dir
    if no_cache:
        cfg.operational.no_cache = True
    if enable_llm_judge:
        cfg.operational.llm_judge.enabled = True
    if enable_wandb:
        cfg.operational.wandb.enabled = True


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="YAML config (pydantic SweepConfig). CLI flags below override individual fields.",
)
# ── Common per-run overrides ────────────────────────────────────────────────
@click.option("--model-id", default=None, help="Override model_id (e.g. google/gemma-3-12b-it).")
@click.option("--n-calibration", type=int, default=None, help="Override calibration.n_calibration.")
@click.option("--n-eval", type=int, default=None, help="Override sweep.n_eval.")
@click.option("--max-seq-len", type=int, default=None, help="Override calibration.max_seq_len.")
@click.option("--batch-size", type=int, default=None, help="Override calibration.batch_size.")
@click.option(
    "--nn-linear-sparsity",
    "-s",
    default=None,
    help="Override sweep.nn_linear_sparsities. Comma-separated, e.g. -s 0.2,0.4,0.6.",
)
@click.option("--device", default=None, help="Override operational.device.")
@click.option("--artifacts-dir", default=None, help="Override operational.artifacts_dir.")
@click.option("--no-cache", is_flag=True, default=False, help="Force operational.no_cache to True (cannot disable from CLI).")
@click.option("--enable-llm-judge", is_flag=True, default=False, help="Force operational.llm_judge.enabled to True.")
@click.option("--enable-wandb", is_flag=True, default=False, help="Force operational.wandb.enabled to True.")
def main(
    config: Optional[str],
    model_id: Optional[str],
    n_calibration: Optional[int],
    n_eval: Optional[int],
    max_seq_len: Optional[int],
    batch_size: Optional[int],
    nn_linear_sparsity: Optional[str],
    device: Optional[str],
    artifacts_dir: Optional[str],
    no_cache: bool,
    enable_llm_judge: bool,
    enable_wandb: bool,
) -> None:
    """Run Wanda pruning sweep: compute saliency once, then evaluate at each sparsity from low to high."""
    # =========================================================================
    # STAGE 1 — SETUP
    #   Load + override config, create the run dir, write run-level metadata,
    #   init W&B (when enabled). Nothing model-/data-loading yet.
    # =========================================================================

    # ── Config: load YAML (or use schema defaults) and apply CLI overrides ──
    cfg = SweepConfig.from_yaml(config) if config else SweepConfig()
    _apply_cli_overrides(
        cfg,
        model_id=model_id,
        n_calibration=n_calibration,
        n_eval=n_eval,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        nn_linear_sparsity=nn_linear_sparsity,
        device=device,
        artifacts_dir=artifacts_dir,
        no_cache=no_cache,
        enable_llm_judge=enable_llm_judge,
        enable_wandb=enable_wandb,
    )

    sparsities = cfg.sweep.nn_linear_sparsities
    print(f"Sweep nn.Linear sparsities: {[f'{s:.1%}' for s in sparsities]}")

    # ── Run-level artifacts setup ───────────────────────────────────────────
    artifacts_root = resolve_artifacts_root(cfg.operational.artifacts_dir)
    run_id = make_run_id()
    run_dir = make_run_dir(artifacts_root, run_id)
    print(f"[artifacts] run_dir: {run_dir}")

    judge_domains_list: list[str] = (
        cfg.operational.llm_judge.domains
        if cfg.operational.llm_judge.domains
        else [cfg.dataset_subset]
    )

    run_metadata = build_run_metadata(
        cfg.model_dump(),
        run_id=run_id,
        script=Path(__file__),
        artifacts_dir_resolved=str(artifacts_root),
        judge_domains_resolved=judge_domains_list if cfg.operational.llm_judge.enabled else None,
    )
    (run_dir / "metadata.json").write_text(json.dumps(run_metadata, indent=2, default=str), encoding="utf-8")

    # ── W&B init ────────────────────────────────────────────────────────────
    # `nn_linear_sparsity` is declared as the X axis so the W&B UI plots loss
    # and per-judge scores against sparsity (not against step idx).
    wandb_run = None
    if cfg.operational.wandb.enabled:
        import wandb  # local import — saves the cost when W&B is off

        wandb_run = wandb.init(
            **resolve_wandb_settings(
                project=cfg.operational.wandb.project,
                entity=cfg.operational.wandb.entity,
                mode=cfg.operational.wandb.mode,
                name=cfg.operational.wandb.name or run_id,
                tags=cfg.operational.wandb.tags,
            ),
            config=run_metadata,
        )
        wandb.define_metric("nn_linear_sparsity")
        wandb.define_metric("model_sparsity", step_metric="nn_linear_sparsity")
        wandb.define_metric("loss", step_metric="nn_linear_sparsity")
        wandb.define_metric("loss_delta_vs_baseline", step_metric="nn_linear_sparsity")
        wandb.define_metric("llm_judge/*", step_metric="nn_linear_sparsity")
        print(f"[wandb] initialized run: {wandb_run.name} ({wandb_run.url})")

    # =========================================================================
    # STAGE 2 — CALIBRATION
    #   Load the model + tokenizer + dataset, optionally pre-load LLM-judge
    #   questions, compute the baseline (pre-pruning) loss, then compute (or
    #   load from cache) the Wanda saliency map from the calibration split.
    # =========================================================================

    print(f"Loading tokenizer and model: {cfg.model_id}")
    model, tokenizer = load_model_and_tokenizer(cfg.model_id, device=cfg.operational.device)

    print(f"Loading dataset: {cfg.dataset_name}/{cfg.dataset_subset}")
    n_total = cfg.calibration.n_calibration + cfg.sweep.n_eval
    ds = load_qa_dataset(cfg.dataset_name, cfg.dataset_subset, n=n_total, seed=42)
    all_texts = format_as_sft_text(ds, tokenizer)
    calib_texts = all_texts[: cfg.calibration.n_calibration]
    eval_texts = all_texts[cfg.calibration.n_calibration :]

    # ── Pre-load LLM-judge data (once; reused per sparsity step) ─────────────
    evaluator: Optional[OneClickLLMJudgeScopingEval] = None
    judge_questions: Optional[dict[str, list[str]]] = None
    judge_answers: Optional[dict[str, list[str]]] = None
    if cfg.operational.llm_judge.enabled:
        print(f"[llm-judge] Loading {cfg.operational.llm_judge.split} split for domains: {judge_domains_list}")
        judge_questions, judge_answers = _load_judge_domains(
            dataset_name=cfg.dataset_name,
            domains=judge_domains_list,
            split=cfg.operational.llm_judge.split,
            n=cfg.operational.llm_judge.n_samples,
        )
        evaluator = OneClickLLMJudgeScopingEval(
            n_samples=cfg.operational.llm_judge.n_samples,
            judge_model=cfg.operational.llm_judge.judge_model,
            train_domain=cfg.dataset_subset,
        )

    print("\n=== Baseline (pre-pruning) ===")
    baseline_loss = compute_loss(
        model, tokenizer, eval_texts,
        max_seq_len=cfg.calibration.max_seq_len,
        batch_size=cfg.calibration.batch_size,
    )
    zeros_before, total_params = count_zeros(model)
    print(f"  Loss:            {baseline_loss:.4f}")
    print(f"  Model sparsity:  {zeros_before}/{total_params} ({zeros_before / total_params:.2%})")

    cache_dir = Path(cfg.operational.cache_dir) if cfg.operational.cache_dir else artifacts_root / "cache"
    saliency_file = cache_path(cache_dir, cfg.model_id, cfg.dataset_subset, "wanda_saliency.safetensors")
    saliency_map = load_or_compute_safetensors(
        path=saliency_file,
        compute_fn=lambda: compute_wanda_saliency(
            model, tokenizer, calib_texts,
            max_seq_len=cfg.calibration.max_seq_len,
            batch_size=cfg.calibration.batch_size,
        ),
        no_cache=cfg.operational.no_cache,
        label="Wanda saliency",
    )

    linear_total = sum(t.numel() for t in saliency_map.values())

    # =========================================================================
    # STAGE 3 — PRUNE + EVAL SWEEP
    #   For each sparsity in cfg.sweep.nn_linear_sparsities:
    #     1. compute the Wanda mask, apply to the model, measure loss,
    #     2. write per-step artifacts (step_metadata.json),
    #     3. (optional) run the LLM judge with both JsonlSinks open,
    #     4. (PLACEHOLDER) PGD recovery — stubbed; see TODO inside the loop,
    #     5. (optional) log to W&B with `nn_linear_sparsity` as the X axis.
    # =========================================================================

    validator = MaskSubsetValidator(enabled=not cfg.operational.low_memory)
    results = []
    for i, sparsity in enumerate(sparsities):
        masks = compute_wanda_masks(saliency_map, sparsity)
        validator.validate_and_update(masks)
        apply_masks_to_model(model, masks)

        pruned_loss = compute_loss(
            model, tokenizer, eval_texts,
            max_seq_len=cfg.calibration.max_seq_len,
            batch_size=cfg.calibration.batch_size,
        )
        zeros_after, _ = count_zeros(model)
        linear_zeros = sum(int((~m).sum().item()) for m in masks.values())
        delta = pruned_loss - baseline_loss
        results.append((sparsity, pruned_loss, delta, zeros_after, linear_zeros))

        # ── Per-step artifacts ──────────────────────────────────────────────
        # TODO(adrianoh) have some way of modularizing this out
        step_dir = make_step_dir(run_dir, i)
        step_metadata = {
            "step_idx": i,
            "nn_linear_sparsity": sparsity,
            "loss": pruned_loss,
            "loss_delta_vs_baseline": delta,
            "linear_zeros": linear_zeros,
            "linear_total": linear_total,
            "linear_sparsity": linear_zeros / linear_total,
            "model_zeros": zeros_after,
            "model_total": total_params,
            "model_sparsity": zeros_after / total_params,
            "baseline_loss": baseline_loss,
        }
        (step_dir / "step_metadata.json").write_text(
            json.dumps(step_metadata, indent=2, default=str), encoding="utf-8"
        )

        print(f"\n=== nn.Linear sparsity {sparsity:.1%} (step {i}) ===")
        print(f"  Loss:                 {pruned_loss:.4f} (delta: {delta:+.4f})")
        print(f"  nn.Linear sparsity:   {linear_zeros}/{linear_total} ({linear_zeros / linear_total:.2%})")
        print(f"  Whole-model sparsity: {zeros_after}/{total_params} ({zeros_after / total_params:.2%})")
        print(f"  Step dir:             {step_dir}")

        # ── LLM-judge per step ─────────────────────────────────────────────
        if cfg.operational.llm_judge.enabled:
            # TODO(adrianoh) have some way of modularizing this out
            assert evaluator is not None and judge_questions is not None and judge_answers is not None
            with (
                JsonlSink(step_dir / "judgements.jsonl") as j_sink,
                JsonlSink(step_dir / "inference.jsonl") as i_sink,
            ):
                scores, _df_json = evaluator.evaluate(
                    model,
                    tokenizer,
                    domain_questions=judge_questions,
                    domain_answers=judge_answers,
                    judgement_sink=j_sink,
                    inference_sink=i_sink,
                )
            (step_dir / "scores.json").write_text(
                json.dumps(scores, indent=2, default=str), encoding="utf-8"
            )
            for k, v in sorted(scores.items()):
                print(f"  llm_judge: {k:<60} {v:.3f}")

        # ── PGD recovery (stub; not implemented) ───────────────────────────
        # TODO(adrianoh) PGD recovery training will be inserted here in a
        # later commit. Plan: after Wanda pruning at `sparsity`, optionally
        # run PGDSFTTrainer to recover loss while keeping zeroed weights
        # zero. Should reuse the JsonlSinks (judgement + inference) and the
        # W&B run already open in this loop, via a custom TrainerCallback
        # that runs the LLM judge mid-training. The PGDConfig sub-config
        # (sae_scoping/utils/sweep_config.py) already has the parameter
        # surface; this is just where the call goes.

        # ── W&B per-step log ───────────────────────────────────────────────
        # nn_linear_sparsity is the X axis (declared via define_metric above);
        # model_sparsity is logged alongside as a separate Y, NOT as the X
        # axis, since two values per step rarely make sense as competing X's.
        if wandb_run is not None:
            log_dict: dict[str, float] = {
                "nn_linear_sparsity": sparsity,
                "model_sparsity": zeros_after / total_params,
                "loss": pruned_loss,
                "loss_delta_vs_baseline": delta,
            }
            if cfg.operational.llm_judge.enabled:
                log_dict.update({k: float(v) for k, v in scores.items()})
            wandb_run.log(log_dict)

    # =========================================================================
    # STAGE 4 — TEARDOWN
    #   Print the per-sparsity summary table, finish the W&B run.
    # =========================================================================

    print(f"\n{'=' * 70}")
    print(f"Summary: {cfg.model_id} on {cfg.dataset_subset}  (run_id={run_id})")
    print(f"{'nn.Linear %':>12} {'Loss':>10} {'Delta':>10} {'Linear %':>10} {'Model %':>10}")
    print(f"{'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for sparsity, loss, delta, zeros, lin_zeros in results:
        print(f"{sparsity:>12.1%} {loss:>10.4f} {delta:>+10.4f} {lin_zeros / linear_total:>10.2%} {zeros / total_params:>10.2%}")
    print(f"\nArtifacts: {run_dir}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
