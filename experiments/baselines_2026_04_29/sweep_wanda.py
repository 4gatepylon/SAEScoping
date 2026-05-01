"""Run Wanda pruning sweep on a model and (optionally) recover via PGD per sparsity.

All settings live in a hierarchical pydantic-yaml config (see
sae_scoping.utils.sweep_config.SweepConfig). A YAML file passed via
--config supplies the bulk of the configuration; defaults are baked into
the pydantic schema. CLI flags exist only for the most-likely-to-change
overrides. Anything you'd otherwise want to tweak goes in the YAML.

The runner has three phases (see ./NAMING.md for the metric/artifact
namespace spec):

    sweep      — one value per sparsity (Wanda prune + per-sparsity loss
                 + LLM judge). X axis: nn_linear_sparsity.
    recovery   — per-train-step value per sparsity (PGDSFTTrainer running
                 SFT with a zero-projection after each optimizer step).
                 X axis: recovery/train_step. Off unless cfg.pgd.enabled.
    elicit     — placeholder; not implemented in this commit.

Per-run artifacts are written under
    $artifacts_root/outputs/{run_id}/
        metadata.json
        baseline.json                    # one-shot pre-pruning metrics
        step_NNN/                        # one per sparsity in cfg.sweep.nn_linear_sparsities
            sweep/
                step_metadata.json       # sparsity, loss, model_sparsity
                judgements.jsonl         # streamed by JsonlSink
                inference.jsonl
                scores.json              # final llm_judge/{domain}/{scope}/... dict
            recovery/                    # populated only when cfg.pgd.enabled
                step_metadata.jsonl      # one row per PGD train-step check-in
                judgements.jsonl
                inference.jsonl
                scores.json              # post-recovery aggregated scores

For the crash/durability/thread-safety guarantees of the streaming JSONL
files, see `JsonlSink` in sae_scoping.evaluation.utils.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

# Regex used by RecoveryEvalCallback to compute late-layer sparsity. Mirrors
# pgd_trainer._LAYER_INDEX_RE so they cannot drift in isolation.
_LAYER_INDEX_RE_RUNNER = re.compile(r"\.layers\.(\d+)\.")

import click
import torch
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTConfig

from sae_scoping.datasets.qa_datasets import format_as_sft_text, load_nonoverlapping_splits, load_qa_dataset
from sae_scoping.evaluation.loss import compute_loss, count_zeros
from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval
from sae_scoping.evaluation.utils import JsonlSink, Sink
from sae_scoping.training.pgd_trainer import PGDSFTTrainer, filter_masks_by_min_layer_idx, freeze_early_side_params
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


# =========================================================================
# Recovery (PGD) callback
# =========================================================================


class RecoveryEvalCallback(TrainerCallback):
    """Per-N-step eval callback during PGD recovery training.

    On every `eval_every_steps` step (and again at training end):
      1. Compute eval loss + model sparsity on the held-out split.
      2. Optionally run the LLM judge against the current model state.
      3. Append a JSONL row to `step_metadata_sink`
         (recovery/step_metadata.jsonl) with the metrics + train_step.
      4. Stream judge rows through `judgement_sink` / `inference_sink`
         (recovery/judgements.jsonl / inference.jsonl).
      5. Log to W&B under `recovery/sparsity=<s>/...` keyed alongside
         `recovery/train_step` so the UI can plot loss / per-judge scores
         against training step.

    Held in sweep_wanda.py for now — coupled to this runner's W&B + sink
    setup. If a second runner ever needs the same shape, factor it out to
    `sae_scoping/training/utils/callbacks/`.
    """

    def __init__(
        self,
        *,
        sparsity: float,
        eval_every_steps: int,
        eval_texts: list[str],
        max_seq_len: int,
        batch_size: int,
        tokenizer: Any,
        baseline_loss: float,
        step_metadata_sink: Sink,
        evaluator: Optional[OneClickLLMJudgeScopingEval] = None,
        domain_questions: Optional[dict[str, list[str]]] = None,
        domain_answers: Optional[dict[str, list[str]]] = None,
        judgement_sink: Optional[Sink] = None,
        inference_sink: Optional[Sink] = None,
        wandb_run: Optional[Any] = None,
        min_layer_idx: Optional[int] = None,
    ) -> None:
        self.sparsity = sparsity
        self.eval_every_steps = eval_every_steps
        self.eval_texts = eval_texts
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.baseline_loss = baseline_loss
        self.step_metadata_sink = step_metadata_sink
        self.evaluator = evaluator
        self.domain_questions = domain_questions
        self.domain_answers = domain_answers
        self.judgement_sink = judgement_sink
        self.inference_sink = inference_sink
        self.wandb_run = wandb_run
        self.min_layer_idx = min_layer_idx
        self._wb_prefix = f"recovery/sparsity={sparsity}/"
        # Last-seen scores so on_train_end can write scores.json.
        self.last_scores: dict[str, float] = {}

    def _do_eval(self, model: Any, train_step: int) -> None:
        loss = compute_loss(
            model,
            self.tokenizer,
            self.eval_texts,
            max_seq_len=self.max_seq_len,
            batch_size=self.batch_size,
        )
        zeros, total = count_zeros(model)
        row: dict[str, Any] = {
            "train_step": train_step,
            "nn_linear_sparsity": self.sparsity,
            "loss": loss,
            "loss_delta_vs_baseline": loss - self.baseline_loss,
            "model_sparsity": zeros / total,
            "model_zeros": zeros,
            "model_total": total,
        }
        # When PGD is layer-restricted, also report sparsity over the *late*
        # layers that PGD actually enforces. Without this metric the
        # whole-model sparsity number is misleading because early-layer
        # pruned weights drift back during recovery (or stay zero only
        # because we additionally froze them — either way, the protected
        # late-layer sparsity is what the run is really claiming).
        if self.min_layer_idx is not None:
            late_zeros = 0
            late_total = 0
            for name, p in model.named_parameters():
                m = _LAYER_INDEX_RE_RUNNER.search(name)
                if m is None or int(m.group(1)) <= self.min_layer_idx:
                    continue
                late_zeros += int((p.data == 0).sum().item())
                late_total += p.numel()
            if late_total > 0:
                row["model_sparsity_late_layers"] = late_zeros / late_total
                row["model_zeros_late_layers"] = late_zeros
                row["model_total_late_layers"] = late_total

        scores: dict[str, float] = {}
        if self.evaluator is not None and self.domain_questions is not None:
            scores, _df_json = self.evaluator.evaluate(
                model,
                self.tokenizer,
                domain_questions=self.domain_questions,
                domain_answers=self.domain_answers,
                judgement_sink=self.judgement_sink,
                inference_sink=self.inference_sink,
            )
            row["llm_judge"] = scores
            self.last_scores = scores

        self.step_metadata_sink(row)

        if self.wandb_run is not None:
            log_payload: dict[str, float] = {
                "recovery/train_step": train_step,
                f"{self._wb_prefix}loss": loss,
                f"{self._wb_prefix}loss_delta_vs_baseline": loss - self.baseline_loss,
                f"{self._wb_prefix}model_sparsity": zeros / total,
            }
            if "model_sparsity_late_layers" in row:
                log_payload[f"{self._wb_prefix}model_sparsity_late_layers"] = row["model_sparsity_late_layers"]
            for k, v in scores.items():
                log_payload[f"{self._wb_prefix}{k}"] = float(v)
            self.wandb_run.log(log_payload)

        delta = loss - self.baseline_loss
        print(f"  [recovery sparsity={self.sparsity:.1%} step {train_step}] loss={loss:.4f} (delta={delta:+.4f})")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_every_steps != 0:
            return
        self._do_eval(model, state.global_step)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        # Always emit a final-step row so the curve has its endpoint.
        self._do_eval(model, state.global_step)


# =========================================================================
# Recovery driver
# =========================================================================


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
    judge_questions: Optional[dict[str, list[str]]],
    judge_answers: Optional[dict[str, list[str]]],
    wandb_run: Optional[Any],
) -> dict[str, float]:
    """Run PGDSFTTrainer at the current sparsity and write recovery/ artifacts.

    Returns the most recent LLM-judge `scores` dict (empty when LLM judge is
    off). Loss curve and any judge rows are streamed through JsonlSinks
    inside `recovery_dir/`.
    """
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
        save_strategy="no",
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
            domain_questions=judge_questions,
            domain_answers=judge_answers,
            judgement_sink=j_sink if evaluator is not None else None,
            inference_sink=i_sink if evaluator is not None else None,
            wandb_run=wandb_run,
            min_layer_idx=cfg.pgd.min_layer_idx,
        )

        trainer = PGDSFTTrainer(
            masks=masks,
            validate_sparsity=pgd_cfg.validate_sparsity,
            model=model,
            args=sft_config,
            train_dataset=sft_dataset,
            callbacks=[callback],
        )
        trainer.train()

        final_scores = callback.last_scores

    if final_scores:
        (recovery_dir / "scores.json").write_text(json.dumps(final_scores, indent=2, default=str), encoding="utf-8")
    return final_scores


# =========================================================================
# CLI plumbing
# =========================================================================


def _apply_cli_overrides(
    cfg: SweepConfig,
    *,
    model_id: Optional[str],
    dataset_subset: Optional[str],
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
    enable_pgd: bool,
    judge_n_samples: Optional[int],
    wandb_project: Optional[str],
    pgd_min_layer_idx: Optional[int],
) -> None:
    """Mutate `cfg` in place, applying any non-None CLI overrides.

    Boolean flags (no_cache / enable_*) are one-way: passing the flag forces
    the corresponding cfg field to True. They cannot flip a YAML-set True
    back to False — use the YAML file for that.
    """
    if model_id is not None:
        cfg.model_id = model_id
    if dataset_subset is not None:
        cfg.dataset_subset = dataset_subset
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
    if enable_pgd:
        cfg.pgd.enabled = True
    if judge_n_samples is not None:
        cfg.operational.llm_judge.n_samples = judge_n_samples
    if wandb_project is not None:
        cfg.operational.wandb.project = wandb_project
    if pgd_min_layer_idx is not None:
        cfg.pgd.min_layer_idx = pgd_min_layer_idx


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="YAML config (pydantic SweepConfig). CLI flags below override individual fields.",
)
# ── Common per-run overrides ────────────────────────────────────────────────
@click.option("--model-id", default=None, help="Override model_id (e.g. google/gemma-3-12b-it).")
@click.option("--dataset-subset", default=None, help="Override dataset_subset (e.g. biology, math, chemistry, physics).")
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
@click.option("--enable-pgd", is_flag=True, default=False, help="Force pgd.enabled to True (run PGD recovery per sparsity).")
@click.option("--judge-n-samples", type=int, default=None, help="Override operational.llm_judge.n_samples.")
@click.option("--wandb-project", default=None, help="Override operational.wandb.project (also reads $WANDB_PROJECT).")
@click.option(
    "--pgd-min-layer-idx",
    type=int,
    default=None,
    help="Override pgd.min_layer_idx. Restrict PGD projection to params whose name encodes a layer index strictly greater than this value. -1 keeps all layer-indexed params; None disables the filter.",
)
def main(
    config: Optional[str],
    model_id: Optional[str],
    dataset_subset: Optional[str],
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
    enable_pgd: bool,
    judge_n_samples: Optional[int],
    wandb_project: Optional[str],
    pgd_min_layer_idx: Optional[int],
) -> None:
    """Run Wanda pruning sweep + (optional) PGD recovery per sparsity."""
    # =========================================================================
    # STAGE 1 — SETUP
    # =========================================================================

    cfg = SweepConfig.from_yaml(config) if config else SweepConfig()
    _apply_cli_overrides(
        cfg,
        model_id=model_id,
        dataset_subset=dataset_subset,
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
        enable_pgd=enable_pgd,
        judge_n_samples=judge_n_samples,
        wandb_project=wandb_project,
        pgd_min_layer_idx=pgd_min_layer_idx,
    )

    sparsities = cfg.sweep.nn_linear_sparsities
    print(f"Sweep nn.Linear sparsities: {[f'{s:.1%}' for s in sparsities]}")
    if cfg.pgd.enabled:
        print(f"[pgd] recovery training enabled: n_train={cfg.pgd.n_train}, lr={cfg.pgd.learning_rate}, max_steps={cfg.pgd.max_steps}")

    artifacts_root = resolve_artifacts_root(cfg.operational.artifacts_dir)
    run_id = make_run_id()
    run_dir = make_run_dir(artifacts_root, run_id)
    print(f"[artifacts] run_dir: {run_dir}")

    judge_domains_list: list[str] = cfg.operational.llm_judge.domains if cfg.operational.llm_judge.domains else [cfg.dataset_subset]

    run_metadata = build_run_metadata(
        cfg.model_dump(),
        run_id=run_id,
        script=Path(__file__),
        artifacts_dir_resolved=str(artifacts_root),
        judge_domains_resolved=judge_domains_list if cfg.operational.llm_judge.enabled else None,
    )
    (run_dir / "metadata.json").write_text(json.dumps(run_metadata, indent=2, default=str), encoding="utf-8")

    # ── W&B init ────────────────────────────────────────────────────────────
    wandb_run = None
    if cfg.operational.wandb.enabled:
        import wandb  # local import — saves the cost when W&B is off

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
        # X-axis declarations (see NAMING.md): nn_linear_sparsity for the
        # cross-sparsity sweep curves, recovery/train_step for the
        # within-sparsity PGD curves.
        wandb.define_metric("nn_linear_sparsity")
        wandb.define_metric("sweep/*", step_metric="nn_linear_sparsity")
        wandb.define_metric("recovery/train_step")
        # NOTE: W&B's define_metric only accepts trailing-* globs ("prefix/*"),
        # not mid-path patterns. We therefore bind everything under recovery/*
        # to the train_step axis; recovery/train_step itself is fine because
        # its own (more specific) definition above wins.
        wandb.define_metric("recovery/*", step_metric="recovery/train_step")
        print(f"[wandb] initialized run: {wandb_run.name} ({wandb_run.url})")

    # =========================================================================
    # STAGE 2 — CALIBRATION
    # =========================================================================

    print(f"Loading tokenizer and model: {cfg.model_id}")
    model, tokenizer = load_model_and_tokenizer(cfg.model_id, device=cfg.operational.device)

    print(f"Loading dataset: {cfg.dataset_name}/{cfg.dataset_subset}")
    if cfg.pgd.enabled:
        # Need three non-overlapping splits when PGD recovery is on so the
        # training set doesn't leak into the eval set used by both the sweep
        # and the recovery callback.
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

    # ── Baseline measurement (sweep/baseline namespace) ─────────────────────
    print("\n=== Baseline (pre-pruning) ===")
    baseline_loss = compute_loss(
        model,
        tokenizer,
        eval_texts,
        max_seq_len=cfg.calibration.max_seq_len,
        batch_size=cfg.calibration.batch_size,
    )
    zeros_before, total_params = count_zeros(model)
    print(f"  Loss:            {baseline_loss:.4f}")
    print(f"  Model sparsity:  {zeros_before}/{total_params} ({zeros_before / total_params:.2%})")

    baseline_metrics: dict[str, Any] = {
        "loss": baseline_loss,
        "model_sparsity": zeros_before / total_params,
        "model_zeros": zeros_before,
        "model_total": total_params,
    }
    (run_dir / "baseline.json").write_text(json.dumps(baseline_metrics, indent=2, default=str), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.log({f"sweep/baseline/{k}": float(v) for k, v in baseline_metrics.items()})

    # ── Wanda saliency map (cached) ─────────────────────────────────────────
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

    # =========================================================================
    # STAGE 3 — PRUNE + EVAL SWEEP (+ optional PGD recovery per sparsity)
    # =========================================================================

    validator = MaskSubsetValidator(enabled=not cfg.operational.low_memory)
    results = []
    for i, sparsity in enumerate(sparsities):
        masks = compute_wanda_masks(saliency_map, sparsity)
        validator.validate_and_update(masks)
        apply_masks_to_model(model, masks)

        pruned_loss = compute_loss(
            model,
            tokenizer,
            eval_texts,
            max_seq_len=cfg.calibration.max_seq_len,
            batch_size=cfg.calibration.batch_size,
        )
        zeros_after, _ = count_zeros(model)
        linear_zeros = sum(int((~m).sum().item()) for m in masks.values())
        delta = pruned_loss - baseline_loss
        results.append((sparsity, pruned_loss, delta, zeros_after, linear_zeros))

        # ── Per-step artifacts (sweep/) ─────────────────────────────────────
        # TODO(adrianoh) extract a `StepMetrics` dataclass — see
        # NAMING.md and the longer note that used to live here in commit
        # e7505c9. The same metric names are now duplicated across
        # step_metadata, the W&B define_metric calls, and the wandb.log
        # payload below; PGD's RecoveryEvalCallback adds a fourth.
        step_dir = make_step_dir(run_dir, i)
        sweep_dir = step_dir / "sweep"
        sweep_dir.mkdir(parents=True, exist_ok=True)
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
        (sweep_dir / "step_metadata.json").write_text(json.dumps(step_metadata, indent=2, default=str), encoding="utf-8")

        print(f"\n=== nn.Linear sparsity {sparsity:.1%} (step {i}) ===")
        print(f"  Loss:                 {pruned_loss:.4f} (delta: {delta:+.4f})")
        print(f"  nn.Linear sparsity:   {linear_zeros}/{linear_total} ({linear_zeros / linear_total:.2%})")
        print(f"  Whole-model sparsity: {zeros_after}/{total_params} ({zeros_after / total_params:.2%})")
        print(f"  Step dir:             {step_dir}")

        sweep_scores: dict[str, float] = {}
        if cfg.operational.llm_judge.enabled:
            assert evaluator is not None and judge_questions is not None and judge_answers is not None
            with (
                JsonlSink(sweep_dir / "judgements.jsonl") as j_sink,
                JsonlSink(sweep_dir / "inference.jsonl") as i_sink,
            ):
                sweep_scores, _df_json = evaluator.evaluate(
                    model,
                    tokenizer,
                    domain_questions=judge_questions,
                    domain_answers=judge_answers,
                    judgement_sink=j_sink,
                    inference_sink=i_sink,
                )
            (sweep_dir / "scores.json").write_text(json.dumps(sweep_scores, indent=2, default=str), encoding="utf-8")
            for k, v in sorted(sweep_scores.items()):
                print(f"  llm_judge: {k:<60} {v:.3f}")

        # ── W&B per-sparsity log (sweep/* namespace) ───────────────────────
        if wandb_run is not None:
            log_dict: dict[str, float] = {
                "nn_linear_sparsity": sparsity,
                "sweep/model_sparsity": zeros_after / total_params,
                "sweep/loss": pruned_loss,
                "sweep/loss_delta_vs_baseline": delta,
            }
            for k, v in sweep_scores.items():
                log_dict[f"sweep/{k}"] = float(v)
            wandb_run.log(log_dict)

        # ── PGD recovery per sparsity (recovery/) ──────────────────────────
        if cfg.pgd.enabled:
            recovery_dir = step_dir / "recovery"
            recovery_dir.mkdir(parents=True, exist_ok=True)
            pgd_masks = masks
            if cfg.pgd.min_layer_idx is not None:
                pgd_masks = filter_masks_by_min_layer_idx(masks, cfg.pgd.min_layer_idx)
                print(
                    f"  [recovery] PGD layer subset: keeping {len(pgd_masks)}/{len(masks)} masks "
                    f"with layer index > {cfg.pgd.min_layer_idx}"
                )
                if not pgd_masks:
                    raise ValueError(
                        f"pgd.min_layer_idx={cfg.pgd.min_layer_idx} filtered out every mask. "
                        f"Either lower the cutoff or disable the filter (set pgd.min_layer_idx=null)."
                    )
                # Freeze every early-side param (incl. tied embed/lm_head if
                # tying ties them through embed_tokens). This is the actual
                # source of the memory savings — without it the optimizer
                # still allocates Adam state for layers 0..min_layer_idx.
                # Idempotent across sparsities: setting requires_grad=False
                # twice is a no-op.
                frozen_names, n_frozen_tensors = freeze_early_side_params(model, cfg.pgd.min_layer_idx)
                print(f"  [recovery] froze {n_frozen_tensors} unique tensor(s) (={len(frozen_names)} alias name(s)) on the early side of layer {cfg.pgd.min_layer_idx}")
            print(f"\n  [recovery] Starting PGD training at sparsity {sparsity:.1%}")
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
                judge_questions=judge_questions,
                judge_answers=judge_answers,
                wandb_run=wandb_run,
            )

    # =========================================================================
    # STAGE 4 — TEARDOWN
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
