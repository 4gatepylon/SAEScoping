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

import gc
import json
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Any, Optional

import click
import torch
from datasets import Dataset
from safetensors.torch import load_file as safetensors_load_file
from transformers import TrainerCallback
from trl import SFTConfig

from sae_scoping.datasets.qa_datasets import format_as_sft_text, load_nonoverlapping_splits, load_qa_dataset
from sae_scoping.evaluation.loss import compute_loss, count_zeros
from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval
from sae_scoping.evaluation.utils import JsonlSink, Sink
from sae_scoping.training.pgd_trainer import PGDSFTTrainer
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
from sae_scoping.utils.click_utils import parse_comma_separated_floats, parse_comma_separated_strings
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
    # Checkpoints (when pgd.save_steps > 0) are written under the run's
    # artifacts dir so each run is self-contained:
    #   <artifacts_root>/outputs/<run_id>/step_NNN/recovery/checkpoints/
    checkpoints_dir = recovery_dir / "checkpoints"
    save_strategy = "steps" if pgd_cfg.save_steps > 0 else "no"
    sft_config = SFTConfig(
        output_dir=str(checkpoints_dir),
        learning_rate=pgd_cfg.learning_rate,
        num_train_epochs=pgd_cfg.num_train_epochs,
        max_steps=pgd_cfg.max_steps,
        per_device_train_batch_size=pgd_cfg.train_batch_size,
        gradient_accumulation_steps=pgd_cfg.gradient_accumulation_steps,
        warmup_ratio=pgd_cfg.warmup_ratio,
        logging_steps=pgd_cfg.logging_steps,
        max_length=cfg.calibration.max_seq_len,
        bf16=True,
        report_to=pgd_cfg.report_to,
        save_strategy=save_strategy,
        save_steps=pgd_cfg.save_steps if pgd_cfg.save_steps > 0 else 500,
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
# Per-sparsity step (extracted so a future commit can run it in a subprocess)
# =========================================================================


def _run_one_sparsity_step(
    *,
    sparsity: float,
    step_idx: int,
    model: Any,
    tokenizer: Any,
    saliency_map: dict[str, torch.Tensor],
    eval_texts: list[str],
    train_texts: list[str],
    baseline_loss: float,
    total_params: int,
    linear_total: int,
    cfg: SweepConfig,
    run_dir: Path,
    validator: MaskSubsetValidator,
    evaluator: Optional[OneClickLLMJudgeScopingEval],
    judge_questions: Optional[dict[str, list[str]]],
    judge_answers: Optional[dict[str, list[str]]],
    wandb_run: Optional[Any],
) -> tuple[float, float, float, int, int]:
    """Apply Wanda mask + eval + (optional) LLM judge + (optional) PGD recovery
    for a single sparsity. Returns the row tuple appended to the summary list.

    Module-level (not a closure) so a parallel-mode dispatcher in a future
    commit can run this in a subprocess. Today it is called once per
    sparsity from a sequential loop in `main`.
    """
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

    # ── Per-step artifacts (sweep/) ─────────────────────────────────────
    # TODO(adrianoh) extract a `StepMetrics` dataclass — see NAMING.md and
    # the longer note that used to live here in commit e7505c9. The same
    # metric names are duplicated across step_metadata, the W&B
    # define_metric calls, and the wandb.log payload below;
    # RecoveryEvalCallback adds a fourth.
    step_dir = make_step_dir(run_dir, step_idx)
    sweep_dir = step_dir / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    step_metadata = {
        "step_idx": step_idx,
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

    print(f"\n=== nn.Linear sparsity {sparsity:.1%} (step {step_idx}) ===")
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

    # ── PGD recovery (recovery/) ───────────────────────────────────────
    if cfg.pgd.enabled:
        recovery_dir = step_dir / "recovery"
        recovery_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  [recovery] Starting PGD training at sparsity {sparsity:.1%}")
        _run_pgd_recovery(
            model=model,
            tokenizer=tokenizer,
            masks=masks,
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

    return sparsity, pruned_loss, delta, zeros_after, linear_zeros


# =========================================================================
# Parallel multi-device dispatch
# =========================================================================
#
# Each worker gets a sub-list of sparsities (round-robin assignment so
# each worker's list is sorted ascending, i.e. a strict superset chain
# safe for the MaskSubsetValidator). Workers reconstruct model + tokenizer
# + saliency_map locally on their assigned device, then call
# `_run_one_sparsity_step` directly.
#
# Trade-offs vs sequential mode:
# * Workers cannot share the master's W&B run (not picklable). In parallel
#   mode, recovery training curves do NOT land in W&B (they remain on
#   disk under step_NNN/recovery/). Sweep-level summary metrics are
#   logged to W&B post-hoc by the master after all workers finish, by
#   re-reading step_NNN/sweep/{step_metadata.json,scores.json}.
# * Cross-sparsity validation by `MaskSubsetValidator` only runs within a
#   worker's sub-list, not across workers (each worker has its own model).


def _resolve_devices(cfg: SweepConfig, devices_arg: Optional[str]) -> list[str]:
    """Resolve the device list. Precedence: --devices arg > [cfg.operational.device]."""
    if devices_arg:
        return parse_comma_separated_strings(devices_arg, sort=True, duplicates="dedup")
    return [cfg.operational.device]


def _worker_main(
    *,
    device: str,
    sparsity_assignments: list[tuple[int, float]],  # [(step_idx, sparsity), ...] sorted ascending by sparsity
    cfg_dict: dict[str, Any],
    run_dir_str: str,
    saliency_path_str: str,
    eval_texts: list[str],
    train_texts: list[str],
    judge_questions: Optional[dict[str, list[str]]],
    judge_answers: Optional[dict[str, list[str]]],
    baseline_loss: float,
    total_params: int,
    linear_total: int,
    result_queue: Any,
) -> None:
    """Spawn-context subprocess entry point.

    Reconstructs model + tokenizer + (optional) evaluator locally on
    `device`, loads the cached saliency map from disk, then drains its
    `sparsity_assignments` through `_run_one_sparsity_step`. Each result
    tuple goes back to the master through `result_queue` as ("ok", tuple);
    any exception goes back as ("err", traceback_string).
    """
    try:
        cfg = SweepConfig(**cfg_dict)
        # Force this worker's device into the cfg so any downstream code
        # (PGD trainer, etc.) sees it as the configured device.
        cfg.operational.device = device
        run_dir = Path(run_dir_str)

        print(f"[worker {device}] loading model {cfg.model_id}")
        model, tokenizer = load_model_and_tokenizer(cfg.model_id, device=device)

        print(f"[worker {device}] loading saliency from {saliency_path_str}")
        saliency_map = safetensors_load_file(saliency_path_str)

        evaluator: Optional[OneClickLLMJudgeScopingEval] = None
        if cfg.operational.llm_judge.enabled:
            evaluator = OneClickLLMJudgeScopingEval(
                n_samples=cfg.operational.llm_judge.n_samples,
                judge_model=cfg.operational.llm_judge.judge_model,
                train_domain=cfg.dataset_subset,
            )

        # Validator only sees this worker's ascending sub-list.
        validator = MaskSubsetValidator(enabled=not cfg.operational.low_memory)

        for step_idx, sparsity in sparsity_assignments:
            result = _run_one_sparsity_step(
                sparsity=sparsity,
                step_idx=step_idx,
                model=model,
                tokenizer=tokenizer,
                saliency_map=saliency_map,
                eval_texts=eval_texts,
                train_texts=train_texts,
                baseline_loss=baseline_loss,
                total_params=total_params,
                linear_total=linear_total,
                cfg=cfg,
                run_dir=run_dir,
                validator=validator,
                evaluator=evaluator,
                judge_questions=judge_questions,
                judge_answers=judge_answers,
                wandb_run=None,  # workers do not log to W&B; master does post-hoc
            )
            result_queue.put(("ok", result))
    except Exception:
        result_queue.put(("err", traceback.format_exc()))


def _dispatch_parallel(
    *,
    devices: list[str],
    sparsities: list[float],
    cfg: SweepConfig,
    run_dir: Path,
    saliency_path: Path,
    eval_texts: list[str],
    train_texts: list[str],
    judge_questions: Optional[dict[str, list[str]]],
    judge_answers: Optional[dict[str, list[str]]],
    baseline_loss: float,
    total_params: int,
    linear_total: int,
) -> list[tuple[float, float, float, int, int]]:
    """Spawn one worker per device and dispatch sparsities round-robin."""
    sparsities_sorted = sorted(sparsities)
    enumerated: list[tuple[int, float]] = list(enumerate(sparsities_sorted))
    assignments: list[list[tuple[int, float]]] = [[] for _ in devices]
    for k, (step_idx, sparsity) in enumerate(enumerated):
        assignments[k % len(devices)].append((step_idx, sparsity))

    print(f"\n[parallel] devices={devices}")
    for d, a in zip(devices, assignments):
        print(f"[parallel]   {d} <- {[f'{s:.1%}' for _, s in a]}")

    ctx = mp.get_context("spawn")
    result_queue: Any = ctx.Queue()
    workers: list[Any] = []
    cfg_dict = cfg.model_dump()
    for d, assignment in zip(devices, assignments):
        if not assignment:
            continue
        w = ctx.Process(
            target=_worker_main,
            kwargs={
                "device": d,
                "sparsity_assignments": assignment,
                "cfg_dict": cfg_dict,
                "run_dir_str": str(run_dir),
                "saliency_path_str": str(saliency_path),
                "eval_texts": eval_texts,
                "train_texts": train_texts,
                "judge_questions": judge_questions,
                "judge_answers": judge_answers,
                "baseline_loss": baseline_loss,
                "total_params": total_params,
                "linear_total": linear_total,
                "result_queue": result_queue,
            },
        )
        w.start()
        workers.append(w)

    results: list[tuple[float, float, float, int, int]] = []
    expected = len(sparsities)
    while len(results) < expected:
        kind, payload = result_queue.get()
        if kind == "err":
            for w in workers:
                if w.is_alive():
                    w.terminate()
            raise RuntimeError(f"Parallel worker failed:\n{payload}")
        results.append(payload)

    for w in workers:
        w.join()

    # Sort results by sparsity so the summary table is in ascending order.
    results.sort(key=lambda r: r[0])
    return results


def _log_sweep_summary_to_wandb_post_hoc(
    *,
    wandb_run: Any,
    run_dir: Path,
    results: list[tuple[float, float, float, int, int]],
    total_params: int,
) -> None:
    """In parallel mode workers can't share the master's wandb_run, so the
    master replays the sweep-level metrics post-hoc by reading
    step_NNN/sweep/scores.json. Recovery curves are NOT replayed (they
    live on disk only — the volume per training step makes pulling them
    into W&B post-hoc unhelpful)."""
    sparsities_sorted = sorted(r[0] for r in results)
    for sparsity, pruned_loss, delta, zeros_after, _linear_zeros in results:
        step_idx = sparsities_sorted.index(sparsity)
        log_dict: dict[str, float] = {
            "nn_linear_sparsity": sparsity,
            "sweep/model_sparsity": zeros_after / total_params,
            "sweep/loss": pruned_loss,
            "sweep/loss_delta_vs_baseline": delta,
        }
        scores_path = run_dir / f"step_{step_idx:03d}" / "sweep" / "scores.json"
        if scores_path.exists():
            scores = json.loads(scores_path.read_text(encoding="utf-8"))
            for k, v in scores.items():
                log_dict[f"sweep/{k}"] = float(v)
        wandb_run.log(log_dict)


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
) -> None:
    """Mutate `cfg` in place, applying any non-None CLI overrides.

    Boolean flags (no_cache / enable_*) are one-way: passing the flag forces
    the corresponding cfg field to True. They cannot flip a YAML-set True
    back to False — use the YAML file for that.

    These exist since they facilitate degging and iteration for common options.
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
@click.option("--device", default=None, help="Override operational.device (single-device mode).")
@click.option(
    "--devices",
    default=None,
    help="Multi-device parallel sweep, comma-separated (e.g. cuda:0,cuda:3,cuda:6). "
    "Round-robin assigns sparsities to workers (one per device). Overrides --device.",
)
@click.option("--artifacts-dir", default=None, help="Override operational.artifacts_dir.")
@click.option("--no-cache", is_flag=True, default=False, help="Force operational.no_cache to True (cannot disable from CLI).")
@click.option("--enable-llm-judge", is_flag=True, default=False, help="Force operational.llm_judge.enabled to True.")
@click.option("--enable-wandb", is_flag=True, default=False, help="Force operational.wandb.enabled to True.")
@click.option("--enable-pgd", is_flag=True, default=False, help="Force pgd.enabled to True (run PGD recovery per sparsity).")
@click.option("--judge-n-samples", type=int, default=None, help="Override operational.llm_judge.n_samples.")
@click.option("--wandb-project", default=None, help="Override operational.wandb.project (also reads $WANDB_PROJECT).")
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
    devices: Optional[str],
    artifacts_dir: Optional[str],
    no_cache: bool,
    enable_llm_judge: bool,
    enable_wandb: bool,
    enable_pgd: bool,
    judge_n_samples: Optional[int],
    wandb_project: Optional[str],
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
    )

    sparsities = cfg.sweep.nn_linear_sparsities
    print(f"Sweep nn.Linear sparsities: {[f'{s:.1%}' for s in sparsities]}")
    if cfg.pgd.enabled:
        print(f"[pgd] recovery training enabled: n_train={cfg.pgd.n_train}, lr={cfg.pgd.learning_rate}, max_steps={cfg.pgd.max_steps}")

    device_list = _resolve_devices(cfg, devices)
    is_parallel = len(device_list) > 1
    if is_parallel:
        print(f"[parallel] dispatching across {len(device_list)} devices: {device_list}")

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
                name=cfg.operational.wandb.name or run_id,
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
        wandb.define_metric("recovery/sparsity=*/*", step_metric="recovery/train_step")
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
        # TODO(adriano) this should be refactored to be using the same codee as ^, which should
        # support the needed arguments to output the desired sizes
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
    # TODO(adrianoh) wher eis the baseline evaluation using LLM Judge evaluator?
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

    if is_parallel:
        # Free the master's model before spawning workers so they have full
        # GPU memory available on their assigned devices.
        del model
        del tokenizer
        del saliency_map
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        results = _dispatch_parallel(
            devices=device_list,
            sparsities=sparsities,
            cfg=cfg,
            run_dir=run_dir,
            saliency_path=saliency_file,
            eval_texts=eval_texts,
            train_texts=train_texts,
            judge_questions=judge_questions,
            judge_answers=judge_answers,
            baseline_loss=baseline_loss,
            total_params=total_params,
            linear_total=linear_total,
        )
        if wandb_run is not None:
            _log_sweep_summary_to_wandb_post_hoc(
                wandb_run=wandb_run,
                run_dir=run_dir,
                results=results,
                total_params=total_params,
            )
    else:
        validator = MaskSubsetValidator(enabled=not cfg.operational.low_memory)
        results = []
        for i, sparsity in enumerate(sparsities):
            results.append(
                _run_one_sparsity_step(
                    sparsity=sparsity,
                    step_idx=i,
                    model=model,
                    tokenizer=tokenizer,
                    saliency_map=saliency_map,
                    eval_texts=eval_texts,
                    train_texts=train_texts,
                    baseline_loss=baseline_loss,
                    total_params=total_params,
                    linear_total=linear_total,
                    cfg=cfg,
                    run_dir=run_dir,
                    validator=validator,
                    evaluator=evaluator,
                    judge_questions=judge_questions,
                    judge_answers=judge_answers,
                    wandb_run=wandb_run,
                )
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
