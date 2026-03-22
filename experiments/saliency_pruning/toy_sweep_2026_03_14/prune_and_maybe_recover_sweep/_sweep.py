"""run_staged_sweep: top-level orchestrator for staged sparsity search."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from dataset_utils import load_qa_dataset
from prune_and_maybe_recover_sweep._exceptions import LeftOutOfBoundsError, RightOutOfBoundsError
from prune_and_maybe_recover_sweep._schemas import (
    SparsityInterval,
    StageResult,
    StagedSweepResult,
)
from prune_and_maybe_recover_sweep._stages import SweepStage


def _run_stage(
    stage: SweepStage,
    interval: SparsityInterval,
    model_name_or_path: str,
    tokenizer: PreTrainedTokenizerBase,
    dataset_eval: Optional[Dataset],
    dataset_recovery: Optional[Dataset],
    device: str,
    output_dir: str,
) -> StageResult:
    """Execute one stage's search loop with pre-loaded data.

    Called by run_staged_sweep after loading datasets.  Also called directly
    by tests to avoid triggering dataset loading.

    Parameters
    ----------
    stage            : stage configuration (evaluator, search, max_steps, …)
    interval         : the resolved [lo, hi] interval for this stage
    model_name_or_path : HF model ID or local path
    tokenizer        : loaded tokenizer (shared across stages)
    dataset_eval     : evaluation dataset subset for this stage
    dataset_recovery : recovery dataset subset for this stage (None = no SFT)
    device           : 'cuda' or 'cpu'
    output_dir       : directory for stage outputs

    Returns
    -------
    StageResult with the narrowed output_interval and per-step records.

    Error handling:
    - If all steps pass and raise_if_out_of_bounds: raises RightOutOfBoundsError
    - If all steps fail and raise_if_out_of_bounds: raises LeftOutOfBoundsError
    - Otherwise emits warnings and returns current bounds.
    """
    stage.evaluator.prepare(model_name_or_path, tokenizer, dataset_eval, device)
    stage.search.start(interval.lo, interval.hi)

    steps = []
    for step_idx in range(stage.max_steps):
        if stage.search.is_converged():
            break

        candidate = stage.search.next_candidate()
        step_dir = str(Path(output_dir) / f"step_{step_idx:04d}")

        result = stage.evaluator.evaluate(
            candidate,
            model_name_or_path,
            tokenizer,
            dataset_eval,
            dataset_recovery,
            device,
            step_dir,
        )
        stage.search.update(candidate, result.is_success)
        steps.append(result)

    if len(steps) >= 2:
        all_pass = all(s.is_success for s in steps)
        all_fail = not any(s.is_success for s in steps)
        if all_pass or all_fail:
            direction = "right (all pass)" if all_pass else "left (all fail)"
            msg = (
                f"[{stage.name}] Out-of-bounds detected after {len(steps)} steps in "
                f"[{interval.lo:.4f}, {interval.hi:.4f}]: {direction}."
            )
            if stage.raise_if_out_of_bounds:
                if all_pass:
                    raise RightOutOfBoundsError(msg)
                raise LeftOutOfBoundsError(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

    lo, hi = stage.search.current_bounds()
    return StageResult(
        name=stage.name,
        input_interval=interval,
        output_interval=SparsityInterval(lo=lo, hi=hi),
        steps=steps,
    )


def run_staged_sweep(
    stages: list[SweepStage],
    model_name_or_path: str,
    initial_interval: SparsityInterval,
    output_dir: str,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    device: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
) -> StagedSweepResult:
    """Run all stages in sequence, each narrowing the sparsity interval.

    For each stage:
      1. Resolve the stage's search interval via stage.interval_spec.
      2. Load eval/recovery datasets (stage.dataset_config).
      3. Load the tokenizer from model_name_or_path (once, shared).
      4. Call stage.evaluator.prepare().
      5. Run the search loop (up to stage.max_steps):
           candidate = stage.search.next_candidate()
           result    = stage.evaluator.evaluate(candidate, ...)
           stage.search.update(candidate, result.is_success)
           check out-of-bounds condition after loop ends
      6. Append StageResult to history.

    Per-stage results are written to JSON under output_dir/stage_NNN_<name>/.
    If wandb_project is set, a single run is opened and each step is logged.

    Parameters
    ----------
    stages               : ordered list of SweepStage configurations
    model_name_or_path   : HF model ID or local path
    initial_interval     : starting [lo, hi] for IntervalSpec.Initial stages
    output_dir           : root output directory; subdirs created per stage
    device               : 'cuda' / 'cpu' (defaults to CUDA if available)
    wandb_project        : WandB project name; no logging if None
    wandb_run_name       : WandB run name; auto-generated if None

    Returns
    -------
    StagedSweepResult with all stage results and the final interval.

    Precondition:  len(stages) >= 1
    Postcondition: len(result.stage_results) == len(stages) unless an
                   OutOfBoundsError is raised mid-run
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    wandb_run = None
    if wandb_project is not None:
        import wandb
        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model": model_name_or_path,
                "initial_lo": initial_interval.lo,
                "initial_hi": initial_interval.hi,
                "n_stages": len(stages),
            },
        )

    history: list[StageResult] = []

    try:
        for stage_idx, stage in enumerate(stages):
            interval = stage.interval_spec.resolve(history, initial_interval)
            stage_dir = str(Path(output_dir) / f"stage_{stage_idx:03d}_{stage.name}")
            Path(stage_dir).mkdir(parents=True, exist_ok=True)

            dataset_eval = load_qa_dataset(
                stage.dataset_config.dataset_name,
                stage.dataset_config.dataset_subset,
                split=stage.dataset_config.split_eval,
                n=stage.dataset_config.n_eval,
                seed=stage.dataset_config.seed,
            )
            dataset_recovery = None
            if stage.dataset_config.n_recovery > 0:
                dataset_recovery = load_qa_dataset(
                    stage.dataset_config.dataset_name,
                    stage.dataset_config.dataset_subset,
                    split=stage.dataset_config.split_recovery,
                    n=stage.dataset_config.n_recovery,
                    seed=stage.dataset_config.seed,
                )

            print(
                f"\n[Stage {stage_idx}: {stage.name}] "
                f"interval=[{interval.lo:.4f}, {interval.hi:.4f}], "
                f"max_steps={stage.max_steps}"
            )

            stage_result = _run_stage(
                stage, interval, model_name_or_path, tokenizer,
                dataset_eval, dataset_recovery, device, stage_dir,
            )
            history.append(stage_result)

            (Path(stage_dir) / "result.json").write_text(
                stage_result.model_dump_json(indent=2)
            )

            if wandb_run is not None:
                for step in stage_result.steps:
                    wandb_run.log({
                        f"{stage.name}/sparsity": step.sparsity,
                        f"{stage.name}/metric_before": step.metric_before,
                        f"{stage.name}/metric_after": step.metric_after,
                        f"{stage.name}/is_success": int(step.is_success),
                    })

            print(
                f"[Stage {stage_idx}: {stage.name}] done. "
                f"output=[{stage_result.output_interval.lo:.4f}, "
                f"{stage_result.output_interval.hi:.4f}]"
            )

    finally:
        if wandb_run is not None:
            wandb_run.finish()

    final_interval = history[-1].output_interval if history else initial_interval
    result = StagedSweepResult(stage_results=history, final_interval=final_interval)
    (Path(output_dir) / "staged_sweep_result.json").write_text(result.model_dump_json(indent=2))
    return result
