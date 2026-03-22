"""run_staged_sweep: top-level orchestrator for staged sparsity search."""

from __future__ import annotations

import warnings
from typing import Optional

from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from prune_and_maybe_recover_sweep._exceptions import LeftOutOfBoundsError, RightOutOfBoundsError
from prune_and_maybe_recover_sweep._schemas import (
    SparsityInterval,
    StageResult,
    StagedSweepResult,
    StepEvalResult,
)
from prune_and_maybe_recover_sweep._stages import SweepStage


def _run_stage(
    stage: SweepStage,
    interval: SparsityInterval,
    model_name_or_path: str,
    tokenizer: PreTrainedTokenizerBase,
    dataset_eval: Dataset,
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
    raise NotImplementedError


def run_staged_sweep(
    stages: list[SweepStage],
    model_name_or_path: str,
    initial_interval: SparsityInterval,
    output_dir: str,
    device: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
) -> StagedSweepResult:
    """Run all stages in sequence, each narrowing the sparsity interval.

    For each stage:
      1. Resolve the stage's search interval via stage.interval_spec.
      2. Load eval/recovery datasets (stage.dataset_config).
      3. Load the tokenizer from model_name_or_path.
      4. Call stage.evaluator.prepare().
      5. Run the search loop (up to stage.max_steps):
           candidate = stage.search.next_candidate()
           result    = stage.evaluator.evaluate(candidate, ...)
           stage.search.update(candidate, result.is_success)
           check out-of-bounds condition after every step
      6. Append StageResult to history.

    Per-stage results are written to JSON under output_dir/stage_NNN/.
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
    raise NotImplementedError
