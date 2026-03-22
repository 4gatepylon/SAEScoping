"""run_staged_sweep: top-level orchestrator for staged sparsity search."""

from __future__ import annotations

from typing import Optional

from prune_and_maybe_recover_sweep._schemas import SparsityInterval, StagedSweepResult
from prune_and_maybe_recover_sweep._stages import SweepStage


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
