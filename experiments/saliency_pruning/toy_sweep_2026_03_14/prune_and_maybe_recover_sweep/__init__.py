"""
prune_and_maybe_recover_sweep
===============================

Staged hyperparameter search over sparsity (or any monotone scalar parameter)
to find the maximum value at which a model meets a quality threshold after an
optional recovery training step.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM DIAGRAM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  User script
      │
      │  stages: list[SweepStage]          ← one SweepStage per phase
      │  initial_interval: SparsityInterval
      ▼
  run_staged_sweep(stages, model_name_or_path, initial_interval, ...)
      │
      │  for each SweepStage:
      │
      │    1. IntervalSpec.resolve(history, initial)
      │          │
      │          ├─ Initial()               → always returns `initial`
      │          ├─ ChainFromPrevious()     → previous stage output
      │          ├─ BoundedByPreviousHi(i)  → [initial.lo, history[i].hi]
      │          └─ SpanFromPreviousStages  → [history[i].lo, history[j].hi]
      │
      │    2. Load datasets (DatasetConfig: n_eval, n_recovery, seed)
      │
      │    3. StepEvaluator.prepare(model, tokenizer, dataset_eval, device)
      │          └─ PruneAndSFTRecoverEvaluator: resolves fraction threshold
      │             by evaluating the unpruned model once
      │
      │    4. SearchAlgorithm.start(lo, hi)
      │
      │    5. Search loop (up to SweepStage.max_steps iterations):
      │
      │         candidate = SearchAlgorithm.next_candidate()
      │              │
      │              ├─ BinarySearch:       midpoint of [lo, hi]
      │              └─ UniformGridSearch:  next point in precision grid, hi→lo
      │
      │         result = StepEvaluator.evaluate(candidate, ...)
      │              └─ PruneAndSFTRecoverEvaluator:
      │                   prune_and_maybe_recover(model, sparsity=candidate, ...)
      │                   returns StepEvalResult(metric_before, metric_after,
      │                                          is_success)
      │
      │         SearchAlgorithm.update(candidate, result.is_success)
      │              ├─ BinarySearch:       is_success → lo=mid, else hi=mid
      │              └─ UniformGridSearch:  advance index; update lo/hi bounds
      │
      │         out-of-bounds check (if raise_if_out_of_bounds=True):
      │              all passes → RightOutOfBoundsError
      │              all fails  → LeftOutOfBoundsError
      │
      │    6. Append StageResult to history
      │       (output_interval = SearchAlgorithm.current_bounds())
      │
      ▼
  StagedSweepResult(stage_results, final_interval)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MONOTONICITY ASSUMPTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All callers assume the feasibility predicate is monotone decreasing in
sparsity: if sparsity X is feasible, then all Y < X are also feasible.
If violated, binary search will produce incorrect bounds without warning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUT-OF-BOUNDS DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When raise_if_out_of_bounds=True (default):
  - All steps fail  → LeftOutOfBoundsError
  - All steps pass  → RightOutOfBoundsError

When raise_if_out_of_bounds=False:
  - A warning is emitted; search continues.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PUBLIC API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  run_staged_sweep          – top-level entry point             (_sweep)
  SweepStage                – stage configuration container     (_stages)
  DatasetConfig             – per-stage dataset sizing          (_schemas)
  SFTRecoveryConfig         – SFT training hyperparameters      (_schemas)
  BinarySearch              – midpoint search algorithm         (_search)
  UniformGridSearch         – precision-grid search algorithm   (_search)
  Initial / ChainFromPrevious / BoundedByPreviousHi /
  SpanFromPreviousStages    – interval resolvers                (_intervals)
  PruneAndSFTRecoverEvaluator – concrete SFT evaluator          (_evaluators)
  SparsityInterval / StepEvalResult / StageResult /
  StagedSweepResult         – result schemas                    (_schemas)
  OutOfBoundsError / LeftOutOfBoundsError /
  RightOutOfBoundsError     – exception hierarchy               (_exceptions)
"""

from prune_and_maybe_recover_sweep._evaluators import PruneAndSFTRecoverEvaluator, StepEvaluator
from prune_and_maybe_recover_sweep._exceptions import (
    LeftOutOfBoundsError,
    OutOfBoundsError,
    RightOutOfBoundsError,
)
from prune_and_maybe_recover_sweep._intervals import (
    BoundedByPreviousHi,
    ChainFromPrevious,
    Initial,
    IntervalSpec,
    SpanFromPreviousStages,
)
from prune_and_maybe_recover_sweep._schemas import (
    DatasetConfig,
    SFTRecoveryConfig,
    SparsityInterval,
    StageResult,
    StagedSweepResult,
    StepEvalResult,
)
from prune_and_maybe_recover_sweep._search import BinarySearch, SearchAlgorithm, UniformGridSearch
from prune_and_maybe_recover_sweep._stages import SweepStage
from prune_and_maybe_recover_sweep._sweep import run_staged_sweep

__all__ = [
    "run_staged_sweep",
    "SweepStage",
    "DatasetConfig",
    "SFTRecoveryConfig",
    "BinarySearch",
    "UniformGridSearch",
    "SearchAlgorithm",
    "Initial",
    "ChainFromPrevious",
    "BoundedByPreviousHi",
    "SpanFromPreviousStages",
    "IntervalSpec",
    "PruneAndSFTRecoverEvaluator",
    "StepEvaluator",
    "SparsityInterval",
    "StepEvalResult",
    "StageResult",
    "StagedSweepResult",
    "OutOfBoundsError",
    "LeftOutOfBoundsError",
    "RightOutOfBoundsError",
]
