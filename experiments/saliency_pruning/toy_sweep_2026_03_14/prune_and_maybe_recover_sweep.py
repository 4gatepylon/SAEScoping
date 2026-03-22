"""
prune_and_maybe_recover_sweep.py

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
sparsity: if sparsity X is feasible (passes the quality threshold after
recovery), then all Y < X are also feasible.

If this assumption is violated (the metric is non-monotone), the binary
search will produce incorrect bounds without warning. This is a known
limitation and is the caller's responsibility to handle.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUT-OF-BOUNDS DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When raise_if_out_of_bounds=True (default):
  - All steps fail  → LeftOutOfBoundsError  ("feasible region is below lo")
  - All steps pass  → RightOutOfBoundsError ("feasible region is above hi")

When raise_if_out_of_bounds=False:
  - A ⚠️ warning is emitted.
  - SearchAlgorithm.update() is called as normal; concrete implementations
    MAY choose to stabilise their bounds (return the same lo/hi repeatedly)
    rather than collapsing to a degenerate interval.  This pathway is
    reserved for future use; current implementations do not special-case it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PUBLIC API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  run_staged_sweep          – top-level entry point
  SweepStage                – data container for one stage's configuration
  DatasetConfig             – per-stage dataset sizing
  SFTRecoveryConfig         – SFT training hyperparameters
  BinarySearch              – midpoint search algorithm
  UniformGridSearch         – precision-grid search algorithm
  Initial / ChainFromPrevious / BoundedByPreviousHi / SpanFromPreviousStages
                            – interval resolvers
  PruneAndSFTRecoverEvaluator – concrete evaluator wrapping prune_and_maybe_recover
  SparsityInterval / StepEvalResult / StageResult / StagedSweepResult
                            – result schemas
  OutOfBoundsError / LeftOutOfBoundsError / RightOutOfBoundsError
                            – exception hierarchy
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import pydantic
from datasets import Dataset
from transformers import PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class OutOfBoundsError(Exception):
    """Raised when the entire search range lies on one side of the boundary.

    Only raised when SweepStage.raise_if_out_of_bounds is True (the default).

    Precondition:  at least two steps were taken so the detection is reliable.
    Postcondition: the stage's output_interval is undefined (exception aborts
                   the stage before a result is returned).
    """


class LeftOutOfBoundsError(OutOfBoundsError):
    """Every candidate in [lo, hi] fails.

    Interpretation: the feasible region (if any) lies entirely to the left of
    (below) the search range.  The model cannot meet the quality threshold
    anywhere in [lo, hi] even after recovery.

    Common causes:
    - lo is already above the true maximum feasible sparsity
    - threshold is too strict for the given model / dataset
    """


class RightOutOfBoundsError(OutOfBoundsError):
    """Every candidate in [lo, hi] passes.

    Interpretation: the feasible region extends entirely to the right of
    (above) the search range.  The model meets the quality threshold even at
    hi after recovery.

    Common causes:
    - hi is already below the true maximum feasible sparsity
    - threshold is too lenient; consider raising it
    """


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SparsityInterval(pydantic.BaseModel):
    """A closed interval [lo, hi] in sparsity space.

    Invariant: 0.0 <= lo <= hi <= 1.0
    """

    lo: float
    hi: float

    @pydantic.model_validator(mode="after")
    def _check_order(self) -> SparsityInterval:
        if self.lo > self.hi:
            raise ValueError(f"SparsityInterval requires lo <= hi, got [{self.lo}, {self.hi}]")
        return self


class StepEvalResult(pydantic.BaseModel):
    """Result of evaluating a single candidate sparsity.

    Fields
    ------
    sparsity        : the candidate tried
    metric_before   : metric of the pruned model *before* any recovery training
    metric_after    : metric after recovery (== metric_before if no recovery ran)
    is_success      : True iff metric_after meets the quality threshold
    extra           : arbitrary stage-specific diagnostics (steps taken, etc.)
    """

    sparsity: float
    metric_before: float
    metric_after: float
    is_success: bool
    extra: dict[str, Any] = {}


class StageResult(pydantic.BaseModel):
    """Result of a complete SweepStage run.

    Fields
    ------
    name             : mirrors SweepStage.name
    input_interval   : the [lo, hi] interval the stage was given
    output_interval  : the narrowed [lo, hi] after the stage's search loop
    steps            : ordered list of per-candidate results

    Guarantee: input_interval.lo <= output_interval.lo
               output_interval.hi <= input_interval.hi
    """

    name: str
    input_interval: SparsityInterval
    output_interval: SparsityInterval
    steps: list[StepEvalResult]


class StagedSweepResult(pydantic.BaseModel):
    """Aggregated result across all stages.

    Fields
    ------
    stage_results    : one StageResult per stage, in execution order
    final_interval   : output_interval of the last stage
    """

    stage_results: list[StageResult]
    final_interval: SparsityInterval


class DatasetConfig(pydantic.BaseModel):
    """Per-stage dataset configuration.

    All stages use the same underlying HuggingFace dataset (dataset_name /
    dataset_subset) but with independently seeded random subsets of different
    sizes.  Using different seeds across stages avoids evaluating on the same
    examples that were used for earlier coarse screening.

    Fields
    ------
    dataset_name    : HuggingFace dataset ID
    dataset_subset  : dataset configuration / subset name
    n_eval          : number of evaluation examples to load
    n_recovery      : number of recovery-SFT training examples to load
    split_eval      : HF dataset split for evaluation
    split_recovery  : HF dataset split for recovery training
    seed            : random seed for subsetting (should differ per stage)
    """

    dataset_name: str = "4gate/StemQAMixture"
    dataset_subset: str = "biology"
    n_eval: int
    n_recovery: int
    split_eval: str = "validation"
    split_recovery: str = "train"
    seed: int = 42


class SFTRecoveryConfig(pydantic.BaseModel):
    """Hyperparameters for the SFT recovery training step.

    Used by PruneAndSFTRecoverEvaluator only.  Other evaluator types define
    their own config schema; this class is intentionally specific to SFT so
    that future RL or other training schemes can introduce their own without
    changing the evaluator interface.

    Fields
    ------
    max_steps           : maximum SFT gradient steps per candidate
    eval_every          : evaluate quality metric every N steps
    batch_size          : per-device training batch size
    learning_rate       : SFT learning rate
    max_seq_len         : maximum tokenised sequence length
    max_new_tokens      : maximum generation tokens (judge metric only)
    give_up_thresholds  : list of GiveUpThreshold; abort recovery early if
                          quality hasn't reached a weaker threshold by N steps
    """

    max_steps: int = 500
    eval_every: int = 50
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_seq_len: int = 1024
    max_new_tokens: int = 256
    give_up_thresholds: list[Any] = []


# ---------------------------------------------------------------------------
# SearchAlgorithm
# ---------------------------------------------------------------------------


class SearchAlgorithm(ABC):
    """Stateful strategy for choosing the next candidate sparsity to probe.

    Contract
    --------
    - start(lo, hi) must be called once before any other method.
    - next_candidate() and update() must be called alternately (next first).
    - current_bounds() returns (lo', hi') with lo <= lo' <= hi' <= hi at any
      point after start().
    - is_converged() must return True within a finite number of calls so that
      the search loop always terminates.

    Out-of-bounds contract (future use)
    ------------------------------------
    When raise_if_out_of_bounds=False and all evaluations have the same
    outcome (all pass or all fail), update() MAY keep the bounds stable
    (i.e. return the same lo/hi on successive calls) to signal stagnation.
    Current implementations do not implement this; they collapse lo→hi as
    normal and let the stage detect the all-pass / all-fail condition.

    Subclasses must be deterministic given the same sequence of updates.
    """

    @abstractmethod
    def start(self, lo: float, hi: float) -> None:
        """Initialise or reset the algorithm for a new search in [lo, hi].

        Precondition:  lo <= hi
        Postcondition: current_bounds() == (lo, hi); is_converged() is False
                       (unless lo == hi exactly)
        """
        raise NotImplementedError

    @abstractmethod
    def next_candidate(self) -> float:
        """Return the next sparsity value to evaluate.

        Precondition:  start() has been called; is_converged() is False
        Postcondition: returned value c satisfies lo <= c <= hi
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, candidate: float, passed: bool) -> None:
        """Record the outcome of evaluating `candidate` and update bounds.

        Precondition:  next_candidate() was called immediately before this
        Postcondition: current_bounds() are within the previous bounds;
                       new_lo >= old_lo and new_hi <= old_hi
        """
        raise NotImplementedError

    @abstractmethod
    def current_bounds(self) -> tuple[float, float]:
        """Return (lo, hi) — the current uncertainty interval.

        Invariant: lo <= hi at all times after start()
        """
        raise NotImplementedError

    @abstractmethod
    def is_converged(self) -> bool:
        """Return True when no further narrowing is possible or useful.

        BinarySearch:    hi - lo < tolerance
        UniformGridSearch: all grid points within [lo, hi] have been visited
        """
        raise NotImplementedError


class BinarySearch(SearchAlgorithm):
    """Classic binary search: next candidate = midpoint of current [lo, hi].

    Each passing result moves lo up to the candidate; each failing result
    moves hi down.  Converges when hi - lo < tolerance.

    Parameters
    ----------
    tolerance : float
        Convergence threshold (default 1e-6).  The search is considered
        converged when hi - lo < tolerance.
    """

    def __init__(self, tolerance: float = 1e-6) -> None:
        self._tolerance = tolerance
        self._lo: float = 0.0
        self._hi: float = 1.0

    def start(self, lo: float, hi: float) -> None:
        """Reset to a fresh search in [lo, hi].

        Precondition:  lo <= hi
        Postcondition: current_bounds() == (lo, hi)
        """
        raise NotImplementedError

    def next_candidate(self) -> float:
        """Return midpoint of current [lo, hi].

        Precondition:  start() called; not converged
        Postcondition: returned value == (lo + hi) / 2
        """
        raise NotImplementedError

    def update(self, candidate: float, passed: bool) -> None:
        """Narrow bounds: passed → lo = candidate; failed → hi = candidate.

        Precondition:  candidate == last result of next_candidate()
        Postcondition: bounds strictly narrowed (old_hi - old_lo > new_hi - new_lo)
                       unless already at tolerance
        """
        raise NotImplementedError

    def current_bounds(self) -> tuple[float, float]:
        """Return (self._lo, self._hi)."""
        raise NotImplementedError

    def is_converged(self) -> bool:
        """Return self._hi - self._lo < self._tolerance."""
        raise NotImplementedError


class UniformGridSearch(SearchAlgorithm):
    """Iterates through a uniform precision grid from hi→lo (descending).

    The grid is generated within [lo, hi] on start().  Each candidate is
    the next unvisited grid point.  The bounds (lo, hi) are updated after
    each step exactly as in BinarySearch (passed → lo=candidate, failed →
    hi=candidate) so that intervals narrow consistently.

    Descending order is preferred for sparsity sweeps: we want to find the
    *highest* feasible sparsity first, stopping as soon as we pass.

    Converges when all grid points within [lo, hi] have been visited.

    Parameters
    ----------
    precision : float
        Grid step size (default 0.05, giving 21 points across [0, 1]).
    """

    def __init__(self, precision: float = 0.05) -> None:
        self._precision = precision
        self._grid: list[float] = []
        self._index: int = 0
        self._lo: float = 0.0
        self._hi: float = 1.0

    def start(self, lo: float, hi: float) -> None:
        """Generate a descending grid within [lo, hi] and reset the index.

        Grid points are all multiples of precision that lie within [lo, hi],
        sorted in descending order (hi → lo).

        Precondition:  lo <= hi; precision > 0
        Postcondition: _grid is non-empty (or converged immediately if no
                       grid point fits); _index = 0
        """
        raise NotImplementedError

    def next_candidate(self) -> float:
        """Return the next unvisited grid point.

        Precondition:  not converged
        Postcondition: returned value is in _grid[_index]; lo <= value <= hi
        """
        raise NotImplementedError

    def update(self, candidate: float, passed: bool) -> None:
        """Advance index; update lo/hi bounds as in BinarySearch.

        Precondition:  candidate == _grid[_index]
        Postcondition: _index incremented by 1; bounds updated
        """
        raise NotImplementedError

    def current_bounds(self) -> tuple[float, float]:
        """Return (self._lo, self._hi)."""
        raise NotImplementedError

    def is_converged(self) -> bool:
        """Return True when _index >= len(_grid)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# IntervalSpec
# ---------------------------------------------------------------------------


class IntervalSpec(ABC):
    """Resolves the [lo, hi] interval for a stage given the run history.

    Contract
    --------
    - resolve() is called exactly once per stage, before start().
    - The returned interval must satisfy:
        initial.lo <= result.lo <= result.hi <= initial.hi
      (output is always within the user-supplied initial interval).
    - Raises IndexError if a required history entry does not exist.
    """

    @abstractmethod
    def resolve(
        self,
        history: list[StageResult],
        initial: SparsityInterval,
    ) -> SparsityInterval:
        """Compute and return the interval this stage should search.

        Parameters
        ----------
        history : already-completed stage results, in execution order
        initial : the user-supplied initial interval for the whole sweep

        Returns
        -------
        SparsityInterval within initial
        """
        raise NotImplementedError


class Initial(IntervalSpec):
    """Always return the user-supplied initial interval (ignores history).

    Use for the first stage or whenever the full search range is desired.
    """

    def resolve(self, history: list[StageResult], initial: SparsityInterval) -> SparsityInterval:
        """Return initial unchanged.

        Postcondition: result == initial
        """
        raise NotImplementedError


class ChainFromPrevious(IntervalSpec):
    """Use the immediately preceding stage's output_interval as this stage's input.

    Use this for simple sequential narrowing where each stage refines the
    previous stage's result.

    Precondition: len(history) >= 1
    Raises IndexError if history is empty.
    """

    def resolve(self, history: list[StageResult], initial: SparsityInterval) -> SparsityInterval:
        """Return history[-1].output_interval.

        Precondition:  len(history) >= 1
        Postcondition: result == history[-1].output_interval
        """
        raise NotImplementedError


class BoundedByPreviousHi(IntervalSpec):
    """Use [initial.lo, history[stage_idx].output_interval.hi].

    Use this for a stricter stage that should search everything from the
    initial lower bound up to an earlier stage's upper bound.

    Example: Stage 0 finds the loose boundary at hi=0.70.  Stage 1 with
    BoundedByPreviousHi(0) searches [0.0, 0.70] for the strict boundary.

    Precondition:  len(history) > stage_idx
    Raises IndexError if stage_idx is out of range.
    """

    def __init__(self, stage_idx: int) -> None:
        self._stage_idx = stage_idx

    def resolve(self, history: list[StageResult], initial: SparsityInterval) -> SparsityInterval:
        """Return SparsityInterval(lo=initial.lo, hi=history[stage_idx].output_interval.hi).

        Precondition:  len(history) > self._stage_idx
        """
        raise NotImplementedError


class SpanFromPreviousStages(IntervalSpec):
    """Use [history[lo_stage_idx].output_interval.lo, history[hi_stage_idx].output_interval.hi].

    Use this for a refinement stage that should span from one stage's lower
    bound to another stage's upper bound.

    Example: Stage 0 (loose) finds hi=0.70; Stage 1 (strict) finds lo=0.51.
    Stage 2 with SpanFromPreviousStages(lo_stage_idx=1, hi_stage_idx=0)
    searches [0.51, 0.70].

    Precondition:  len(history) > max(lo_stage_idx, hi_stage_idx)
                   lo <= hi after resolution
    Raises IndexError if either index is out of range.
    Raises ValueError if the resolved lo > hi.
    """

    def __init__(self, lo_stage_idx: int, hi_stage_idx: int) -> None:
        self._lo_stage_idx = lo_stage_idx
        self._hi_stage_idx = hi_stage_idx

    def resolve(self, history: list[StageResult], initial: SparsityInterval) -> SparsityInterval:
        """Return SparsityInterval(lo=history[lo_idx].lo, hi=history[hi_idx].hi).

        Precondition:  both indices in range; resolved lo <= hi
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# StepEvaluator
# ---------------------------------------------------------------------------


class StepEvaluator(ABC):
    """Black-box evaluation of a single candidate sparsity.

    The evaluator encapsulates everything that happens when we "try" a
    sparsity value: the erasure operation (pruning), optional recovery
    training (SFT, RL, …), metric evaluation, and success determination.

    Callers (the search loop) interact only through prepare() and evaluate().
    They never need to know the training scheme or metric internals.

    Contract
    --------
    prepare() is called ONCE before the search loop for a given stage.  It
    may load models, resolve fraction thresholds, or cache expensive state.

    evaluate() may be called multiple times after prepare().  Each call is
    independent: it should reload the model from model_name_or_path so that
    in-place pruning from a previous call does not affect the current one.

    Thread safety: not required.  Calls are sequential within a stage.
    """

    @property
    @abstractmethod
    def metric_type(self) -> str:
        """'loss' (lower=better) or 'judge' (higher=better).

        Determines the direction of the threshold comparison in is_success.
        Must be consistent between prepare() and evaluate() calls.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        dataset_evaluation: Dataset,
        device: str,
    ) -> None:
        """One-time setup called before the search loop begins.

        Implementations may:
        - Resolve a fraction threshold by evaluating the unpruned model once.
        - Pre-tokenise the evaluation dataset.
        - Any other per-stage initialisation.

        Precondition:  called exactly once per stage, before any evaluate()
        Postcondition: the evaluator is ready to accept evaluate() calls
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        sparsity: float,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        dataset_evaluation: Dataset,
        dataset_recovery: Optional[Dataset],
        device: str,
        output_dir: str,
    ) -> StepEvalResult:
        """Evaluate sparsity: prune + optional recovery + metric.

        Precondition:  prepare() has been called for this stage
                       0.0 <= sparsity <= 1.0
                       dataset_recovery is not None when threshold >= 0
        Postcondition: returns StepEvalResult where
                       metric_after >= metric_before  (for loss; pruning hurts)
                       is_success == True iff metric_after meets threshold
        """
        raise NotImplementedError


class PruneAndSFTRecoverEvaluator(StepEvaluator):
    """Evaluator backed by prune_and_maybe_recover (SFT recovery).

    prepare() resolves fraction thresholds by loading the model once and
    running evaluate_model on the unpruned weights, then caching the
    effective absolute threshold for all subsequent evaluate() calls.

    evaluate() calls prune_and_maybe_recover() from prune_and_maybe_recover.py.
    The model is loaded fresh from model_name_or_path on every call (because
    pruning modifies weights in-place and checkpoints must be reloaded).

    Parameters
    ----------
    saliency_path    : path to .safetensors saliency map
    saliency_type    : 'gradient' or 'taylor'
    metric_type      : 'loss' or 'judge'
    threshold        : quality threshold (absolute value or multiplier for fraction mode)
    threshold_mode   : 'absolute' or 'fraction'
    recovery_config  : SFT training hyperparameters
    """

    def __init__(
        self,
        saliency_path: Path,
        saliency_type: str,
        metric_type: str,
        threshold: float,
        threshold_mode: str,
        recovery_config: SFTRecoveryConfig,
    ) -> None:
        self._saliency_path = saliency_path
        self._saliency_type = saliency_type
        self._metric_type = metric_type
        self._threshold = threshold
        self._threshold_mode = threshold_mode
        self._recovery_config = recovery_config
        self._effective_threshold: Optional[float] = None

    @property
    def metric_type(self) -> str:
        """Return 'loss' or 'judge'."""
        return self._metric_type

    def prepare(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        dataset_evaluation: Dataset,
        device: str,
    ) -> None:
        """Resolve fraction threshold if needed; cache effective threshold.

        When threshold_mode='fraction' and threshold >= 0:
          Loads model from model_name_or_path, calls evaluate_model on the
          UNPRUNED weights to get baseline_metric, then sets:
            _effective_threshold = threshold * baseline_metric
          Deletes the model and clears CUDA cache before returning.

        When threshold_mode='absolute' or threshold < 0:
          Sets _effective_threshold = threshold directly.

        Precondition:  model_name_or_path loadable; dataset_evaluation non-empty
        Postcondition: self._effective_threshold is not None
        """
        raise NotImplementedError

    def evaluate(
        self,
        sparsity: float,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        dataset_evaluation: Dataset,
        dataset_recovery: Optional[Dataset],
        device: str,
        output_dir: str,
    ) -> StepEvalResult:
        """Call prune_and_maybe_recover at the given sparsity.

        Precondition:  prepare() called; _effective_threshold is set
        Postcondition: returned StepEvalResult.sparsity == sparsity
                       is_success == is_metric_passing(metric_after,
                                        metric_type, _effective_threshold)
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SweepStage
# ---------------------------------------------------------------------------


class SweepStage(pydantic.BaseModel):
    """Configuration for one search stage (pure data container).

    Holds everything needed to execute one pass of the search loop.
    Execution logic lives in run_staged_sweep, not here.

    Fields
    ------
    name                 : human-readable label (used in logs and output JSON)
    evaluator            : StepEvaluator instance (black-box prune+recovery)
    search               : SearchAlgorithm instance
    interval_spec        : how to resolve [lo, hi] from previous history
    max_steps            : maximum search steps before early exit
    dataset_config       : which examples to load for eval and recovery
    raise_if_out_of_bounds : if True (default), raise on all-pass/all-fail
                             if False, emit warning and let search decide

    Contract
    --------
    The search loop iterates at most max_steps times.  If is_converged()
    fires before max_steps, the loop exits early.  If max_steps is reached
    first, the stage returns its best current bounds — early termination is
    not an error.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    name: str
    evaluator: StepEvaluator
    search: SearchAlgorithm
    interval_spec: IntervalSpec
    max_steps: int
    dataset_config: DatasetConfig
    raise_if_out_of_bounds: bool = True


# ---------------------------------------------------------------------------
# run_staged_sweep
# ---------------------------------------------------------------------------


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
    model_name_or_path   : HF model ID or local path (passed to each evaluator)
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
