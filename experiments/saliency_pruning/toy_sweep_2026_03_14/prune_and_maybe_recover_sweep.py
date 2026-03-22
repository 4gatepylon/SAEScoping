"""
prune_and_maybe_recover_sweep.py

Staged hyperparameter search over sparsity (or any monotone scalar parameter)
to find the maximum value at which a model meets a quality threshold after an
optional recovery training step.

See prune_and_maybe_recover_sweep_old.py for the previous single-stage
binary-search implementation that this replaces.
"""

from __future__ import annotations

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
    See SweepStage docstring for the no-raise contract.
    """


class LeftOutOfBoundsError(OutOfBoundsError):
    """Every candidate in [lo, hi] fails — the feasible region is entirely
    to the left of (below) the search range."""


class RightOutOfBoundsError(OutOfBoundsError):
    """Every candidate in [lo, hi] passes — the feasible region extends
    entirely to the right of (above) the search range."""


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SparsityInterval(pydantic.BaseModel):
    lo: float
    hi: float


class StepEvalResult(pydantic.BaseModel):
    sparsity: float
    metric_before: float
    metric_after: float
    is_success: bool
    extra: dict[str, Any] = {}


class StageResult(pydantic.BaseModel):
    name: str
    input_interval: SparsityInterval
    output_interval: SparsityInterval
    steps: list[StepEvalResult]


class StagedSweepResult(pydantic.BaseModel):
    stage_results: list[StageResult]
    final_interval: SparsityInterval


class DatasetConfig(pydantic.BaseModel):
    """Per-stage dataset sizing. Same underlying HF dataset; different sample counts."""

    dataset_name: str = "4gate/StemQAMixture"
    dataset_subset: str = "biology"
    n_eval: int
    n_recovery: int
    split_eval: str = "validation"
    split_recovery: str = "train"
    seed: int = 42


class SFTRecoveryConfig(pydantic.BaseModel):
    """Hyperparameters for the SFT recovery training step.

    Passed into PruneAndSFTRecoverEvaluator. Other evaluator types (e.g. RL)
    define their own config schema; this class is specific to SFT.
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

    Lifecycle per stage:
        1. search.start(lo, hi)          — initialise / reset for this stage
        2. while not search.is_converged():
               candidate = search.next_candidate()
               search.update(candidate, passed)
        3. lo, hi = search.current_bounds()

    Subclasses must be deterministic given the same sequence of updates.
    """

    @abstractmethod
    def start(self, lo: float, hi: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def next_candidate(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def update(self, candidate: float, passed: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def current_bounds(self) -> tuple[float, float]:
        raise NotImplementedError

    @abstractmethod
    def is_converged(self) -> bool:
        raise NotImplementedError


class BinarySearch(SearchAlgorithm):
    """Classic binary search: next candidate = midpoint of [lo, hi].

    Converges when hi - lo < tolerance.
    """

    def __init__(self, tolerance: float = 1e-6) -> None:
        self._tolerance = tolerance
        self._lo: float = 0.0
        self._hi: float = 1.0

    def start(self, lo: float, hi: float) -> None:
        raise NotImplementedError

    def next_candidate(self) -> float:
        raise NotImplementedError

    def update(self, candidate: float, passed: bool) -> None:
        raise NotImplementedError

    def current_bounds(self) -> tuple[float, float]:
        raise NotImplementedError

    def is_converged(self) -> bool:
        raise NotImplementedError


class UniformGridSearch(SearchAlgorithm):
    """Iterates through a uniform grid from hi→lo (descending) within [lo, hi].

    The grid is regenerated on each start() call based on the precision.
    Converges when all grid points within [lo, hi] have been visited.

    Descending order is preferred because for sparsity sweeps we want to
    probe the most aggressive (highest) feasible sparsity first.
    """

    def __init__(self, precision: float = 0.05) -> None:
        self._precision = precision
        self._grid: list[float] = []
        self._index: int = 0
        self._lo: float = 0.0
        self._hi: float = 1.0

    def start(self, lo: float, hi: float) -> None:
        raise NotImplementedError

    def next_candidate(self) -> float:
        raise NotImplementedError

    def update(self, candidate: float, passed: bool) -> None:
        raise NotImplementedError

    def current_bounds(self) -> tuple[float, float]:
        raise NotImplementedError

    def is_converged(self) -> bool:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# IntervalSpec
# ---------------------------------------------------------------------------


class IntervalSpec(ABC):
    """Resolves the [lo, hi] interval a stage will search.

    Called once per stage before the search loop begins.
    Must return an interval that is within the initial interval.
    """

    @abstractmethod
    def resolve(
        self,
        history: list[StageResult],
        initial: SparsityInterval,
    ) -> SparsityInterval:
        raise NotImplementedError


class Initial(IntervalSpec):
    """Always use the user-supplied initial interval (ignores history)."""

    def resolve(self, history: list[StageResult], initial: SparsityInterval) -> SparsityInterval:
        raise NotImplementedError


class ChainFromPrevious(IntervalSpec):
    """Use the previous stage's output interval as this stage's input.

    Raises IndexError if there is no previous stage.
    """

    def resolve(self, history: list[StageResult], initial: SparsityInterval) -> SparsityInterval:
        raise NotImplementedError


class BoundedByPreviousHi(IntervalSpec):
    """Use [initial.lo, history[stage_idx].output_interval.hi].

    Useful for a strict-threshold stage that should search everything below
    the loose stage's upper bound (e.g. 'search [0, hi_loose]').
    """

    def __init__(self, stage_idx: int) -> None:
        self._stage_idx = stage_idx

    def resolve(self, history: list[StageResult], initial: SparsityInterval) -> SparsityInterval:
        raise NotImplementedError


class SpanFromPreviousStages(IntervalSpec):
    """Use [history[lo_stage_idx].output_interval.lo, history[hi_stage_idx].output_interval.hi].

    Useful for a final refinement stage that should search the zone between
    a strict lower bound and a loose upper bound.
    """

    def __init__(self, lo_stage_idx: int, hi_stage_idx: int) -> None:
        self._lo_stage_idx = lo_stage_idx
        self._hi_stage_idx = hi_stage_idx

    def resolve(self, history: list[StageResult], initial: SparsityInterval) -> SparsityInterval:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# StepEvaluator
# ---------------------------------------------------------------------------


class StepEvaluator(ABC):
    """Evaluates a single candidate sparsity and returns a metric + success flag.

    Designed to be a black box: the caller does not need to know whether the
    internal recovery uses SFT, RL, or any other training scheme.

    Lifecycle per stage:
        evaluator.prepare(model_name_or_path, tokenizer, dataset_eval, device)
        for each candidate:
            result = evaluator.evaluate(sparsity, ..., dataset_recovery, ...)
    """

    @property
    @abstractmethod
    def metric_type(self) -> str:
        """'loss' or 'judge' — determines comparison direction in is_success."""
        raise NotImplementedError

    @abstractmethod
    def prepare(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        dataset_evaluation: Dataset,
        device: str,
    ) -> None:
        """One-time setup before the search loop (e.g. resolve fraction thresholds)."""
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
        """Run prune + optional recovery at the given sparsity; return metrics."""
        raise NotImplementedError


class PruneAndSFTRecoverEvaluator(StepEvaluator):
    """Evaluator that calls prune_and_maybe_recover (SFT-based recovery).

    Holds saliency map path, criterion, quality threshold, and SFT hyperparams.
    In prepare(), resolves fraction thresholds by running the unpruned model once.
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
        return self._metric_type

    def prepare(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase,
        dataset_evaluation: Dataset,
        device: str,
    ) -> None:
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
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SweepStage
# ---------------------------------------------------------------------------


class SweepStage(pydantic.BaseModel):
    """Configuration for one search stage.

    A stage is a complete search within an interval. It is a pure data container;
    the execution logic lives in run_staged_sweep.
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

    Returns a StagedSweepResult with per-stage history and the final interval.
    """
    raise NotImplementedError
