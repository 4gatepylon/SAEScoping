"""Mock-based unit tests for the staged sparsity sweep.

All tests run on CPU without model loading.  Mock implementations of the
abstract classes are defined here in the test module (per the project rule
that testing implementations belong in the test code).

Sections
--------
A  SearchAlgorithm (BinarySearch, UniformGridSearch) — pure logic
B  IntervalSpec resolvers — pure logic with StageResult fixtures
C  _run_stage — single-stage orchestration using MockEvaluator
D  Multi-stage scenarios (2-stage chaining, 3-stage complex intervals)
E  Error / edge cases (all-pass, all-fail, immediate convergence, 1-step)
"""

from __future__ import annotations

import math
from typing import Optional

import pytest

from prune_and_maybe_recover_sweep import (
    BinarySearch,
    BoundedByPreviousHi,
    ChainFromPrevious,
    DatasetConfig,
    Initial,
    LeftOutOfBoundsError,
    RightOutOfBoundsError,
    SpanFromPreviousStages,
    SparsityInterval,
    StageResult,
    StepEvalResult,
    SweepStage,
    UniformGridSearch,
    _run_stage,
)
from prune_and_maybe_recover_sweep._evaluators import StepEvaluator
from prune_and_maybe_recover_sweep._intervals import IntervalSpec
from prune_and_maybe_recover_sweep._search import SearchAlgorithm


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def simulate_search(
    search: SearchAlgorithm,
    lo: float,
    hi: float,
    feasible_fn: "callable[[float], bool]",
    max_steps: int,
) -> tuple[float, float]:
    """Drive a SearchAlgorithm to completion using a predefined feasibility function.

    Returns current_bounds() after convergence or max_steps.
    """
    search.start(lo, hi)
    for _ in range(max_steps):
        if search.is_converged():
            break
        candidate = search.next_candidate()
        search.update(candidate, feasible_fn(candidate))
    return search.current_bounds()


def make_stage_result(
    name: str,
    lo: float,
    hi: float,
    input_lo: Optional[float] = None,
    input_hi: Optional[float] = None,
) -> StageResult:
    """Create a StageResult fixture with the given output interval."""
    return StageResult(
        name=name,
        input_interval=SparsityInterval(
            lo=input_lo if input_lo is not None else lo,
            hi=input_hi if input_hi is not None else hi,
        ),
        output_interval=SparsityInterval(lo=lo, hi=hi),
        steps=[],
    )


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------


class _MockEvaluator(StepEvaluator):
    """Mock evaluator: passes when sparsity <= threshold (for loss-style metric).

    The feasibility function is: passes when sparsity <= threshold.
    metric_before = metric_after = sparsity (simplified).
    """

    def __init__(self, threshold: float, metric_type: str = "loss") -> None:
        self._threshold = threshold
        self._type = metric_type
        self.calls: list[float] = []

    @property
    def metric_type(self) -> str:
        return self._type

    def prepare(self, model_name_or_path, tokenizer, dataset_evaluation, device) -> None:
        pass

    def evaluate(
        self,
        sparsity,
        model_name_or_path,
        tokenizer,
        dataset_evaluation,
        dataset_recovery,
        device,
        output_dir,
    ) -> StepEvalResult:
        self.calls.append(sparsity)
        passes = sparsity <= self._threshold
        return StepEvalResult(
            sparsity=sparsity,
            metric_before=sparsity,
            metric_after=sparsity,
            is_success=passes,
        )


class _AlwaysPassEvaluator(_MockEvaluator):
    """Passes for every sparsity value."""

    def __init__(self) -> None:
        super().__init__(threshold=1.0)

    def evaluate(self, sparsity, model_name_or_path, tokenizer, dataset_evaluation,
                 dataset_recovery, device, output_dir) -> StepEvalResult:
        self.calls.append(sparsity)
        return StepEvalResult(
            sparsity=sparsity, metric_before=sparsity, metric_after=sparsity, is_success=True,
        )


class _AlwaysFailEvaluator(_MockEvaluator):
    """Fails for every sparsity value."""

    def __init__(self) -> None:
        super().__init__(threshold=-1.0)

    def evaluate(self, sparsity, model_name_or_path, tokenizer, dataset_evaluation,
                 dataset_recovery, device, output_dir) -> StepEvalResult:
        self.calls.append(sparsity)
        return StepEvalResult(
            sparsity=sparsity, metric_before=sparsity, metric_after=sparsity, is_success=False,
        )


_DUMMY_DATASET_CONFIG = DatasetConfig(n_eval=2, n_recovery=2)


def _make_stage(
    name: str,
    evaluator: StepEvaluator,
    search: SearchAlgorithm,
    interval_spec: IntervalSpec,
    max_steps: int = 20,
    raise_if_out_of_bounds: bool = True,
) -> SweepStage:
    return SweepStage(
        name=name,
        evaluator=evaluator,
        search=search,
        interval_spec=interval_spec,
        max_steps=max_steps,
        dataset_config=_DUMMY_DATASET_CONFIG,
        raise_if_out_of_bounds=raise_if_out_of_bounds,
    )


# _run_stage shared args (no real model/tokenizer/dataset needed — mock evaluator ignores them)
_FAKE_MODEL = "FAKE_MODEL"
_FAKE_TOKENIZER = None
_FAKE_DATASET = None
_FAKE_DEVICE = "cpu"
_FAKE_DIR = "/tmp/test_staged_sweep"


# ---------------------------------------------------------------------------
# Section A: BinarySearch algorithm
# ---------------------------------------------------------------------------


def test_binary_search_midpoint() -> None:
    """next_candidate must return the exact midpoint of current [lo, hi]."""
    search = BinarySearch()
    search.start(0.0, 1.0)
    assert search.next_candidate() == pytest.approx(0.5)

    search.start(0.2, 0.6)
    assert search.next_candidate() == pytest.approx(0.4)


def test_binary_search_pass_moves_lo() -> None:
    """A passing result must raise lo to the candidate."""
    search = BinarySearch()
    search.start(0.0, 1.0)
    mid = search.next_candidate()  # 0.5
    search.update(mid, passed=True)
    lo, hi = search.current_bounds()
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(1.0)


def test_binary_search_fail_moves_hi() -> None:
    """A failing result must lower hi to the candidate."""
    search = BinarySearch()
    search.start(0.0, 1.0)
    mid = search.next_candidate()  # 0.5
    search.update(mid, passed=False)
    lo, hi = search.current_bounds()
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(0.5)


def test_binary_search_converges_to_threshold() -> None:
    """BinarySearch on step-function feasible_fn converges within tolerance."""
    true_boundary = 0.7

    def feasible(s: float) -> bool:
        return s <= true_boundary

    lo, hi = simulate_search(BinarySearch(tolerance=1e-4), 0.0, 1.0, feasible, max_steps=100)
    assert lo <= true_boundary <= hi
    assert hi - lo < 1e-3


def test_binary_search_already_converged_at_equal_bounds() -> None:
    """When lo == hi, is_converged must return True immediately after start."""
    search = BinarySearch(tolerance=1e-6)
    search.start(0.5, 0.5)
    assert search.is_converged()


def test_binary_search_resets_on_start() -> None:
    """Calling start() a second time must reset state to the new [lo, hi]."""
    search = BinarySearch()
    search.start(0.0, 1.0)
    search.next_candidate()
    search.update(0.5, True)

    search.start(0.2, 0.4)
    assert search.current_bounds() == pytest.approx((0.2, 0.4))
    assert not search.is_converged()


# ---------------------------------------------------------------------------
# Section B: UniformGridSearch algorithm
# ---------------------------------------------------------------------------


def test_uniform_grid_descending_order() -> None:
    """Grid points must be iterated from hi to lo (descending)."""
    search = UniformGridSearch(precision=0.2)
    search.start(0.0, 0.6)
    candidates = []
    for _ in range(10):
        if search.is_converged():
            break
        c = search.next_candidate()
        candidates.append(c)
        search.update(c, False)
    assert candidates == sorted(candidates, reverse=True)


def test_uniform_grid_all_points_visited() -> None:
    """All grid points in [lo, hi] must be visited (no skipping)."""
    precision = 0.1
    search = UniformGridSearch(precision=precision)
    search.start(0.0, 0.5)
    visited = []
    for _ in range(20):
        if search.is_converged():
            break
        c = search.next_candidate()
        visited.append(round(c, 8))
        search.update(c, False)
    expected = sorted({round(k * precision, 8) for k in range(6)}, reverse=True)
    assert visited == expected


def test_uniform_grid_converges_after_all_visited() -> None:
    """is_converged must return True once all grid points have been visited."""
    search = UniformGridSearch(precision=0.5)
    search.start(0.0, 1.0)
    steps = 0
    while not search.is_converged():
        search.update(search.next_candidate(), False)
        steps += 1
    assert steps == 3  # 0.0, 0.5, 1.0


def test_uniform_grid_empty_range() -> None:
    """When [lo, hi] is so small no grid points fit, search is immediately converged."""
    search = UniformGridSearch(precision=0.2)
    search.start(0.11, 0.19)
    assert search.is_converged()


def test_uniform_grid_bound_updates() -> None:
    """Bounds should narrow like BinarySearch after each update."""
    search = UniformGridSearch(precision=0.1)
    search.start(0.0, 1.0)

    c = search.next_candidate()  # 1.0 (descending)
    search.update(c, False)
    _, hi = search.current_bounds()
    assert hi == pytest.approx(1.0)

    c2 = search.next_candidate()  # 0.9
    search.update(c2, True)
    lo, _ = search.current_bounds()
    assert lo == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Section C: IntervalSpec resolvers
# ---------------------------------------------------------------------------


def test_initial_returns_initial() -> None:
    """Initial() must always return the user-supplied initial interval."""
    spec = Initial()
    initial = SparsityInterval(lo=0.1, hi=0.9)
    assert spec.resolve([], initial) == initial
    history = [make_stage_result("s0", 0.3, 0.6)]
    assert spec.resolve(history, initial) == initial


def test_chain_from_previous_returns_last_output() -> None:
    """ChainFromPrevious() must return the last stage's output_interval."""
    spec = ChainFromPrevious()
    history = [
        make_stage_result("s0", 0.3, 0.7),
        make_stage_result("s1", 0.4, 0.6),
    ]
    result = spec.resolve(history, SparsityInterval(lo=0.0, hi=1.0))
    assert result.lo == pytest.approx(0.4)
    assert result.hi == pytest.approx(0.6)


def test_chain_from_previous_raises_on_empty_history() -> None:
    """ChainFromPrevious() must raise IndexError when history is empty."""
    spec = ChainFromPrevious()
    with pytest.raises(IndexError):
        spec.resolve([], SparsityInterval(lo=0.0, hi=1.0))


def test_bounded_by_previous_hi() -> None:
    """BoundedByPreviousHi(0) must use [initial.lo, history[0].output_interval.hi]."""
    spec = BoundedByPreviousHi(stage_idx=0)
    initial = SparsityInterval(lo=0.0, hi=1.0)
    history = [make_stage_result("s0", 0.4, 0.75)]
    result = spec.resolve(history, initial)
    assert result.lo == pytest.approx(0.0)
    assert result.hi == pytest.approx(0.75)


def test_span_from_previous_stages() -> None:
    """SpanFromPreviousStages(lo_idx, hi_idx) spans the two stages' bounds."""
    spec = SpanFromPreviousStages(lo_stage_idx=1, hi_stage_idx=0)
    initial = SparsityInterval(lo=0.0, hi=1.0)
    history = [
        make_stage_result("loose", 0.5, 0.80),   # hi=0.80
        make_stage_result("strict", 0.55, 0.65),  # lo=0.55
    ]
    result = spec.resolve(history, initial)
    assert result.lo == pytest.approx(0.55)
    assert result.hi == pytest.approx(0.80)


def test_span_from_previous_stages_raises_if_lo_gt_hi() -> None:
    """SpanFromPreviousStages must raise ValueError when resolved lo > hi."""
    spec = SpanFromPreviousStages(lo_stage_idx=0, hi_stage_idx=1)
    initial = SparsityInterval(lo=0.0, hi=1.0)
    history = [
        make_stage_result("s0", 0.7, 0.9),  # lo=0.7
        make_stage_result("s1", 0.2, 0.4),  # hi=0.4
    ]
    with pytest.raises(ValueError):
        spec.resolve(history, initial)


def test_bounded_by_previous_hi_raises_on_missing_index() -> None:
    """BoundedByPreviousHi(5) must raise IndexError when history has fewer stages."""
    spec = BoundedByPreviousHi(stage_idx=5)
    with pytest.raises(IndexError):
        spec.resolve([], SparsityInterval(lo=0.0, hi=1.0))


# ---------------------------------------------------------------------------
# Section D: _run_stage — single-stage orchestration
# ---------------------------------------------------------------------------


def test_run_stage_binary_narrows_interval() -> None:
    """_run_stage with BinarySearch and threshold=0.7 must produce lo<=0.7<=hi."""
    evaluator = _MockEvaluator(threshold=0.7)
    stage = _make_stage("s0", evaluator, BinarySearch(tolerance=1e-3), Initial(), max_steps=30)
    interval = SparsityInterval(lo=0.0, hi=1.0)

    result = _run_stage(
        stage, interval, _FAKE_MODEL, _FAKE_TOKENIZER,
        _FAKE_DATASET, _FAKE_DATASET, _FAKE_DEVICE, _FAKE_DIR,
    )

    assert result.output_interval.lo <= 0.7
    assert result.output_interval.hi >= 0.7
    assert result.output_interval.hi - result.output_interval.lo < 0.01
    assert len(result.steps) > 0


def test_run_stage_uniform_narrows_interval() -> None:
    """_run_stage with UniformGridSearch and threshold=0.6 must bracket the boundary."""
    evaluator = _MockEvaluator(threshold=0.6)
    stage = _make_stage(
        "s0", evaluator, UniformGridSearch(precision=0.1), Initial(), max_steps=50,
    )
    interval = SparsityInterval(lo=0.0, hi=1.0)

    result = _run_stage(
        stage, interval, _FAKE_MODEL, _FAKE_TOKENIZER,
        _FAKE_DATASET, _FAKE_DATASET, _FAKE_DEVICE, _FAKE_DIR,
    )

    assert result.output_interval.lo <= 0.6
    assert result.output_interval.hi >= 0.6


def test_run_stage_max_steps_respected() -> None:
    """_run_stage must stop after max_steps evaluations."""
    evaluator = _MockEvaluator(threshold=0.5)
    stage = _make_stage("s0", evaluator, BinarySearch(tolerance=1e-9), Initial(), max_steps=5)
    interval = SparsityInterval(lo=0.0, hi=1.0)

    result = _run_stage(
        stage, interval, _FAKE_MODEL, _FAKE_TOKENIZER,
        _FAKE_DATASET, _FAKE_DATASET, _FAKE_DEVICE, _FAKE_DIR,
    )

    assert len(result.steps) <= 5


def test_run_stage_steps_record_sparsity() -> None:
    """Each step in stage.steps must record the sparsity that was evaluated."""
    evaluator = _MockEvaluator(threshold=0.5)
    stage = _make_stage("s0", evaluator, BinarySearch(tolerance=0.01), Initial(), max_steps=20)
    interval = SparsityInterval(lo=0.0, hi=1.0)

    result = _run_stage(
        stage, interval, _FAKE_MODEL, _FAKE_TOKENIZER,
        _FAKE_DATASET, _FAKE_DATASET, _FAKE_DEVICE, _FAKE_DIR,
    )

    for step in result.steps:
        assert 0.0 <= step.sparsity <= 1.0


def test_run_stage_all_fail_raises_left_out_of_bounds() -> None:
    """_run_stage must raise LeftOutOfBoundsError when all evaluations fail."""
    evaluator = _AlwaysFailEvaluator()
    stage = _make_stage(
        "s0", evaluator, BinarySearch(tolerance=1e-3), Initial(),
        max_steps=10, raise_if_out_of_bounds=True,
    )
    interval = SparsityInterval(lo=0.0, hi=1.0)

    with pytest.raises(LeftOutOfBoundsError):
        _run_stage(
            stage, interval, _FAKE_MODEL, _FAKE_TOKENIZER,
            _FAKE_DATASET, _FAKE_DATASET, _FAKE_DEVICE, _FAKE_DIR,
        )


def test_run_stage_all_pass_raises_right_out_of_bounds() -> None:
    """_run_stage must raise RightOutOfBoundsError when all evaluations pass."""
    evaluator = _AlwaysPassEvaluator()
    stage = _make_stage(
        "s0", evaluator, BinarySearch(tolerance=1e-3), Initial(),
        max_steps=10, raise_if_out_of_bounds=True,
    )
    interval = SparsityInterval(lo=0.0, hi=1.0)

    with pytest.raises(RightOutOfBoundsError):
        _run_stage(
            stage, interval, _FAKE_MODEL, _FAKE_TOKENIZER,
            _FAKE_DATASET, _FAKE_DATASET, _FAKE_DEVICE, _FAKE_DIR,
        )


def test_run_stage_all_fail_no_raise_emits_warning() -> None:
    """With raise_if_out_of_bounds=False, all-fail should warn and return a result."""
    evaluator = _AlwaysFailEvaluator()
    stage = _make_stage(
        "s0", evaluator, BinarySearch(tolerance=1e-3), Initial(),
        max_steps=5, raise_if_out_of_bounds=False,
    )
    interval = SparsityInterval(lo=0.0, hi=1.0)

    with pytest.warns(UserWarning):
        result = _run_stage(
            stage, interval, _FAKE_MODEL, _FAKE_TOKENIZER,
            _FAKE_DATASET, _FAKE_DATASET, _FAKE_DEVICE, _FAKE_DIR,
        )

    assert isinstance(result, StageResult)


def test_run_stage_all_pass_no_raise_emits_warning() -> None:
    """With raise_if_out_of_bounds=False, all-pass should warn and return a result."""
    evaluator = _AlwaysPassEvaluator()
    stage = _make_stage(
        "s0", evaluator, BinarySearch(tolerance=1e-3), Initial(),
        max_steps=5, raise_if_out_of_bounds=False,
    )
    interval = SparsityInterval(lo=0.0, hi=1.0)

    with pytest.warns(UserWarning):
        result = _run_stage(
            stage, interval, _FAKE_MODEL, _FAKE_TOKENIZER,
            _FAKE_DATASET, _FAKE_DATASET, _FAKE_DEVICE, _FAKE_DIR,
        )

    assert isinstance(result, StageResult)


# ---------------------------------------------------------------------------
# Section E: Multi-stage scenarios
# ---------------------------------------------------------------------------


def test_two_stage_chain_from_previous() -> None:
    """Two stages chained: stage 1's interval is stage 0's output."""
    history: list[StageResult] = []

    # Stage 0: loose threshold 0.8
    evaluator0 = _MockEvaluator(threshold=0.8)
    stage0 = _make_stage("loose", evaluator0, BinarySearch(tolerance=0.02), Initial(), max_steps=20)
    interval0 = SparsityInterval(lo=0.0, hi=1.0)
    result0 = _run_stage(
        stage0, interval0, _FAKE_MODEL, _FAKE_TOKENIZER,
        _FAKE_DATASET, _FAKE_DATASET, _FAKE_DEVICE, _FAKE_DIR,
    )
    history.append(result0)

    # Stage 1: strict threshold 0.6, interval from previous output
    evaluator1 = _MockEvaluator(threshold=0.6)
    stage1 = _make_stage(
        "strict", evaluator1, BinarySearch(tolerance=0.02), ChainFromPrevious(), max_steps=20,
    )
    interval1 = stage1.interval_spec.resolve(history, interval0)
    result1 = _run_stage(
        stage1, interval1, _FAKE_MODEL, _FAKE_TOKENIZER,
        _FAKE_DATASET, _FAKE_DATASET, _FAKE_DEVICE, _FAKE_DIR,
    )

    # The strict search is within the loose output interval
    assert interval1.lo >= result0.output_interval.lo
    assert interval1.hi <= result0.output_interval.hi

    # The strict result brackets the strict threshold
    assert result1.output_interval.lo <= 0.6
    assert result1.output_interval.hi >= 0.6


def test_three_stage_span_from_previous() -> None:
    """Three-stage sweep: stage 2 spans [strict_lo, loose_hi]."""
    initial = SparsityInterval(lo=0.0, hi=1.0)
    history: list[StageResult] = []

    # Stage 0: loose threshold 0.8 → hi around 0.8
    evaluator0 = _MockEvaluator(threshold=0.8)
    stage0 = _make_stage("loose", evaluator0, BinarySearch(tolerance=0.02), Initial(), max_steps=20)
    result0 = _run_stage(
        stage0, initial, _FAKE_MODEL, _FAKE_TOKENIZER,
        _FAKE_DATASET, _FAKE_DATASET, _FAKE_DEVICE, _FAKE_DIR,
    )
    history.append(result0)

    # Stage 1: strict threshold 0.6, bounded by stage 0's hi
    evaluator1 = _MockEvaluator(threshold=0.6)
    spec1 = BoundedByPreviousHi(stage_idx=0)
    stage1 = _make_stage("strict", evaluator1, BinarySearch(tolerance=0.02), spec1, max_steps=20)
    interval1 = stage1.interval_spec.resolve(history, initial)
    result1 = _run_stage(
        stage1, interval1, _FAKE_MODEL, _FAKE_TOKENIZER,
        _FAKE_DATASET, _FAKE_DATASET, _FAKE_DEVICE, _FAKE_DIR,
    )
    history.append(result1)

    # Stage 2: spans [strict lo, loose hi]
    evaluator2 = _MockEvaluator(threshold=0.7)
    spec2 = SpanFromPreviousStages(lo_stage_idx=1, hi_stage_idx=0)
    stage2 = _make_stage(
        "refine", evaluator2, UniformGridSearch(precision=0.02), spec2, max_steps=30,
    )
    interval2 = stage2.interval_spec.resolve(history, initial)

    # The refinement interval must span from strict lo to loose hi
    assert interval2.lo == pytest.approx(result1.output_interval.lo, abs=0.01)
    assert interval2.hi == pytest.approx(result0.output_interval.hi, abs=0.01)

    result2 = _run_stage(
        stage2, interval2, _FAKE_MODEL, _FAKE_TOKENIZER,
        _FAKE_DATASET, _FAKE_DATASET, _FAKE_DEVICE, _FAKE_DIR,
    )

    assert result2.output_interval.lo <= 0.7 <= result2.output_interval.hi


def test_one_step_max_steps_single_candidate() -> None:
    """A stage with max_steps=1 must evaluate exactly one candidate."""
    evaluator = _MockEvaluator(threshold=0.5)
    stage = _make_stage("s0", evaluator, BinarySearch(tolerance=1e-9), Initial(), max_steps=1)
    interval = SparsityInterval(lo=0.0, hi=1.0)

    result = _run_stage(
        stage, interval, _FAKE_MODEL, _FAKE_TOKENIZER,
        _FAKE_DATASET, _FAKE_DATASET, _FAKE_DEVICE, _FAKE_DIR,
    )

    assert len(result.steps) == 1
    assert result.steps[0].sparsity == pytest.approx(0.5)
