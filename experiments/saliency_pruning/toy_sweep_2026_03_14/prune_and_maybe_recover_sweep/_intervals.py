"""IntervalSpec: resolvers that pick [lo, hi] for each stage from run history."""

from __future__ import annotations

from abc import ABC, abstractmethod

from prune_and_maybe_recover_sweep._schemas import SparsityInterval, StageResult


class IntervalSpec(ABC):
    """Resolves the [lo, hi] interval for a stage given the run history.

    Contract
    --------
    - resolve() is called exactly once per stage, before start().
    - The returned interval must satisfy:
        initial.lo <= result.lo <= result.hi <= initial.hi
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
        return initial


class ChainFromPrevious(IntervalSpec):
    """Use the immediately preceding stage's output_interval as this stage's input.

    Precondition: len(history) >= 1
    Raises IndexError if history is empty.
    """

    def resolve(self, history: list[StageResult], initial: SparsityInterval) -> SparsityInterval:
        """Return history[-1].output_interval.

        Precondition:  len(history) >= 1
        Postcondition: result == history[-1].output_interval
        """
        if not history:
            raise IndexError("ChainFromPrevious requires at least one preceding stage result")
        return history[-1].output_interval


class BoundedByPreviousHi(IntervalSpec):
    """Use [initial.lo, history[stage_idx].output_interval.hi].

    Useful for a stricter stage that needs to search from the initial lower
    bound up to an earlier stage's upper bound.

    Example: Stage 0 (loose) finds hi=0.70.  Stage 1 with
    BoundedByPreviousHi(0) searches [initial.lo, 0.70].

    Precondition:  len(history) > stage_idx
    Raises IndexError if stage_idx is out of range.
    """

    def __init__(self, stage_idx: int) -> None:
        self._stage_idx = stage_idx

    def resolve(self, history: list[StageResult], initial: SparsityInterval) -> SparsityInterval:
        """Return SparsityInterval(lo=initial.lo, hi=history[stage_idx].output_interval.hi).

        Precondition:  len(history) > self._stage_idx
        """
        return SparsityInterval(
            lo=initial.lo,
            hi=history[self._stage_idx].output_interval.hi,
        )


class SpanFromPreviousStages(IntervalSpec):
    """Use [history[lo_stage_idx].output_interval.lo, history[hi_stage_idx].output_interval.hi].

    Useful for a refinement stage spanning from one stage's lower bound to
    another stage's upper bound.

    Example: Stage 0 (loose) hi=0.70; Stage 1 (strict) lo=0.51.
    Stage 2 with SpanFromPreviousStages(1, 0) searches [0.51, 0.70].

    Precondition:  both indices in range; resolved lo <= hi
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
        lo = history[self._lo_stage_idx].output_interval.lo
        hi = history[self._hi_stage_idx].output_interval.hi
        if lo > hi:
            raise ValueError(
                f"SpanFromPreviousStages resolved lo={lo} > hi={hi}: "
                "check stage indices and ensure stages have narrowed properly"
            )
        return SparsityInterval(lo=lo, hi=hi)
