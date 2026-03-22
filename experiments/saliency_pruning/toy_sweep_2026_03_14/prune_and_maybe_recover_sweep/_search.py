"""Search algorithm abstractions and concrete implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod


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
    outcome, update() MAY keep bounds stable to signal stagnation.  Current
    implementations do not implement this; they collapse normally.

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
        Convergence threshold (default 1e-6).
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
        self._lo = lo
        self._hi = hi

    def next_candidate(self) -> float:
        """Return midpoint of current [lo, hi].

        Postcondition: returned value == (lo + hi) / 2
        """
        return (self._lo + self._hi) / 2.0

    def update(self, candidate: float, passed: bool) -> None:
        """Narrow bounds: passed → lo = candidate; failed → hi = candidate.

        Postcondition: bounds strictly narrowed (unless at tolerance)
        """
        if passed:
            self._lo = candidate
        else:
            self._hi = candidate

    def current_bounds(self) -> tuple[float, float]:
        """Return (self._lo, self._hi)."""
        return (self._lo, self._hi)

    def is_converged(self) -> bool:
        """Return self._hi - self._lo < self._tolerance."""
        return (self._hi - self._lo) < self._tolerance


class UniformGridSearch(SearchAlgorithm):
    """Iterates through a uniform precision grid from hi→lo (descending).

    The grid is generated within [lo, hi] on start().  Each step visits the
    next unvisited grid point.  Bounds (lo, hi) are updated after each step
    using the same rule as BinarySearch (passed → lo=candidate, failed →
    hi=candidate) for consistency.

    Descending order is preferred for sparsity sweeps (find highest feasible
    sparsity first).

    Converges when all grid points have been visited.

    Parameters
    ----------
    precision : float
        Grid step size (default 0.05).
    """

    def __init__(self, precision: float = 0.05) -> None:
        self._precision = precision
        self._grid: list[float] = []
        self._index: int = 0
        self._lo: float = 0.0
        self._hi: float = 1.0

    def start(self, lo: float, hi: float) -> None:
        """Generate a descending grid within [lo, hi] and reset the index.

        Grid points are all multiples of precision within [lo, hi], sorted
        descending (hi → lo).

        Precondition:  lo <= hi; precision > 0
        Postcondition: _grid contains all k such that lo <= k*precision <= hi;
                       _index = 0
        """
        self._lo = lo
        self._hi = hi
        self._index = 0
        step = self._precision
        k_lo = int(lo / step)
        k_hi = int(hi / step + 1e-9)
        self._grid = sorted(
            [k * step for k in range(k_lo, k_hi + 1) if lo - 1e-9 <= k * step <= hi + 1e-9],
            reverse=True,
        )

    def next_candidate(self) -> float:
        """Return the next unvisited grid point.

        Precondition:  not converged
        Postcondition: returned value == _grid[_index]; lo <= value <= hi
        """
        return self._grid[self._index]

    def update(self, candidate: float, passed: bool) -> None:
        """Advance index; update lo/hi bounds as in BinarySearch.

        Precondition:  candidate == _grid[_index]
        Postcondition: _index incremented by 1; bounds updated
        """
        self._index += 1
        if passed:
            self._lo = candidate
        else:
            self._hi = candidate

    def current_bounds(self) -> tuple[float, float]:
        """Return (self._lo, self._hi)."""
        return (self._lo, self._hi)

    def is_converged(self) -> bool:
        """Return True when _index >= len(_grid)."""
        return self._index >= len(self._grid)
