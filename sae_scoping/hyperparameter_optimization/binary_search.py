"""Binary search over a hyperparameter range to find the maximum value
whose train_and_eval output falls within acceptable bounds."""

from typing import Callable, Tuple, Union

Numeric = Union[int, float]


class BinarySearchError(Exception):
    """Base exception for binary search errors."""


class InvalidRangeError(BinarySearchError):
    """Raised when lo >= hi."""


class NoProgressError(BinarySearchError):
    """Raised when the search fails to narrow the bounds."""


def binary_search_step(
    lo: Numeric,
    hi: Numeric,
    train_and_eval: Callable[[Numeric], float],
    eval_lo: float,
    eval_hi: float,
    max_steps: int,
) -> Tuple[Numeric, Numeric]:
    """Find the largest hyperparameter in [lo, hi] where train_and_eval(x) is in [eval_lo, eval_hi].

    Uses binary search: evaluate midpoint, narrow bounds based on whether
    the eval result is within the acceptable range.

    Args:
        lo: Lower bound of hyperparameter range.
        hi: Upper bound of hyperparameter range.
        train_and_eval: Function mapping a hyperparameter value to an eval metric.
        eval_lo: Lower acceptable bound on the eval metric.
        eval_hi: Upper acceptable bound on the eval metric.
        max_steps: Maximum number of binary search steps.

    Returns:
        A strictly narrower (lo, hi) tuple.

    Raises:
        InvalidRangeError: If lo >= hi or max_steps <= 0.
        NoProgressError: If the search cannot narrow the bounds.
    """
    if lo >= hi:
        raise InvalidRangeError(f"lo={lo} >= hi={hi}")
    if max_steps <= 0:
        raise InvalidRangeError("max_steps must be > 0")

    use_int = isinstance(lo, int) and isinstance(hi, int)
    orig_lo, orig_hi = lo, hi

    for _ in range(max_steps):
        if use_int:
            mid = (lo + hi) // 2
            if mid == lo:  # can't subdivide further for ints
                break
        else:
            mid = (lo + hi) / 2.0

        val = train_and_eval(mid)

        if val < eval_lo:
            # Eval too low — hyperparameter too aggressive, search lower half
            hi = mid
        elif val > eval_hi:
            # Eval too high — hyperparameter too conservative, search upper half
            lo = mid
        else:
            # Within bounds — try to push higher (find max feasible)
            lo = mid

    if lo == orig_lo and hi == orig_hi:
        raise NoProgressError("Binary search made no progress — bounds unchanged")

    return lo, hi
