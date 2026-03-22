"""Exception hierarchy for staged sparsity sweep."""

from __future__ import annotations


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
    """


class RightOutOfBoundsError(OutOfBoundsError):
    """Every candidate in [lo, hi] passes.

    Interpretation: the feasible region extends entirely to the right of
    (above) the search range.  The model meets the quality threshold even at
    hi after recovery.
    """
