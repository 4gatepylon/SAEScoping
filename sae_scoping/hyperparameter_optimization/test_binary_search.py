import pytest
from sae_scoping.hyperparameter_optimization import (
    InvalidRangeError,
    NoProgressError,
    binary_search_step,
)


def test_finds_max_feasible_float():
    """Search for max x in [0, 1] where f(x) = 1 - x stays in [0.3, 1.0]."""
    lo, hi = binary_search_step(
        lo=0.0, hi=1.0,
        train_and_eval=lambda x: 1.0 - x,
        eval_lo=0.3, eval_hi=1.0,
        max_steps=20,
    )
    # Max feasible is 0.7; lo should converge near it
    assert lo > 0.6
    assert hi < 0.8
    assert lo < hi


def test_finds_max_feasible_int():
    """Integer mode: search [0, 100] where f(x) = 100 - x, want eval in [30, 100]."""
    lo, hi = binary_search_step(
        lo=0, hi=100,
        train_and_eval=lambda x: 100 - x,
        eval_lo=30, eval_hi=100,
        max_steps=20,
    )
    assert isinstance(lo, int)
    assert isinstance(hi, int)
    assert lo >= 68
    assert hi <= 72


def test_strictly_narrows():
    """Result must be strictly narrower than input."""
    lo, hi = binary_search_step(
        lo=0.0, hi=1.0,
        train_and_eval=lambda x: x,
        eval_lo=0.0, eval_hi=1.0,
        max_steps=1,
    )
    assert (lo > 0.0) or (hi < 1.0)


def test_error_on_invalid_range():
    with pytest.raises(InvalidRangeError):
        binary_search_step(1.0, 0.0, lambda x: x, 0.0, 1.0, max_steps=5)

    with pytest.raises(InvalidRangeError):
        binary_search_step(1.0, 1.0, lambda x: x, 0.0, 1.0, max_steps=5)


def test_error_on_zero_steps():
    with pytest.raises(InvalidRangeError):
        binary_search_step(0.0, 1.0, lambda x: x, 0.0, 1.0, max_steps=0)


def test_eval_too_low_narrows_hi():
    """If eval is always below eval_lo, hi should decrease."""
    lo, hi = binary_search_step(
        lo=0.0, hi=1.0,
        train_and_eval=lambda x: -1.0,
        eval_lo=0.0, eval_hi=1.0,
        max_steps=5,
    )
    assert hi < 1.0


def test_eval_too_high_narrows_lo():
    """If eval is always above eval_hi, lo should increase."""
    lo, hi = binary_search_step(
        lo=0.0, hi=1.0,
        train_and_eval=lambda x: 100.0,
        eval_lo=0.0, eval_hi=1.0,
        max_steps=5,
    )
    assert lo > 0.0


def test_int_no_progress_raises():
    """Int search with range [0, 1] can compute mid=0=lo, should raise."""
    with pytest.raises(NoProgressError):
        binary_search_step(
            lo=0, hi=1,
            train_and_eval=lambda x: x,
            eval_lo=0, eval_hi=100,
            max_steps=5,
        )
