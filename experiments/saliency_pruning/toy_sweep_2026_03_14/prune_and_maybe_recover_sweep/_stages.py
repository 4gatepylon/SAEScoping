"""SweepStage: data container for one stage's configuration."""

from __future__ import annotations

import pydantic

from prune_and_maybe_recover_sweep._evaluators import StepEvaluator
from prune_and_maybe_recover_sweep._intervals import IntervalSpec
from prune_and_maybe_recover_sweep._schemas import DatasetConfig
from prune_and_maybe_recover_sweep._search import SearchAlgorithm


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
    fires before max_steps, the loop exits early.  Reaching max_steps is
    not an error; the stage returns its best current bounds.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    name: str
    evaluator: StepEvaluator
    search: SearchAlgorithm
    interval_spec: IntervalSpec
    max_steps: int
    dataset_config: DatasetConfig
    raise_if_out_of_bounds: bool = True
