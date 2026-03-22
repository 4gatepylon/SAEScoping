"""Pydantic schemas for staged sparsity sweep inputs and outputs."""

from __future__ import annotations

from typing import Any

import pydantic


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
    extra           : arbitrary stage-specific diagnostics
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
    sizes.

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

    Specific to PruneAndSFTRecoverEvaluator; other evaluator types define
    their own config schema.

    Fields
    ------
    max_steps           : maximum SFT gradient steps per candidate
    eval_every          : evaluate quality metric every N steps
    batch_size          : per-device training batch size
    learning_rate       : SFT learning rate
    max_seq_len         : maximum tokenised sequence length
    max_new_tokens      : maximum generation tokens (judge metric only)
    give_up_thresholds  : abort recovery early if quality not met by N steps
    """

    max_steps: int = 500
    eval_every: int = 50
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    max_seq_len: int = 1024
    max_new_tokens: int = 256
    give_up_thresholds: list[Any] = []
