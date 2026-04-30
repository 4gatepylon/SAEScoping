"""Hierarchical pydantic-yaml config schema for the pruning sweep script.

Top-level: `SweepConfig`. Loaded from YAML by `SweepConfig.from_yaml(path)`.
Each sub-section is a separate pydantic model so the field path tells you
where every setting slots in (e.g. `calibration.n_calibration`,
`sweep.nn_linear_sparsities`, `operational.wandb.project`,
`operational.llm_judge.judge_model`, `pgd.learning_rate`).

CLI flags on the runner remain the most-likely-to-change overrides
(--device, -s, --artifacts-dir, --enable-wandb, etc.). Defaults live here
so a YAML file can be empty / minimal and still produce a valid run.

The `pgd` sub-config currently has its parameter surface ready but is NOT
yet wired into the runner — see the TODO in sweep_wanda.py.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic_yaml import parse_yaml_file_as


class _Frozen(BaseModel):
    """Common base: forbid extra fields so typos don't silently drop."""

    model_config = ConfigDict(extra="forbid")


# ── Sub-configs ───────────────────────────────────────────────────────────


class CalibrationConfig(_Frozen):
    """Data + tokenization knobs used to compute Wanda saliency."""

    n_calibration: int = 128
    max_seq_len: int = 2048
    batch_size: int = 1


class PruningSweepConfig(_Frozen):
    """The sparsity sweep itself (and per-sparsity loss eval)."""

    nn_linear_sparsities: list[float] = Field(default_factory=lambda: [0.5])
    n_eval: int = 64


class WandbConfig(_Frozen):
    """W&B logging. Resolution at runtime: arg > env > default (see wandb_utils)."""

    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    mode: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[str] = None  # comma-separated; parsed at init time


class LLMJudgeConfig(_Frozen):
    """LLM-judge evaluation per sparsity step."""

    enabled: bool = False
    domains: Optional[list[str]] = None  # None → use [SweepConfig.dataset_subset]
    n_samples: int = 50
    judge_model: str = "gpt-4.1-nano"
    split: str = "validation"


class OperationalConfig(_Frozen):
    """Where the experiment runs and where outputs go.

    Holds device + paths + the two logging sub-systems (W&B, LLM judge).
    """

    device: str = "cuda:0"
    artifacts_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    no_cache: bool = False
    low_memory: bool = False

    wandb: WandbConfig = Field(default_factory=WandbConfig)
    llm_judge: LLMJudgeConfig = Field(default_factory=LLMJudgeConfig)


class PGDConfig(_Frozen):
    """PGD recovery training (projected SFT keeping zeroed weights at zero).

    `save_steps`: when > 0, the SFT trainer writes a model checkpoint every
    `save_steps` optimizer steps. Checkpoints land under
    `<artifacts_root>/outputs/<run_id>/step_NNN/recovery/checkpoints/`
    (the runner overrides SFTConfig.output_dir to that path; this field
    only controls the *cadence*). Set to 0 to disable checkpointing.

    Note: there is intentionally no `output_dir` field here — the
    checkpoint location is fully derived from the run's artifacts dir so
    every run is self-contained.
    """

    enabled: bool = False
    n_train: int = 2000
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    max_steps: int = -1
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    logging_steps: int = 10
    eval_every_steps: int = 50
    save_steps: int = 0  # 0 = no checkpointing; >0 = save every N optimizer steps
    validate_sparsity: bool = True
    report_to: str = "none"


# ── Top-level ─────────────────────────────────────────────────────────────


class SweepConfig(_Frozen):
    """Top-level sweep config. Load from YAML; CLI flags override.

    Layout:
        model_id, dataset_name, dataset_subset      # top-level (shared by everything)
        calibration: CalibrationConfig              # for Wanda saliency
        sweep:       PruningSweepConfig             # sparsities + n_eval
        operational: OperationalConfig              # device, paths, wandb, llm-judge
        pgd:         PGDConfig                      # stub; commit 4 wires it up
    """

    model_id: str = "google/gemma-3-4b-it"
    dataset_name: str = "4gate/StemQAMixture"
    dataset_subset: str = "biology"

    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    sweep: PruningSweepConfig = Field(default_factory=PruningSweepConfig)
    operational: OperationalConfig = Field(default_factory=OperationalConfig)
    pgd: PGDConfig = Field(default_factory=PGDConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "SweepConfig":
        return parse_yaml_file_as(cls, path)
