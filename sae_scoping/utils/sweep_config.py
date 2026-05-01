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
    """PGD recovery training.

    NOT YET WIRED UP. Fields here mirror the click flags of the legacy
    `sweep_wanda_with_pgd_recovery.py`; commit 4 (PGD merge) will read
    them from this sub-config. Until then the runner ignores `pgd`.
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
    output_dir: str = "./outputs_pgd"
    validate_sparsity: bool = True
    report_to: str = "none"
    # Optimizer + memory knobs. For ≥9B models on a single 80GB GPU, fp32
    # Adam state alone is ~72GB so the default "adamw_torch" OOMs.
    # "adamw_bnb_8bit" (requires bitsandbytes) reduces optimizer state to
    # ~9GB. Other valid values: "adafactor", "adamw_torch_fused", etc.
    optim: str = "adamw_torch"
    # gradient_checkpointing trades ~30% throughput for 4-5x activation
    # memory savings. Necessary when activation memory is the bottleneck;
    # mostly orthogonal to optimizer state.
    gradient_checkpointing: bool = False
    # Restrict PGD recovery to layers strictly past min_layer_idx. When set
    # (None disables), the runner does TWO things:
    #   1. Filters the Wanda mask dict so the PGD zero-projection (and
    #      per-step validate_sparsity walk) only touches layers > N.
    #   2. Freezes every "early-side" parameter via requires_grad=False —
    #      every param with `.layers.<M>.` for M <= N, plus embed_tokens.
    #      Tied-weight semantics: if a tensor is shared between an early-side
    #      name and a late-side name (e.g. `lm_head.weight` tied to
    #      `embed_tokens.weight`), the *whole tensor is frozen* — anything
    #      before layer N is not trained even if the same tensor is also
    #      used after layer N. This is the source of the actual memory +
    #      compute savings (no Adam state, no gradient buffer for frozen
    #      params); the mask filter alone only saves the per-step zeroing
    #      walk.
    # Example: min_layer_idx=31 on gemma-2-9b-it (42 layers) trains only
    # layers 32..41 plus the root final norm + (un-tied) lm_head, and
    # matches the deepest GemmaScope SAE used on the aruna branch.
    min_layer_idx: Optional[int] = None


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
