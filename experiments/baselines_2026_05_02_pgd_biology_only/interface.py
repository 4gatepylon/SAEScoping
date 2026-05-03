"""Pydantic schemas for the wanda+PGD+elicitation sweep.

Single source of truth for: model configs, experiment configs, the three
step instance types (Calibration / PGD / Elicitation), and the compiled
dependency graph that the scheduler consumes.

Loosely modeled on `sae_scoping.utils.sweep_config.SweepConfig`, but split
along the experiment's three-step computational model. The runner (and
each per-step CLI) reads ONE of these YAMLs from disk and treats it as
authoritative; CLI flags only override device/GPU pin and a few debug
toggles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic_yaml import parse_yaml_file_as

# ── Common ───────────────────────────────────────────────────────────────

StepType = Literal["calibrate", "pgd", "elicit"]
OOMRetryMode = Literal["ClearAndRetryIfNoCheckpointElseContinue"]


class FrozenFromYaml(BaseModel):
    """Common base: forbid extra fields so typos don't silently drop."""

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FrozenFromYaml":
        return parse_yaml_file_as(cls, str(path))


# ── Model config ─────────────────────────────────────────────────────────



class TrainerWrapperConfig(FrozenFromYaml):
    """Knobs that live OUTSIDE SFTConfig: early stopping, eval cadence, etc."""

    eval_every_steps: int = 100
    # Min fraction-of-vanilla LLM-judge score below which the run aborts
    # early (and still saves a final checkpoint per the README).
    pgd_min_relevance_frac: float = 1.00
    pgd_min_fluency_frac: float = 1.00
    elicit_min_score_frac: float = 0.90
    # If None, train every layer (used by the small models).
    min_layer_idx: Optional[int] = None
    # Forwarded to compute_loss / OneClickLLMJudgeScopingEval.
    max_seq_len: int = 800
    eval_batch_size: int = 2


class ModelConfig(FrozenFromYaml):
    """Per-model file. One YAML per (model_id, target hardware) pair."""

    model_id: str
    # NOTE: SFT is just the serialized version of SFTConfig from trl
    sft: dict[str, Any] = Field(default_factory=dict)
    wrapper: TrainerWrapperConfig = Field(default_factory=TrainerWrapperConfig)
    # Wanda calibration knobs (forward-only, so often larger bs than PGD).
    calibration_batch_size: int = 2
    calibration_max_seq_len: int = 800


# ── Experiment config ────────────────────────────────────────────────────


class LLMJudgeConfig(FrozenFromYaml):
    enabled: bool = True
    judge_model: str = "gpt-4.1-nano"
    n_samples: int = 50
    split: str = "validation"


class WandbConfig(FrozenFromYaml):
    enabled: bool = True
    project: str = "deleteme__baselines_2026_05_02_pgd_biology_only__placeholder"


class OperationalConfig(FrozenFromYaml):
    """Where things live + GPU pool + memory knobs."""

    # Subfolder under $SAESCOPING_ARTIFACTS_LOCATION; usually
    # "baselines_2026_05_02_pgd_biology_only".
    artifacts_subdir: str = "baselines_2026_05_02_pgd_biology_only"
    # Logical CUDA devices the scheduler may dispatch onto.
    devices: list[str] = Field(default_factory=lambda: ["cuda:0"])
    # Only one retry mode is supported: try again until you get at least
    # one checkpoint (then continue as if you succeeded). The batch is
    # decreased by powers of 2. If no checkpoint is saved because this is
    # not meant to save a checkpoint, then it just retries.
    oom_retry_mode: OOMRetryMode = "ClearAndRetryIfNoCheckpointElseContinue"
    no_cache: bool = False
    # Off by default (Reminder 13): elicitation only emits judge logs to
    # save disk. Flipping this writes weights to elicitation_checkpoints/.
    save_elicitation_checkpoints: bool = False
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    llm_judge: LLMJudgeConfig = Field(default_factory=LLMJudgeConfig)

class ExperimentConfig(FrozenFromYaml):
    """Top-level experiment file. One per `run_*.sh`."""

    name: str  # e.g. "mini_test" / "mini_real" / "full_real"

    # Filenames of ModelConfig YAMLs, resolved relative to the experiment YAML.
    model_configs: list[str] = Field(default_factory=list)
    scope_domains: list[str] = Field(default_factory=lambda: ["biology", "chemistry", "math", "physics"])
    # If None, defaults to (all_domains \ {scope_domain}) at schedule time.
    elicitation_domains: Optional[list[str]] = None
    sparsities: list[float] = Field(default_factory=lambda: [0.5, 0.6, 0.7, 0.8])
    # Calibration-step informational sweep over thresholds applied to the
    # already-computed saliency map. Must be a SUPERSET of `sparsities`
    # (and include 0.0 to record the vanilla baseline). No retraining.
    calibration_sweep_sparsities: list[float] = Field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )

    # NOTE subsets = domains
    dataset_name: str = "4gate/StemQAMixture"
    n_calibration: int = 4000
    n_train: int = 36000
    n_eval: int = 200

    operational: OperationalConfig = Field(default_factory=OperationalConfig)
    # Merged on top of each model config's `sft` dict at step-launch time.
    # This is how mini_test clamps max_steps/save_steps without forking
    # the model YAMLs.
    sft_overrides: dict[str, Any] = Field(default_factory=dict)



# ── Step instances + dependency graph ────────────────────────────────────


class CalibrateStep(FrozenFromYaml):
    type: Literal["calibrate"] = "calibrate"
    model_id: str
    scope_domain: str
    # Path (under artifacts subdir) where the saliency map lands.
    saliency_path: str


class PGDStep(FrozenFromYaml):
    type: Literal["pgd"] = "pgd"
    model_id: str
    scope_domain: str
    sparsity: float
    saliency_path: str  # input from CalibrateStep
    checkpoint_dir: str  # output dir


class ElicitStep(FrozenFromYaml):
    type: Literal["elicit"] = "elicit"
    model_id: str
    scope_domain: str
    sparsity: float
    elicitation_domain: str
    pgd_checkpoint_dir: str  # input from PGDStep
    checkpoint_dir: str  # output dir


Step = Annotated[
    CalibrateStep | PGDStep | ElicitStep,
    Field(discriminator="type"),
]


class DependencyGraphNode(FrozenFromYaml):
    step_id: str  # stable hash, used as the W&B run name suffix
    step: Step
    # step_ids of upstream nodes that must complete before this one runs.
    deps: list[str] = Field(default_factory=list)


class DependencyGraph(FrozenFromYaml):
    """Compiled graph; serialized to runtime_state_mirror/dependency_graph.yaml."""

    experiment_name: str
    nodes: list[DependencyGraphNode] = Field(default_factory=list)


# ── Pre-compiled step spec ────────────────────────────────────────────────


class StepSpec(FrozenFromYaml):
    """Self-contained, pre-compiled spec for executing one step.

    Written by the scheduler at graph-compile time into
    runtime_state_mirror/step_specs/{step_id}.yaml. Each script accepts a
    single --step-spec flag pointing at one of these files.

    The sft_overrides from ExperimentConfig are already merged into
    model_config.sft at compile time — no further merging at runtime.
    """

    step: Step
    model_config: ModelConfig
    dataset_name: str
    scope_domains: list[str]
    n_calibration: int
    n_train: int
    n_eval: int
    calibration_sweep_sparsities: list[float]
    artifacts_subdir: str
    no_cache: bool
    save_elicitation_checkpoints: bool = False
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    llm_judge: LLMJudgeConfig = Field(default_factory=LLMJudgeConfig)
    device: str = "cuda:0"


# ── Helpers ──────────────────────────────────────────────────────────────


def _slash_safe(model_id: str) -> str:
    """Replace `/` with `__` so HF model IDs are usable as path / W&B components.

    Reminder 4: never inline `model_id.replace("/", "__")` — go through here.
    """
    return model_id.replace("/", "__")


def make_step_id(step: Step) -> str:
    """Stable 8-char hash of a step's identity tuple. Used in W&B + filesystem."""
    # TODO(claude): include hashable subset of model/sweep config so two
    # runs with the same axes but different hyperparams don't collide.
    import hashlib

    parts = [step.type, _slash_safe(step.model_id), step.scope_domain]
    if isinstance(step, (PGDStep, ElicitStep)):
        parts.append(f"sp{step.sparsity}")
    if isinstance(step, ElicitStep):
        parts.append(step.elicitation_domain)
    return hashlib.sha256("__".join(parts).encode()).hexdigest()[:8]


def wandb_run_name(step: Step) -> str:
    """`{step_type}__{model_short}__{scope}__sp{sparsity}[__{elicit}]__{hash8}`."""
    short = step.model_id.split("/")[-1]
    bits = [step.type, short, step.scope_domain]
    if isinstance(step, (PGDStep, ElicitStep)):
        bits.append(f"sp{step.sparsity}")
    if isinstance(step, ElicitStep):
        bits.append(step.elicitation_domain)
    bits.append(make_step_id(step))
    return "__".join(bits)
