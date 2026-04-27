"""
Shared schemas, constants, and path helpers for the baseline sweeps.

Used by both calibration_and_recovery_sweep.py and elicitation_sweep.py.

Artifact layout on disk:

  {artifact_dir}/
    {model_slug}/
      {domain}/
        wanda_saliency.safetensors
        random_saliency.safetensors
        ema_grads.safetensors
        taylor_saliency.safetensors
        gradient_saliency.safetensors
        {method}/
          result.json               <- MethodDomainResult
          sp_0.30/
            masks.safetensors       <- boolean keep-masks (True=keep)
            metrics.json            <- SparsityResult
            recovered/              <- HF checkpoint (model + tokenizer)
          sp_0.50/
            ...
    manifest.json                   <- SweepManifest (top-level index)
"""
from __future__ import annotations

import json  # BUG TODO(adriano): unused import, delete
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


# ============================================================================
# Grid constants
# ============================================================================

METHODS = ["wanda", "random", "taylor", "gradient"]
MODELS = ["google/gemma-2-9b-it", "google/gemma-3-12b-it"]
DOMAINS = ["biology", "math", "physics", "chemistry"]

DEFAULT_SPARSITY_LEVELS = [0.3, 0.5, 0.7]
DEFAULT_ARTIFACT_DIR = Path("./artifacts")
DEFAULT_DATASET = "4gate/StemQAMixture"

SALIENCY_FILENAMES = {
    "wanda": "wanda_saliency.safetensors",
    "random": "random_saliency.safetensors",
    "taylor": "taylor_saliency.safetensors",
    "gradient": "gradient_saliency.safetensors",
}


# ============================================================================
# Path helpers
# ============================================================================


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "--")


def saliency_path(artifact_dir: Path, model_id: str, domain: str, method: str) -> Path:
    return artifact_dir / model_slug(model_id) / domain / SALIENCY_FILENAMES[method]


def sparsity_dir(
    artifact_dir: Path, model_id: str, domain: str, method: str, sparsity: float,
) -> Path:
    return artifact_dir / model_slug(model_id) / domain / method / f"sp_{sparsity:.2f}"


def recovered_model_dir(
    artifact_dir: Path, model_id: str, domain: str, method: str, sparsity: float,
) -> Path:
    return sparsity_dir(artifact_dir, model_id, domain, method, sparsity) / "recovered"


def masks_path(
    artifact_dir: Path, model_id: str, domain: str, method: str, sparsity: float,
) -> Path:
    return sparsity_dir(artifact_dir, model_id, domain, method, sparsity) / "masks.safetensors"


# ============================================================================
# Pydantic models — artifact interface between sweep phases
# ============================================================================


class SparsityResult(BaseModel):
    """Artifacts and metrics for one (method, model, domain, sparsity) point."""

    sparsity: float
    masks_path: Path
    pruned_loss_train: float | None = None
    pruned_loss_test: float | None = None
    recovered_model_dir: Path | None = None
    recovered_loss_train: float | None = None
    recovered_loss_test: float | None = None


class MethodDomainResult(BaseModel):
    """All results for one (method, model, domain) tuple across sparsity levels."""

    method: str
    model_id: str
    domain: str
    dataset_name: str
    saliency_path: Path
    results: list[SparsityResult]


class SweepManifest(BaseModel):
    """Top-level index of all completed experiments.

    The elicitation sweep reads this to discover artifacts from the
    calibration sweep without walking the filesystem.
    """

    artifact_dir: Path
    entries: list[MethodDomainResult] = Field(default_factory=list)

    def save(self, path: Path | None = None) -> None:
        p = path or (self.artifact_dir / "manifest.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> SweepManifest:
        return cls.model_validate_json(path.read_text())


# ============================================================================
# SFT config resolution
# ============================================================================

_SFT_DEFAULTS_PATH = Path(__file__).parent / "sft_defaults.yaml"

# Fields managed by the sweep scripts. Users must not set these in the YAML
# or via --sft-overrides — doing so raises ValueError.
RESERVED_SFT_FIELDS = frozenset({
    "output_dir",
    "save_strategy",
    "report_to",
    "dataset_text_field",
})


def _check_no_reserved(cfg: dict, source: str) -> None:
    """Raise if cfg contains any reserved SFT fields."""
    conflicts = RESERVED_SFT_FIELDS & cfg.keys()
    if conflicts:
        raise ValueError(
            f"Reserved SFT fields in {source}: {sorted(conflicts)}. "
            f"These are managed by the sweep scripts and must not be set manually."
        )


def load_sft_defaults(
    phase: str,
    model_id: str,
    overrides: dict | None = None,
    defaults_path: Path = _SFT_DEFAULTS_PATH,
) -> dict:
    """Resolve SFT config for a given training phase and model.

    Resolution order (later wins):
      base -> phase -> models.<model_id>.base -> models.<model_id>.<phase> -> overrides

    Args:
        phase: "recovery" or "elicitation".
        model_id: HuggingFace model ID (e.g. "google/gemma-2-9b-it").
        overrides: Optional dict of CLI overrides (from --sft-overrides).
        defaults_path: Path to sft_defaults.yaml.

    Returns:
        Merged dict of SFTConfig kwargs (excluding reserved fields, which
        are added by the caller).
    """
    with open(defaults_path) as f:
        raw = yaml.safe_load(f)
    # BUG TODO(adriano): yaml.safe_load returns None for empty files, crashing on raw.get() below
    raw = raw or {}

    result = {}

    base = raw.get("base") or {}
    _check_no_reserved(base, "sft_defaults.yaml/base")
    result.update(base)

    phase_cfg = raw.get(phase) or {}
    _check_no_reserved(phase_cfg, f"sft_defaults.yaml/{phase}")
    result.update(phase_cfg)

    model_cfgs = raw.get("models") or {}
    if model_id in model_cfgs:
        mcfg = model_cfgs[model_id] or {}
        model_base = mcfg.get("base") or {}
        _check_no_reserved(model_base, f"sft_defaults.yaml/models/{model_id}/base")
        result.update(model_base)

        model_phase = mcfg.get(phase) or {}
        _check_no_reserved(model_phase, f"sft_defaults.yaml/models/{model_id}/{phase}")
        result.update(model_phase)

    if overrides:
        _check_no_reserved(overrides, "--sft-overrides")
        result.update(overrides)

    return result


def make_sft_config(
    phase: str,
    model_id: str,
    output_dir: str | Path,
    overrides: dict | None = None,
    defaults_path: Path = _SFT_DEFAULTS_PATH,
):
    """Build a complete SFTConfig for the given phase and model.

    Loads defaults from YAML, merges overrides, and adds reserved fields.
    Deferred import of trl.SFTConfig so this module stays fast to import.
    """
    from trl import SFTConfig

    cfg = load_sft_defaults(phase, model_id, overrides, defaults_path)

    cfg["output_dir"] = str(output_dir)
    cfg["save_strategy"] = "no"
    cfg["report_to"] = "none"
    cfg["dataset_text_field"] = "text"

    return SFTConfig(**cfg)
