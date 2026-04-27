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

from pathlib import Path

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


def model_id_from_slug(slug: str) -> str:
    return slug.replace("--", "/")


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
