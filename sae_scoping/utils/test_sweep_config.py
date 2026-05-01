"""Tests for `sae_scoping.utils.sweep_config`."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sae_scoping.utils.sweep_config import (
    CalibrationConfig,
    LLMJudgeConfig,
    OperationalConfig,
    PGDConfig,
    PruningSweepConfig,
    SweepConfig,
    WandbConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENT_YAMLS = sorted(
    list((_REPO_ROOT / "experiments" / "baselines_2026_04_30_pgd_after_sae").glob("*.yaml"))
    + list((_REPO_ROOT / "experiments" / "baselines_2026_04_30_before_pgd_ood_eval").glob("*.yaml"))
)


@pytest.mark.parametrize("yaml_path", _EXPERIMENT_YAMLS, ids=lambda p: p.parent.name + "/" + p.name)
def test_experiment_yaml_parses_into_sweep_config(yaml_path: Path) -> None:
    """Every YAML in the late-2026-04-30 experiment folders must parse into a
    valid SweepConfig (catches typos, extra keys, wrong types). Tier 0 of the
    no-GPU integration check."""
    cfg = SweepConfig.from_yaml(str(yaml_path))
    # Sanity: a few invariants we expect to hold across all of these configs.
    assert cfg.model_id.startswith("google/")
    assert cfg.dataset_name == "4gate/StemQAMixture"
    assert all(0.0 < s < 1.0 for s in cfg.sweep.nn_linear_sparsities)


def test_default_sweep_config_is_valid_and_complete() -> None:
    """An empty config still produces a fully populated SweepConfig."""
    cfg = SweepConfig()
    assert cfg.model_id == "google/gemma-3-4b-it"
    assert cfg.dataset_subset == "biology"
    assert cfg.calibration.n_calibration == 128
    assert cfg.sweep.nn_linear_sparsities == [0.5]
    assert cfg.sweep.n_eval == 64
    assert cfg.operational.device == "cuda:0"
    assert cfg.operational.wandb.enabled is False
    assert cfg.operational.llm_judge.enabled is False
    assert cfg.pgd.enabled is False


def test_field_paths_disambiguate_settings() -> None:
    """Each setting lives at exactly one obvious path."""
    cfg = SweepConfig()
    # Calibration vs. sweep vs. judge n_*  — three different "n" fields, three different paths.
    assert hasattr(cfg.calibration, "n_calibration")
    assert hasattr(cfg.sweep, "n_eval")
    assert hasattr(cfg.operational.llm_judge, "n_samples")
    # learning_rate and judge_model are unambiguously under pgd / llm_judge.
    assert cfg.pgd.learning_rate == 2e-5
    assert cfg.operational.llm_judge.judge_model == "gpt-4.1-nano"


def test_extra_keys_rejected() -> None:
    """Typos in YAML must fail loudly, not silently drop."""
    with pytest.raises(ValidationError):
        SweepConfig(model_id="x", typo_field="oops")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        CalibrationConfig(n_calibration=64, batch_zise=2)  # type: ignore[call-arg]


def test_yaml_round_trip(tmp_path: Path) -> None:
    """YAML written by hand parses to a SweepConfig with the right values."""
    yaml_path = tmp_path / "sweep.yaml"
    yaml_path.write_text(
        """
model_id: google/gemma-3-12b-it
dataset_subset: math
calibration:
  n_calibration: 32
  max_seq_len: 512
sweep:
  nn_linear_sparsities: [0.2, 0.4, 0.6]
  n_eval: 16
operational:
  device: cuda:1
  wandb:
    enabled: true
    project: deleteme
    tags: smoke,wanda
  llm_judge:
    enabled: true
    n_samples: 5
    domains: [biology, math]
pgd:
  enabled: false
  learning_rate: 1.0e-4
""",
        encoding="utf-8",
    )
    cfg = SweepConfig.from_yaml(str(yaml_path))
    assert cfg.model_id == "google/gemma-3-12b-it"
    assert cfg.dataset_subset == "math"
    assert cfg.calibration.n_calibration == 32
    assert cfg.calibration.max_seq_len == 512
    assert cfg.sweep.nn_linear_sparsities == [0.2, 0.4, 0.6]
    assert cfg.sweep.n_eval == 16
    assert cfg.operational.device == "cuda:1"
    assert cfg.operational.wandb.enabled is True
    assert cfg.operational.wandb.project == "deleteme"
    assert cfg.operational.wandb.tags == "smoke,wanda"
    assert cfg.operational.llm_judge.enabled is True
    assert cfg.operational.llm_judge.n_samples == 5
    assert cfg.operational.llm_judge.domains == ["biology", "math"]
    assert cfg.pgd.learning_rate == 1e-4


def test_yaml_partial_uses_defaults(tmp_path: Path) -> None:
    """A YAML that only sets a few fields fills the rest with defaults."""
    yaml_path = tmp_path / "sweep.yaml"
    yaml_path.write_text("model_id: google/gemma-3-4b-it\n", encoding="utf-8")
    cfg = SweepConfig.from_yaml(str(yaml_path))
    assert cfg.model_id == "google/gemma-3-4b-it"
    # Defaults survived everywhere else.
    assert cfg.calibration.n_calibration == 128
    assert cfg.sweep.nn_linear_sparsities == [0.5]
    assert cfg.operational.device == "cuda:0"


def test_subconfig_isolation() -> None:
    """Sub-configs construct independently."""
    assert CalibrationConfig().n_calibration == 128
    assert PruningSweepConfig().n_eval == 64
    assert OperationalConfig().device == "cuda:0"
    assert WandbConfig().enabled is False
    assert LLMJudgeConfig().split == "validation"
    assert PGDConfig().enabled is False
    # PGD config has its full surface even though the runner ignores it.
    assert PGDConfig().learning_rate == 2e-5
    assert PGDConfig().train_batch_size == 2
