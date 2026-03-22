"""
Unit tests for prune_and_maybe_recover_sweep.py

Tests the pure-Python logic that does not require a model, GPU, or network:
- GiveUpThreshold / SweepStepResult / SweepResult pydantic schemas
- is_metric_passing: direction per metric type
- is_metric_better: direction per metric type
- CheckpointCache: capacity, priority ordering, acceptance rules
- _checkpoint_sort_key: sort stability
- SweepRecoveryCallback.on_step_end: early-stop and give-up logic (mocked)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from prune_and_maybe_recover_sweep import (
    CheckpointCache,
    GiveUpThreshold,
    SweepRecoveryCallback,
    SweepResult,
    SweepStepResult,
    _checkpoint_sort_key,
    is_metric_better,
    is_metric_passing,
)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_give_up_threshold_schema() -> None:
    g = GiveUpThreshold(steps=50, threshold=2.5)
    assert g.steps == 50
    assert g.threshold == 2.5


def test_sweep_step_result_schema() -> None:
    r = SweepStepResult(
        step_index=0,
        sparsity=0.5,
        n_weights_zeroed=100,
        metric_type="loss",
        metric_before_recovery=3.0,
        metric_after_recovery=2.4,
        recovery_steps=50,
        recovery_stopped_early=True,
        gave_up=False,
        is_success=True,
    )
    assert r.sparsity == 0.5
    assert r.is_success is True


def test_sweep_result_schema() -> None:
    step = SweepStepResult(
        step_index=0, sparsity=0.5, n_weights_zeroed=10, metric_type="loss",
        metric_before_recovery=3.0, metric_after_recovery=2.4,
        recovery_steps=5, recovery_stopped_early=False,
        gave_up=False, is_success=True,
    )
    result = SweepResult(
        best_sparsity=0.5, best_metric=2.4, metric_type="loss",
        threshold=2.5, k_min=0.0, k_max=1.0,
        steps=[step], cached_checkpoint_dirs=["/tmp/step_0"],
    )
    assert result.best_sparsity == 0.5
    json_str = result.model_dump_json()
    assert "best_sparsity" in json_str


def test_sweep_result_no_success() -> None:
    result = SweepResult(
        best_sparsity=None, best_metric=None,
        metric_type="loss", threshold=2.5, k_min=0.0, k_max=1.0,
        steps=[], cached_checkpoint_dirs=[],
    )
    assert result.best_sparsity is None


# ---------------------------------------------------------------------------
# is_metric_passing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric,threshold,expected", [
    (2.0, 2.5, True),   # loss below threshold → pass
    (2.5, 2.5, True),   # loss equal threshold → pass
    (3.0, 2.5, False),  # loss above threshold → fail
])
def test_is_metric_passing_loss(metric, threshold, expected) -> None:
    assert is_metric_passing(metric, "loss", threshold) == expected


@pytest.mark.parametrize("metric,threshold,expected", [
    (0.8, 0.7, True),   # judge above threshold → pass
    (0.7, 0.7, True),   # judge equal threshold → pass
    (0.5, 0.7, False),  # judge below threshold → fail
])
def test_is_metric_passing_judge(metric, threshold, expected) -> None:
    assert is_metric_passing(metric, "judge", threshold) == expected


# ---------------------------------------------------------------------------
# is_metric_better
# ---------------------------------------------------------------------------


def test_is_metric_better_loss() -> None:
    assert is_metric_better(1.9, 2.0, "loss") is True
    assert is_metric_better(2.1, 2.0, "loss") is False
    assert is_metric_better(2.0, 2.0, "loss") is False


def test_is_metric_better_judge() -> None:
    assert is_metric_better(0.9, 0.8, "judge") is True
    assert is_metric_better(0.7, 0.8, "judge") is False
    assert is_metric_better(0.8, 0.8, "judge") is False


# ---------------------------------------------------------------------------
# _checkpoint_sort_key
# ---------------------------------------------------------------------------


def test_checkpoint_sort_key_loss() -> None:
    high_sp = (0.8, 2.0, "/a")
    low_sp = (0.3, 1.5, "/b")
    assert _checkpoint_sort_key(high_sp, "loss") > _checkpoint_sort_key(low_sp, "loss")


def test_checkpoint_sort_key_judge_tiebreak() -> None:
    same_sp_good = (0.5, 0.9, "/a")
    same_sp_bad = (0.5, 0.6, "/b")
    assert _checkpoint_sort_key(same_sp_good, "judge") > _checkpoint_sort_key(same_sp_bad, "judge")


def test_checkpoint_sort_key_loss_tiebreak() -> None:
    same_sp_low_loss = (0.5, 1.5, "/a")
    same_sp_high_loss = (0.5, 3.0, "/b")
    assert _checkpoint_sort_key(same_sp_low_loss, "loss") > _checkpoint_sort_key(same_sp_high_loss, "loss")


# ---------------------------------------------------------------------------
# CheckpointCache
# ---------------------------------------------------------------------------


def test_checkpoint_cache_accepts_up_to_capacity() -> None:
    cache = CheckpointCache(capacity=2, metric_type="loss")
    assert cache.try_add(0.3, 2.0, "/a") is True
    assert cache.try_add(0.5, 2.5, "/b") is True
    assert len(cache.checkpoint_dirs()) == 2


def test_checkpoint_cache_evicts_worst_when_full() -> None:
    cache = CheckpointCache(capacity=2, metric_type="loss")
    cache.try_add(0.3, 2.0, "/low_sparsity")
    cache.try_add(0.5, 2.5, "/mid_sparsity")
    # Add higher-sparsity entry — should evict /low_sparsity
    cache.try_add(0.7, 3.0, "/high_sparsity")
    dirs = cache.checkpoint_dirs()
    assert len(dirs) == 2
    assert "/low_sparsity" not in dirs
    assert "/high_sparsity" in dirs


def test_checkpoint_cache_judge_priority() -> None:
    cache = CheckpointCache(capacity=2, metric_type="judge")
    cache.try_add(0.5, 0.6, "/ok")
    cache.try_add(0.5, 0.9, "/great")
    cache.try_add(0.5, 0.1, "/bad")
    dirs = cache.checkpoint_dirs()
    assert "/bad" not in dirs
    assert "/great" in dirs


def test_checkpoint_cache_dirs_ordered_best_first() -> None:
    cache = CheckpointCache(capacity=3, metric_type="loss")
    cache.try_add(0.3, 2.0, "/low")
    cache.try_add(0.7, 3.0, "/high")
    cache.try_add(0.5, 2.5, "/mid")
    dirs = cache.checkpoint_dirs()
    assert dirs[0] == "/high"


# ---------------------------------------------------------------------------
# SweepRecoveryCallback — mocked model
# ---------------------------------------------------------------------------


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _make_callback(
    threshold: float = 2.5,
    metric_type: str = "loss",
    eval_every: int = 1,
    give_up_thresholds: list[GiveUpThreshold] | None = None,
) -> SweepRecoveryCallback:
    return SweepRecoveryCallback(
        eval_every=eval_every,
        threshold=threshold,
        metric_type=metric_type,
        tokenizer=MagicMock(),
        eval_texts=["dummy"],
        eval_conversations=[[{"role": "user", "content": "hi"}]],
        batch_size=1,
        max_seq_len=32,
        max_new_tokens=16,
        give_up_thresholds=give_up_thresholds or [],
    )


def _make_trainer_state(step: int) -> MagicMock:
    state = MagicMock()
    state.global_step = step
    return state


def _make_control() -> MagicMock:
    control = MagicMock()
    control.should_training_stop = False
    return control


def test_callback_stops_early_when_loss_threshold_met() -> None:
    cb = _make_callback(threshold=2.5, metric_type="loss")
    with patch.object(cb, "_compute_metric", return_value=2.0):
        control = _make_control()
        result = cb.on_step_end(MagicMock(), _make_trainer_state(1), control, model=_TinyNet())
        assert result.should_training_stop is True
        assert cb.gave_up is False


def test_callback_does_not_stop_when_loss_above_threshold() -> None:
    cb = _make_callback(threshold=2.5, metric_type="loss")
    with patch.object(cb, "_compute_metric", return_value=3.0):
        control = _make_control()
        result = cb.on_step_end(MagicMock(), _make_trainer_state(1), control, model=_TinyNet())
        assert result.should_training_stop is False
        assert cb.gave_up is False


def test_callback_stops_early_when_judge_threshold_met() -> None:
    cb = _make_callback(threshold=0.7, metric_type="judge")
    with patch.object(cb, "_compute_metric", return_value=0.8):
        control = _make_control()
        result = cb.on_step_end(MagicMock(), _make_trainer_state(1), control, model=_TinyNet())
        assert result.should_training_stop is True


def test_callback_give_up_triggered() -> None:
    rules = [GiveUpThreshold(steps=10, threshold=2.5)]
    cb = _make_callback(threshold=1.0, metric_type="loss", give_up_thresholds=rules)
    with patch.object(cb, "_compute_metric", return_value=3.0):
        control = _make_control()
        result = cb.on_step_end(MagicMock(), _make_trainer_state(10), control, model=_TinyNet())
        assert result.should_training_stop is True
        assert cb.gave_up is True


def test_callback_give_up_not_triggered_before_steps() -> None:
    rules = [GiveUpThreshold(steps=10, threshold=2.5)]
    cb = _make_callback(threshold=1.0, metric_type="loss", give_up_thresholds=rules)
    with patch.object(cb, "_compute_metric", return_value=3.0):
        control = _make_control()
        result = cb.on_step_end(MagicMock(), _make_trainer_state(5), control, model=_TinyNet())
        assert result.should_training_stop is False
        assert cb.gave_up is False


def test_callback_skips_eval_when_not_on_interval() -> None:
    cb = _make_callback(eval_every=5)
    with patch.object(cb, "_compute_metric") as mock_eval:
        control = _make_control()
        cb.on_step_end(MagicMock(), _make_trainer_state(3), control, model=_TinyNet())
        mock_eval.assert_not_called()


def test_callback_skips_eval_when_model_none() -> None:
    cb = _make_callback()
    with patch.object(cb, "_compute_metric") as mock_eval:
        control = _make_control()
        cb.on_step_end(MagicMock(), _make_trainer_state(1), control, model=None)
        mock_eval.assert_not_called()


def test_callback_records_last_metric() -> None:
    cb = _make_callback(threshold=2.5, metric_type="loss")
    with patch.object(cb, "_compute_metric", return_value=2.8):
        control = _make_control()
        cb.on_step_end(MagicMock(), _make_trainer_state(1), control, model=_TinyNet())
    assert cb.last_metric == pytest.approx(2.8)


# ---------------------------------------------------------------------------
# SweepResult JSON round-trip
# ---------------------------------------------------------------------------


def test_sweep_result_json_round_trip(tmp_path: Path) -> None:
    result = SweepResult(
        best_sparsity=0.6,
        best_metric=2.3,
        metric_type="loss",
        threshold=2.5,
        k_min=0.1,
        k_max=0.9,
        steps=[],
        cached_checkpoint_dirs=[str(tmp_path / "ckpt0")],
    )
    json_path = tmp_path / "sweep_result.json"
    json_path.write_text(result.model_dump_json(indent=2))
    loaded = SweepResult.model_validate_json(json_path.read_text())
    assert loaded.best_sparsity == pytest.approx(0.6)
    assert loaded.metric_type == "loss"
    assert len(loaded.cached_checkpoint_dirs) == 1
