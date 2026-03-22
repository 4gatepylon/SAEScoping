"""
Unit + component tests for prune_and_maybe_recover_sweep.py

Split into two sections:

Section A — Pure-logic unit tests (no model, no GPU required)
    Covers schemas, is_metric_passing, is_metric_better, CheckpointCache,
    _checkpoint_sort_key, and SweepRecoveryCallback with a mocked _compute_metric.

Section B — Component tests with a real (tiny) saved model on CPU
    Creates a 1-layer Qwen2 model from config (no download), saves it to a temp
    directory, then calls run_sweep_step and prune_and_maybe_recover_sweep
    directly.  Uses the Qwen2.5-0.5B tokenizer (expected to be cached).

    Verifies:
    - run_sweep_step prune-only: weights zeroed, metric computed, no recovery
    - run_sweep_step with recovery: SFT runs, metric logged, step result valid
    - prune_and_maybe_recover_sweep binary search: bounds advance correctly
    - JSON output files written at the right paths
    - CheckpointCache populated from successful steps
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from prune_and_maybe_recover_sweep_old import (
    CheckpointCache,
    GiveUpThreshold,
    SweepResult,
    SweepStepResult,
    _checkpoint_sort_key,
    is_metric_better,
    is_metric_passing,
    prune_and_maybe_recover_sweep,
    run_sweep_step,
    SweepRecoveryCallback,
)


# ---------------------------------------------------------------------------
# Section A — Pure-logic unit tests
# ---------------------------------------------------------------------------


# ── Schemas ──────────────────────────────────────────────────────────────


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
    assert "best_sparsity" in result.model_dump_json()


def test_sweep_result_no_success() -> None:
    result = SweepResult(
        best_sparsity=None, best_metric=None,
        metric_type="loss", threshold=2.5, k_min=0.0, k_max=1.0,
        steps=[], cached_checkpoint_dirs=[],
    )
    assert result.best_sparsity is None


# ── is_metric_passing ─────────────────────────────────────────────────────


@pytest.mark.parametrize("metric,threshold,expected", [
    (2.0, 2.5, True),
    (2.5, 2.5, True),
    (3.0, 2.5, False),
])
def test_is_metric_passing_loss(metric, threshold, expected) -> None:
    assert is_metric_passing(metric, "loss", threshold) == expected


@pytest.mark.parametrize("metric,threshold,expected", [
    (0.8, 0.7, True),
    (0.7, 0.7, True),
    (0.5, 0.7, False),
])
def test_is_metric_passing_judge(metric, threshold, expected) -> None:
    assert is_metric_passing(metric, "judge", threshold) == expected


# ── is_metric_better ──────────────────────────────────────────────────────


def test_is_metric_better_loss() -> None:
    assert is_metric_better(1.9, 2.0, "loss") is True
    assert is_metric_better(2.1, 2.0, "loss") is False
    assert is_metric_better(2.0, 2.0, "loss") is False


def test_is_metric_better_judge() -> None:
    assert is_metric_better(0.9, 0.8, "judge") is True
    assert is_metric_better(0.7, 0.8, "judge") is False
    assert is_metric_better(0.8, 0.8, "judge") is False


# ── _checkpoint_sort_key ──────────────────────────────────────────────────


def test_checkpoint_sort_key_higher_sparsity_better() -> None:
    high_sp = (0.8, 2.0, "/a")
    low_sp = (0.3, 1.5, "/b")
    assert _checkpoint_sort_key(high_sp, "loss") > _checkpoint_sort_key(low_sp, "loss")


def test_checkpoint_sort_key_judge_metric_tiebreak() -> None:
    assert _checkpoint_sort_key((0.5, 0.9, "/a"), "judge") > _checkpoint_sort_key((0.5, 0.6, "/b"), "judge")


def test_checkpoint_sort_key_loss_tiebreak() -> None:
    assert _checkpoint_sort_key((0.5, 1.5, "/a"), "loss") > _checkpoint_sort_key((0.5, 3.0, "/b"), "loss")


# ── CheckpointCache ───────────────────────────────────────────────────────


def test_checkpoint_cache_accepts_up_to_capacity() -> None:
    cache = CheckpointCache(capacity=2, metric_type="loss")
    assert cache.try_add(0.3, 2.0, "/a") is True
    assert cache.try_add(0.5, 2.5, "/b") is True
    assert len(cache.checkpoint_dirs()) == 2


def test_checkpoint_cache_evicts_worst_when_full() -> None:
    cache = CheckpointCache(capacity=2, metric_type="loss")
    cache.try_add(0.3, 2.0, "/low_sparsity")
    cache.try_add(0.5, 2.5, "/mid_sparsity")
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


def test_checkpoint_cache_dirs_best_first() -> None:
    cache = CheckpointCache(capacity=3, metric_type="loss")
    cache.try_add(0.3, 2.0, "/low")
    cache.try_add(0.7, 3.0, "/high")
    cache.try_add(0.5, 2.5, "/mid")
    assert cache.checkpoint_dirs()[0] == "/high"


# ── SweepRecoveryCallback ─────────────────────────────────────────────────


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


def test_callback_stops_early_on_loss_threshold_met() -> None:
    cb = _make_callback(threshold=2.5, metric_type="loss")
    with patch.object(cb, "_compute_metric", return_value=2.0):
        control = _make_control()
        result = cb.on_step_end(MagicMock(), _make_trainer_state(1), control, model=_TinyNet())
        assert result.should_training_stop is True
        assert cb.gave_up is False


def test_callback_continues_when_loss_above_threshold() -> None:
    cb = _make_callback(threshold=2.5, metric_type="loss")
    with patch.object(cb, "_compute_metric", return_value=3.0):
        control = _make_control()
        result = cb.on_step_end(MagicMock(), _make_trainer_state(1), control, model=_TinyNet())
        assert result.should_training_stop is False
        assert cb.gave_up is False


def test_callback_stops_early_on_judge_threshold_met() -> None:
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


def test_callback_skips_eval_off_interval() -> None:
    cb = _make_callback(eval_every=5)
    with patch.object(cb, "_compute_metric") as mock_eval:
        cb.on_step_end(MagicMock(), _make_trainer_state(3), _make_control(), model=_TinyNet())
        mock_eval.assert_not_called()


def test_callback_skips_eval_when_model_none() -> None:
    cb = _make_callback()
    with patch.object(cb, "_compute_metric") as mock_eval:
        cb.on_step_end(MagicMock(), _make_trainer_state(1), _make_control(), model=None)
        mock_eval.assert_not_called()


def test_callback_records_last_metric() -> None:
    cb = _make_callback(threshold=2.5, metric_type="loss")
    with patch.object(cb, "_compute_metric", return_value=2.8):
        cb.on_step_end(MagicMock(), _make_trainer_state(1), _make_control(), model=_TinyNet())
    assert cb.last_metric == pytest.approx(2.8)


# ── SweepResult JSON round-trip ───────────────────────────────────────────


def test_sweep_result_json_round_trip(tmp_path: Path) -> None:
    result = SweepResult(
        best_sparsity=0.6, best_metric=2.3, metric_type="loss",
        threshold=2.5, k_min=0.1, k_max=0.9, steps=[],
        cached_checkpoint_dirs=[str(tmp_path / "ckpt0")],
    )
    json_path = tmp_path / "sweep_result.json"
    json_path.write_text(result.model_dump_json(indent=2))
    loaded = SweepResult.model_validate_json(json_path.read_text())
    assert loaded.best_sparsity == pytest.approx(0.6)
    assert loaded.metric_type == "loss"
    assert len(loaded.cached_checkpoint_dirs) == 1


# ---------------------------------------------------------------------------
# Section B — Component tests with a real (tiny) saved model on CPU
#
# These tests create a 1-layer Qwen2 model from scratch (no network access),
# save it to a temp directory, then run the full sweep pipeline on CPU.
# ---------------------------------------------------------------------------

_TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _make_tiny_qwen_dir(save_dir: str) -> str:
    """Create a 1-layer Qwen2 model from config, matching the Qwen2.5 tokenizer's vocab size.

    The tokenizer (loaded from HF cache) is also saved alongside the model so that
    run_sweep_step and prune_and_maybe_recover_sweep can load both from the same dir.
    """
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = Qwen2Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return save_dir


def _make_saliency_for_dir(model_dir: str, saliency_path: str) -> str:
    """Generate a random saliency map for the saved model's parameters."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    config = Qwen2Config.from_pretrained(model_dir)
    model = Qwen2ForCausalLM(config)
    torch.manual_seed(7)
    saliency = {
        name: torch.rand(param.shape)
        for name, param in model.named_parameters()
    }
    save_file(saliency, saliency_path)
    return saliency_path


def _make_fake_dataset(n: int = 8) -> Dataset:
    rows = [
        {"question": f"What is {i} + {i}?", "answer": f"The answer is {i + i}."}
        for i in range(n)
    ]
    return Dataset.from_list(rows)


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Save a tiny Qwen2 model + tokenizer once per test module."""
    save_dir = str(tmp_path_factory.mktemp("tiny_model"))
    return _make_tiny_qwen_dir(save_dir)


@pytest.fixture(scope="module")
def saliency_path(tiny_model_dir: str, tmp_path_factory: pytest.TempPathFactory) -> str:
    sal_dir = tmp_path_factory.mktemp("saliency")
    return _make_saliency_for_dir(tiny_model_dir, str(sal_dir / "saliency.safetensors"))


@pytest.fixture
def fake_dataset() -> Dataset:
    return _make_fake_dataset()


# ── run_sweep_step: prune-only ─────────────────────────────────────────────


def test_run_sweep_step_prune_only(
    tiny_model_dir: str, saliency_path: str, fake_dataset: Dataset, tmp_path: Path
) -> None:
    """run_sweep_step with threshold=-1 must prune weights without running recovery."""
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    result = run_sweep_step(
        step_index=0,
        model_name_or_path=tiny_model_dir,
        tokenizer=tokenizer,
        saliency_path=saliency_path,
        sparsity=0.3,
        dataset_evaluation=fake_dataset,
        saliency_type="gradient",
        param_regex=None,
        metric_type="loss",
        threshold=-1.0,
        dataset_recovery=None,
        max_steps_recovery=5,
        eval_every=2,
        batch_size=2,
        learning_rate=1e-4,
        max_seq_len=32,
        max_new_tokens=16,
        give_up_thresholds=[],
        step_output_dir=str(tmp_path / "step_0"),
        device="cpu",
    )

    assert result.n_weights_zeroed > 0, (
        f"❌ run_sweep_step_prune_only: expected weights zeroed, got {result.n_weights_zeroed}"
    )
    assert result.recovery_steps == 0, (
        f"❌ run_sweep_step_prune_only: expected 0 recovery steps, got {result.recovery_steps}"
    )
    assert result.is_success is True
    assert result.gave_up is False
    assert result.metric_before_recovery == result.metric_after_recovery
    assert result.metric_before_recovery > 0.0

    print(
        f"✅ run_sweep_step_prune_only: zeroed {result.n_weights_zeroed} weights, "
        f"loss={result.metric_before_recovery:.4f}, no recovery"
    )


# ── run_sweep_step: with recovery (easy threshold) ────────────────────────


def test_run_sweep_step_with_recovery(
    tiny_model_dir: str, saliency_path: str, fake_dataset: Dataset, tmp_path: Path
) -> None:
    """run_sweep_step with threshold=999 triggers recovery; SFT runs, metric logged."""
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    result = run_sweep_step(
        step_index=0,
        model_name_or_path=tiny_model_dir,
        tokenizer=tokenizer,
        saliency_path=saliency_path,
        sparsity=0.3,
        dataset_evaluation=fake_dataset,
        saliency_type="gradient",
        param_regex=None,
        metric_type="loss",
        threshold=999.0,
        dataset_recovery=fake_dataset,
        max_steps_recovery=6,
        eval_every=3,
        batch_size=2,
        learning_rate=1e-4,
        max_seq_len=32,
        max_new_tokens=16,
        give_up_thresholds=[],
        step_output_dir=str(tmp_path / "step_0"),
        device="cpu",
    )

    assert result.n_weights_zeroed > 0, "❌ Expected weights to be pruned"
    assert result.is_success is True, "❌ Expected success with threshold=999"
    assert result.gave_up is False
    assert result.metric_after_recovery > 0.0

    print(
        f"✅ run_sweep_step_with_recovery: zeroed {result.n_weights_zeroed} weights, "
        f"loss before={result.metric_before_recovery:.4f}, "
        f"loss after={result.metric_after_recovery:.4f}, "
        f"steps={result.recovery_steps}"
    )


# ── prune_and_maybe_recover_sweep: binary search advances correctly ────────


def test_sweep_prune_only_all_steps_succeed(
    tiny_model_dir: str, saliency_path: str, fake_dataset: Dataset, tmp_path: Path
) -> None:
    """With threshold=-1, all 3 sweep steps succeed and best_sparsity is found."""
    result = prune_and_maybe_recover_sweep(
        model_name_or_path=tiny_model_dir,
        saliency_path=saliency_path,
        dataset_evaluation=fake_dataset,
        k_min=0.1,
        k_max=0.5,
        metric_type="loss",
        threshold=-1.0,
        max_steps_sweep=3,
        output_dir=str(tmp_path / "sweep"),
        device="cpu",
        wandb_project=None,
    )

    assert result.best_sparsity is not None, "❌ best_sparsity should be found"
    assert len(result.steps) == 3
    assert all(s.is_success for s in result.steps), "❌ all steps should succeed"
    # All steps are prune-only
    assert all(s.recovery_steps == 0 for s in result.steps)
    # Binary search pushes lo upward → best_sparsity > midpoint of [0.1, 0.5]
    assert result.best_sparsity > 0.3, (
        f"❌ best_sparsity={result.best_sparsity:.4f} should exceed midpoint 0.3"
    )
    # JSON files written
    assert (tmp_path / "sweep" / "sweep_result.json").exists()
    for i in range(3):
        assert (tmp_path / "sweep" / f"step_{i:03d}_result.json").exists()
    # SweepResult can be round-tripped from JSON
    loaded = SweepResult.model_validate_json(
        (tmp_path / "sweep" / "sweep_result.json").read_text()
    )
    assert loaded.best_sparsity == pytest.approx(result.best_sparsity)

    print(
        f"✅ sweep_prune_only: 3 steps, best_sparsity={result.best_sparsity:.4f}, "
        f"sparsities={[round(s.sparsity, 3) for s in result.steps]}"
    )


def test_sweep_impossible_threshold_all_fail(
    tiny_model_dir: str, saliency_path: str, fake_dataset: Dataset, tmp_path: Path
) -> None:
    """With threshold=0.001, all 3 steps fail and best_sparsity is None."""
    result = prune_and_maybe_recover_sweep(
        model_name_or_path=tiny_model_dir,
        saliency_path=saliency_path,
        dataset_evaluation=fake_dataset,
        k_min=0.1,
        k_max=0.5,
        metric_type="loss",
        threshold=0.001,
        dataset_recovery=fake_dataset,
        max_steps_recovery=4,
        eval_every=2,
        batch_size=2,
        max_seq_len=32,
        max_steps_sweep=3,
        output_dir=str(tmp_path / "sweep"),
        device="cpu",
        wandb_project=None,
    )

    assert result.best_sparsity is None, (
        f"❌ best_sparsity should be None when all fail, got {result.best_sparsity}"
    )
    assert all(not s.is_success for s in result.steps), "❌ all steps should fail"
    # Binary search pushes hi downward → best_sparsity tracks None
    sparsities = [s.sparsity for s in result.steps]
    assert sparsities[1] < sparsities[0], (
        "❌ second step should try lower sparsity after first step failed"
    )

    print(
        f"✅ sweep_impossible_threshold: 3 steps all failed, "
        f"sparsities={[round(s, 3) for s in sparsities]}"
    )


def test_sweep_checkpoint_cache_populated(
    tiny_model_dir: str, saliency_path: str, fake_dataset: Dataset, tmp_path: Path
) -> None:
    """Successful steps are added to the CheckpointCache (cached_checkpoint_dirs)."""
    result = prune_and_maybe_recover_sweep(
        model_name_or_path=tiny_model_dir,
        saliency_path=saliency_path,
        dataset_evaluation=fake_dataset,
        k_min=0.1,
        k_max=0.5,
        threshold=-1.0,
        max_steps_sweep=3,
        num_cache=3,
        output_dir=str(tmp_path / "sweep"),
        device="cpu",
        wandb_project=None,
    )

    # All 3 steps succeed → up to num_cache dirs should be cached
    assert len(result.cached_checkpoint_dirs) > 0, (
        "❌ cached_checkpoint_dirs should not be empty after successful steps"
    )
    assert len(result.cached_checkpoint_dirs) <= 3

    print(
        f"✅ sweep_checkpoint_cache: {len(result.cached_checkpoint_dirs)} dirs cached"
    )
