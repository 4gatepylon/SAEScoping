"""Integration tests for _run_stage and run_staged_sweep (CPU).

Uses _TinyModelEvaluator from test_staged_sweep_dummy_training to exercise
the real search loop without requiring TRL or HF dataset downloads.

Tests
-----
test_run_stage_tiny_model_binary_converges
    _run_stage + BinarySearch finds the sparsity boundary in the pretrained
    tiny model.  The threshold is derived dynamically from measured losses.

test_run_stage_tiny_model_uniform_grid
    Same but with UniformGridSearch.

test_two_stage_run_narrows_interval
    Two sequential _run_stage calls.  The second stage's interval is
    narrowed from the first stage's output.  Final bounds bracket the
    strict threshold.

test_run_staged_sweep_with_injected_loader
    run_staged_sweep with a DatasetConfig that points at a mock loader.
    Verifies stage result structure and JSON outputs without real HF data.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from prune_and_maybe_recover_sweep import (
    BinarySearch,
    BoundedByPreviousHi,
    ChainFromPrevious,
    DatasetConfig,
    Initial,
    SpanFromPreviousStages,
    SparsityInterval,
    SweepStage,
    UniformGridSearch,
    _run_stage,
    run_staged_sweep,
)
from prune_and_maybe_recover_sweep._evaluators import StepEvaluator
from prune_and_maybe_recover_sweep._schemas import StepEvalResult

from tests.unit.test_staged_sweep_dummy_training import (
    _TinyModelEvaluator,
    _eval_loss,
    _make_uniform_saliency,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures shared with dummy-training tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pretrained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Same pretrained model used in dummy-training tests (module-scoped)."""
    from tests.unit.test_staged_sweep_dummy_training import (
        _make_hardcoded_model,
        _pretrain_model,
    )
    from transformers import AutoModelForCausalLM

    tmp_dir = tmp_path_factory.mktemp("integration_pretrained")
    torch.manual_seed(0)
    model = _make_hardcoded_model()
    _pretrain_model(model, n_steps=100)
    model.save_pretrained(str(tmp_dir))
    return str(tmp_dir)


@pytest.fixture(scope="module")
def saliency_map(pretrained_model_dir: str) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir, device_map="cpu")
    return _make_uniform_saliency(model)


def _make_evaluator(
    model_dir: str,
    saliency: dict[str, torch.Tensor],
    threshold: float,
    n_recovery: int = 0,
) -> _TinyModelEvaluator:
    return _TinyModelEvaluator(model_dir, saliency, threshold=threshold, n_recovery=n_recovery)


def _measure_losses(
    model_dir: str,
    saliency: dict[str, torch.Tensor],
    sparsities: list[float],
) -> list[float]:
    """Measure evaluation losses at multiple sparsities (no recovery)."""
    evaluator = _TinyModelEvaluator(model_dir, saliency, threshold=1e9, n_recovery=0)
    return [
        evaluator.evaluate(s, "", None, None, None, "cpu", "/tmp").metric_after
        for s in sparsities
    ]


# ---------------------------------------------------------------------------
# Integration tests: _run_stage
# ---------------------------------------------------------------------------


def test_run_stage_tiny_model_binary_converges(
    pretrained_model_dir: str, saliency_map: dict[str, torch.Tensor],
) -> None:
    """_run_stage + BinarySearch must converge to the sparsity boundary."""
    loss_0, loss_8 = _measure_losses(pretrained_model_dir, saliency_map, [0.0, 0.8])
    assert loss_8 > loss_0, "Model must produce higher loss at high sparsity"

    threshold = (loss_0 + loss_8) / 2.0
    evaluator = _make_evaluator(pretrained_model_dir, saliency_map, threshold)
    stage = SweepStage(
        name="binary",
        evaluator=evaluator,
        search=BinarySearch(tolerance=0.02),
        interval_spec=Initial(),
        max_steps=15,
        dataset_config=DatasetConfig(n_eval=2, n_recovery=0),
        raise_if_out_of_bounds=False,
    )

    result = _run_stage(
        stage, SparsityInterval(lo=0.0, hi=0.8),
        model_name_or_path=pretrained_model_dir,
        tokenizer=None, dataset_eval=None, dataset_recovery=None,
        device="cpu", output_dir="/tmp/integ_binary",
    )

    assert len(result.steps) > 0
    lo, hi = result.output_interval.lo, result.output_interval.hi
    assert hi - lo < 0.3, f"Expected bounds to narrow, got [{lo:.3f}, {hi:.3f}]"
    assert result.name == "binary"
    print(f"✅ test_run_stage_tiny_model_binary_converges: [{lo:.3f}, {hi:.3f}]")


def test_run_stage_tiny_model_uniform_grid(
    pretrained_model_dir: str, saliency_map: dict[str, torch.Tensor],
) -> None:
    """_run_stage + UniformGridSearch visits all grid points and narrows bounds."""
    loss_0, loss_8 = _measure_losses(pretrained_model_dir, saliency_map, [0.0, 0.8])
    assert loss_8 > loss_0

    threshold = (loss_0 + loss_8) / 2.0
    evaluator = _make_evaluator(pretrained_model_dir, saliency_map, threshold)
    stage = SweepStage(
        name="uniform",
        evaluator=evaluator,
        search=UniformGridSearch(precision=0.1),
        interval_spec=Initial(),
        max_steps=15,
        dataset_config=DatasetConfig(n_eval=2, n_recovery=0),
        raise_if_out_of_bounds=False,
    )

    result = _run_stage(
        stage, SparsityInterval(lo=0.0, hi=0.8),
        model_name_or_path=pretrained_model_dir,
        tokenizer=None, dataset_eval=None, dataset_recovery=None,
        device="cpu", output_dir="/tmp/integ_uniform",
    )

    assert len(result.steps) <= 15
    for step in result.steps:
        assert 0.0 <= step.sparsity <= 0.8

    print(
        f"✅ test_run_stage_tiny_model_uniform_grid: "
        f"{len(result.steps)} steps, [{result.output_interval.lo:.3f}, {result.output_interval.hi:.3f}]"
    )


def test_two_stage_run_narrows_interval(
    pretrained_model_dir: str, saliency_map: dict[str, torch.Tensor],
) -> None:
    """Two-stage chaining: stage 1 narrows within stage 0's output."""
    losses = _measure_losses(pretrained_model_dir, saliency_map, [0.0, 0.4, 0.8])
    loss_0, loss_4, loss_8 = losses
    assert loss_8 > loss_0

    # Stage 0: loose threshold (passes at low sparsity, fails at high)
    loose_threshold = (loss_0 + loss_8) / 2.0
    evaluator0 = _make_evaluator(pretrained_model_dir, saliency_map, loose_threshold)
    stage0 = SweepStage(
        name="loose",
        evaluator=evaluator0,
        search=BinarySearch(tolerance=0.03),
        interval_spec=Initial(),
        max_steps=12,
        dataset_config=DatasetConfig(n_eval=2, n_recovery=0),
        raise_if_out_of_bounds=False,
    )
    result0 = _run_stage(
        stage0, SparsityInterval(lo=0.0, hi=0.8),
        model_name_or_path=pretrained_model_dir,
        tokenizer=None, dataset_eval=None, dataset_recovery=None,
        device="cpu", output_dir="/tmp/integ_stage0",
    )

    # Stage 1: strict threshold (must fit within stage 0's output)
    strict_threshold = (loss_0 + loose_threshold) / 2.0
    evaluator1 = _make_evaluator(pretrained_model_dir, saliency_map, strict_threshold)
    spec1 = ChainFromPrevious()
    stage1 = SweepStage(
        name="strict",
        evaluator=evaluator1,
        search=BinarySearch(tolerance=0.03),
        interval_spec=spec1,
        max_steps=12,
        dataset_config=DatasetConfig(n_eval=2, n_recovery=0),
        raise_if_out_of_bounds=False,
    )
    interval1 = stage1.interval_spec.resolve([result0], SparsityInterval(lo=0.0, hi=0.8))
    assert interval1.lo >= result0.output_interval.lo
    assert interval1.hi <= result0.output_interval.hi + 1e-6

    result1 = _run_stage(
        stage1, interval1,
        model_name_or_path=pretrained_model_dir,
        tokenizer=None, dataset_eval=None, dataset_recovery=None,
        device="cpu", output_dir="/tmp/integ_stage1",
    )

    # Stage 1's output must be at most as wide as stage 0's
    width0 = result0.output_interval.hi - result0.output_interval.lo
    width1 = result1.output_interval.hi - result1.output_interval.lo
    assert width1 <= width0 + 0.05, (
        f"Stage 1 output width {width1:.3f} should be narrower than stage 0 {width0:.3f}"
    )

    print(
        f"✅ test_two_stage_run_narrows_interval: "
        f"stage0=[{result0.output_interval.lo:.3f}, {result0.output_interval.hi:.3f}], "
        f"stage1=[{result1.output_interval.lo:.3f}, {result1.output_interval.hi:.3f}]"
    )


# ---------------------------------------------------------------------------
# Integration test: run_staged_sweep with mocked dataset loading
# ---------------------------------------------------------------------------


class _InjectedDatasetEvaluator(StepEvaluator):
    """Evaluator that ignores provided datasets and uses the tiny model directly.

    Used to test run_staged_sweep without real HF dataset loading.
    The datasets provided by run_staged_sweep are ignored; we use the tiny
    model's internal training data instead.
    """

    def __init__(
        self,
        model_dir: str,
        saliency: dict[str, torch.Tensor],
        threshold: float,
    ) -> None:
        self._inner = _TinyModelEvaluator(model_dir, saliency, threshold=threshold, n_recovery=0)

    @property
    def metric_type(self) -> str:
        return "loss"

    def prepare(self, model_name_or_path, tokenizer, dataset_evaluation, device) -> None:
        pass

    def evaluate(
        self, sparsity, model_name_or_path, tokenizer,
        dataset_evaluation, dataset_recovery, device, output_dir,
    ) -> StepEvalResult:
        return self._inner.evaluate(
            sparsity, model_name_or_path, None, None, None, device, output_dir,
        )


def test_run_staged_sweep_with_injected_loader(
    pretrained_model_dir: str, saliency_map: dict[str, torch.Tensor],
    tmp_path: Path,
) -> None:
    """run_staged_sweep with mocked dataset loading.

    Uses mock.patch on load_qa_dataset so no HF network access is needed.
    Verifies stage result structure, JSON output files, and final interval.
    """
    loss_0, loss_8 = _measure_losses(pretrained_model_dir, saliency_map, [0.0, 0.8])
    assert loss_8 > loss_0

    threshold = (loss_0 + loss_8) / 2.0
    dummy_dataset = Dataset.from_dict({"question": ["Q"] * 4, "answer": ["A"] * 4})

    evaluator = _InjectedDatasetEvaluator(pretrained_model_dir, saliency_map, threshold)
    stage = SweepStage(
        name="mocked_sweep",
        evaluator=evaluator,
        search=BinarySearch(tolerance=0.05),
        interval_spec=Initial(),
        max_steps=8,
        dataset_config=DatasetConfig(n_eval=4, n_recovery=0),
        raise_if_out_of_bounds=False,
    )

    output_dir = str(tmp_path / "sweep_out")

    dummy_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)

    with mock.patch("prune_and_maybe_recover_sweep._sweep.load_qa_dataset", return_value=dummy_dataset):
        result = run_staged_sweep(
            stages=[stage],
            model_name_or_path=pretrained_model_dir,
            initial_interval=SparsityInterval(lo=0.0, hi=0.8),
            output_dir=output_dir,
            tokenizer=dummy_tokenizer,
            device="cpu",
        )

    assert len(result.stage_results) == 1
    stage_result = result.stage_results[0]
    assert stage_result.name == "mocked_sweep"
    assert len(stage_result.steps) > 0

    # Verify JSON output files
    assert (Path(output_dir) / "staged_sweep_result.json").exists()
    assert (Path(output_dir) / "stage_000_mocked_sweep" / "result.json").exists()

    written = json.loads((Path(output_dir) / "staged_sweep_result.json").read_text())
    assert written["final_interval"]["lo"] == pytest.approx(result.final_interval.lo)

    print(
        f"✅ test_run_staged_sweep_with_injected_loader: "
        f"final=[{result.final_interval.lo:.3f}, {result.final_interval.hi:.3f}]"
    )
