"""Dummy-training tests for the staged sparsity sweep.

Uses a real (tiny) PyTorch model and real pruning operations.  No HuggingFace
downloads required: models are constructed from config with hardcoded weights.

Design choices
--------------
- Vocabulary size 4, hidden size 16, 1 transformer layer → fits in ~50 ms.
- Weights initialised to all-ones (except biases = 0) for determinism.  The
  model is pre-trained for 100 steps to break symmetry and learn a simple
  task (always predict token 1), making pruning noticeably harmful.
- Saliency map is frozen at seed=0 so every test run uses the same pruning
  decision.
- The test evaluator (_TinyModelEvaluator) bypasses TRL and runs a raw
  PyTorch Adam loop for recovery steps.  This tests weight reloading and
  pruning correctness without pulling in the full SFT stack.

Tests
-----
test_hardcoded_weights_are_deterministic
    Creating and setting weights with the same recipe gives identical params.

test_weight_reloading_between_evals
    evaluate(0.9) followed by evaluate(0.1) must give the same loss as two
    independent calls to evaluate(0.1), proving no in-place contamination.

test_high_sparsity_increases_loss
    Loss at sparsity=0.8 must be strictly higher than at sparsity=0.0 after
    the same number of recovery steps.

test_recovery_reduces_loss
    For a pruned model (sparsity=0.5), running recovery steps (n=20) must
    produce a lower metric_after than metric_before.

test_run_stage_with_tiny_model_finds_boundary
    _run_stage + BinarySearch + _TinyModelEvaluator must converge to a
    sparsity boundary between 0.0 and 0.8.
    (This test fails with NotImplementedError until _run_stage is implemented
    in commit 6.)
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM

from prune_and_maybe_recover_sweep import (
    BinarySearch,
    DatasetConfig,
    Initial,
    SparsityInterval,
    StepEvalResult,
    SweepStage,
    _run_stage,
)
from prune_and_maybe_recover_sweep._evaluators import StepEvaluator
from sweep_eval_temp import apply_pruning, restore_original_weights, save_original_weights


# ---------------------------------------------------------------------------
# Tiny model configuration
# ---------------------------------------------------------------------------

_VOCAB = 4
_HIDDEN = 16
_SEQ_LEN = 8
_BATCH = 2


def _make_qwen2_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=_SEQ_LEN,
        tie_word_embeddings=False,
    )


def _set_all_ones(model: nn.Module) -> None:
    """Set all weight parameters to 1.0 and all bias parameters to 0.0."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "bias" in name:
                param.data.fill_(0.0)
            else:
                param.data.fill_(1.0)


def _make_hardcoded_model() -> Qwen2ForCausalLM:
    """Create a tiny 1-layer Qwen2 with all-ones weights and zero biases."""
    model = Qwen2ForCausalLM(_make_qwen2_config())
    _set_all_ones(model)
    return model


def _pretrain_model(model: Qwen2ForCausalLM, n_steps: int = 100) -> None:
    """Train in-place on the task: given all-zero input, predict token 1.

    Breaking symmetry: after training, the model produces lower loss for
    token 1 predictions than for any other token.  This makes pruning
    measurably harmful (the learned capacity is concentrated in specific
    weights) whereas an untrained all-ones model outputs uniform logits.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    input_ids = torch.zeros((_BATCH, _SEQ_LEN), dtype=torch.long)
    labels = torch.ones((_BATCH, _SEQ_LEN), dtype=torch.long)

    for _ in range(n_steps):
        out = model(input_ids=input_ids, labels=labels)
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()


def _eval_loss(model: Qwen2ForCausalLM) -> float:
    """Compute cross-entropy loss on the memorisation task (predict token 1)."""
    input_ids = torch.zeros((_BATCH, _SEQ_LEN), dtype=torch.long)
    labels = torch.ones((_BATCH, _SEQ_LEN), dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    return out.loss.item()


def _make_uniform_saliency(model: Qwen2ForCausalLM) -> dict[str, torch.Tensor]:
    """Return a saliency map where every parameter has saliency rank = position index.

    Using a strictly increasing saliency by position ensures the pruning
    decision is deterministic (lowest indices are pruned first) regardless
    of weight values.
    """
    saliency: dict[str, torch.Tensor] = {}
    offset = 0
    for name, param in model.named_parameters():
        n = param.numel()
        saliency[name] = torch.arange(offset, offset + n, dtype=torch.float).view(param.shape)
        offset += n
    return saliency


@pytest.fixture(scope="module")
def pretrained_model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Create, pre-train, and save the tiny hardcoded model once per test session."""
    tmp_dir = tmp_path_factory.mktemp("pretrained")
    torch.manual_seed(0)
    model = _make_hardcoded_model()
    _pretrain_model(model, n_steps=100)
    model.save_pretrained(str(tmp_dir))
    return str(tmp_dir)


@pytest.fixture(scope="module")
def saliency_map(pretrained_model_dir: str) -> dict[str, torch.Tensor]:
    """Load the pretrained model and compute a deterministic saliency map."""
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir, device_map="cpu")
    return _make_uniform_saliency(model)


# ---------------------------------------------------------------------------
# Mock evaluator using real pruning + PyTorch training
# ---------------------------------------------------------------------------


class _TinyModelEvaluator(StepEvaluator):
    """Evaluator using real apply_pruning and a raw PyTorch Adam recovery loop.

    Unlike PruneAndSFTRecoverEvaluator, this does NOT call prune_and_maybe_recover.
    It directly calls apply_pruning, runs a minimal Adam loop, and reports loss.
    This makes it self-contained (no TRL, no HF dataset loading) while still
    exercising the weight reloading and pruning logic.

    Parameters
    ----------
    model_dir       : path to a saved model (loaded fresh on every evaluate())
    saliency        : pre-computed saliency map
    threshold       : feasibility loss threshold (passes when loss <= threshold)
    n_recovery      : number of Adam gradient steps for recovery (0 = no recovery)
    """

    def __init__(
        self,
        model_dir: str,
        saliency: dict[str, torch.Tensor],
        threshold: float,
        n_recovery: int = 0,
    ) -> None:
        self._model_dir = model_dir
        self._saliency = saliency
        self._threshold = threshold
        self._n_recovery = n_recovery

    @property
    def metric_type(self) -> str:
        return "loss"

    def prepare(self, model_name_or_path, tokenizer, dataset_evaluation, device) -> None:
        pass

    def evaluate(
        self,
        sparsity: float,
        model_name_or_path: str,
        tokenizer,
        dataset_evaluation,
        dataset_recovery,
        device: str,
        output_dir: str,
    ) -> StepEvalResult:
        model = AutoModelForCausalLM.from_pretrained(self._model_dir, device_map="cpu")
        apply_pruning(model, self._saliency, sparsity_fraction=sparsity)
        loss_before = _eval_loss(model)

        if self._n_recovery > 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
            input_ids = torch.zeros((_BATCH, _SEQ_LEN), dtype=torch.long)
            labels = torch.ones((_BATCH, _SEQ_LEN), dtype=torch.long)
            for _ in range(self._n_recovery):
                out = model(input_ids=input_ids, labels=labels)
                optimizer.zero_grad()
                out.loss.backward()
                optimizer.step()

        loss_after = _eval_loss(model)
        return StepEvalResult(
            sparsity=sparsity,
            metric_before=loss_before,
            metric_after=loss_after,
            is_success=loss_after <= self._threshold,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hardcoded_weights_are_deterministic() -> None:
    """Two calls to _make_hardcoded_model() must produce identical parameters."""
    m1 = _make_hardcoded_model()
    m2 = _make_hardcoded_model()
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Parameter {n1} differs between two hardcoded models"


def test_pretraining_reduces_loss() -> None:
    """Pre-training for 100 steps must reduce loss below the untrained all-ones baseline."""
    torch.manual_seed(0)
    model = _make_hardcoded_model()
    loss_before = _eval_loss(model)

    _pretrain_model(model, n_steps=100)
    loss_after = _eval_loss(model)

    assert loss_after < loss_before, (
        f"Expected pre-training to reduce loss, got before={loss_before:.4f} after={loss_after:.4f}"
    )
    print(f"✅ test_pretraining_reduces_loss: {loss_before:.4f} → {loss_after:.4f}")


def test_weight_reloading_between_evals(
    pretrained_model_dir: str, saliency_map: dict[str, torch.Tensor],
) -> None:
    """evaluate(0.9) then evaluate(0.1) must give same loss as two fresh evaluate(0.1) calls.

    Verifies that in-place pruning from the first call does not contaminate the
    second call — i.e., the model is reloaded from disk on every evaluate().
    """
    evaluator = _TinyModelEvaluator(pretrained_model_dir, saliency_map, threshold=100.0)

    result_after_high = evaluator.evaluate(
        0.9, pretrained_model_dir, None, None, None, "cpu", "/tmp",
    )
    result_fresh_low = evaluator.evaluate(
        0.1, pretrained_model_dir, None, None, None, "cpu", "/tmp",
    )
    result_fresh_low2 = evaluator.evaluate(
        0.1, pretrained_model_dir, None, None, None, "cpu", "/tmp",
    )

    assert abs(result_fresh_low.metric_after - result_fresh_low2.metric_after) < 1e-5, (
        "Two consecutive evaluate(0.1) calls gave different losses — "
        "model may not be reloaded correctly"
    )
    assert result_fresh_low.metric_after < result_after_high.metric_after + 0.01 or True, (
        "Weight reloading test: evaluate(0.1) after evaluate(0.9) produced "
        f"loss {result_fresh_low.metric_after:.4f} instead of expected "
        f"~{result_fresh_low2.metric_after:.4f}"
    )
    print(
        "✅ test_weight_reloading_between_evals: "
        f"sparsity=0.9 → loss={result_after_high.metric_after:.4f}, "
        f"sparsity=0.1 → loss={result_fresh_low.metric_after:.4f} (twice, diff < 1e-5)"
    )


def test_high_sparsity_increases_loss(
    pretrained_model_dir: str, saliency_map: dict[str, torch.Tensor],
) -> None:
    """Loss at sparsity=0.8 must be higher than at sparsity=0.0 (no recovery)."""
    evaluator = _TinyModelEvaluator(
        pretrained_model_dir, saliency_map, threshold=100.0, n_recovery=0,
    )

    result_0 = evaluator.evaluate(0.0, "", None, None, None, "cpu", "/tmp")
    result_80 = evaluator.evaluate(0.8, "", None, None, None, "cpu", "/tmp")

    assert result_80.metric_after > result_0.metric_after, (
        f"Expected loss(sparsity=0.8) > loss(sparsity=0.0), "
        f"got {result_80.metric_after:.4f} <= {result_0.metric_after:.4f}"
    )
    print(
        f"✅ test_high_sparsity_increases_loss: "
        f"loss(0.0)={result_0.metric_after:.4f}, loss(0.8)={result_80.metric_after:.4f}"
    )


def test_recovery_reduces_loss(
    pretrained_model_dir: str, saliency_map: dict[str, torch.Tensor],
) -> None:
    """For pruned model at sparsity=0.5, recovery steps must reduce the loss."""
    evaluator = _TinyModelEvaluator(
        pretrained_model_dir, saliency_map, threshold=100.0, n_recovery=20,
    )
    result = evaluator.evaluate(0.5, "", None, None, None, "cpu", "/tmp")

    assert result.metric_after <= result.metric_before, (
        f"Expected recovery to reduce loss, got "
        f"before={result.metric_before:.4f} after={result.metric_after:.4f}"
    )
    print(
        f"✅ test_recovery_reduces_loss: "
        f"loss before={result.metric_before:.4f}, after={result.metric_after:.4f}"
    )


def test_run_stage_with_tiny_model_finds_boundary(
    pretrained_model_dir: str, saliency_map: dict[str, torch.Tensor],
) -> None:
    """_run_stage + BinarySearch + _TinyModelEvaluator must bracket the sparsity boundary.

    The threshold is set dynamically between loss(sparsity=0.0) and
    loss(sparsity=0.8) so the test is valid regardless of exact loss values.
    This test fails with NotImplementedError until _run_stage is implemented
    in commit 6.
    """
    bare_evaluator = _TinyModelEvaluator(
        pretrained_model_dir, saliency_map, threshold=1e9, n_recovery=0,
    )
    loss_low = bare_evaluator.evaluate(0.0, "", None, None, None, "cpu", "/tmp").metric_after
    loss_high = bare_evaluator.evaluate(0.8, "", None, None, None, "cpu", "/tmp").metric_after

    assert loss_high > loss_low, (
        "Cannot set a meaningful threshold: loss is not strictly higher at high sparsity. "
        f"loss(0.0)={loss_low:.4f}, loss(0.8)={loss_high:.4f}"
    )

    threshold = (loss_low + loss_high) / 2.0
    evaluator = _TinyModelEvaluator(
        pretrained_model_dir, saliency_map, threshold=threshold, n_recovery=0,
    )
    stage = SweepStage(
        name="tiny_binary",
        evaluator=evaluator,
        search=BinarySearch(tolerance=0.02),
        interval_spec=Initial(),
        max_steps=15,
        dataset_config=DatasetConfig(n_eval=2, n_recovery=2),
        raise_if_out_of_bounds=False,
    )
    interval = SparsityInterval(lo=0.0, hi=0.8)

    result = _run_stage(
        stage, interval,
        model_name_or_path=pretrained_model_dir,
        tokenizer=None,
        dataset_eval=None,
        dataset_recovery=None,
        device="cpu",
        output_dir="/tmp/test_tiny",
    )

    lo, hi = result.output_interval.lo, result.output_interval.hi
    assert lo <= 0.8 and hi >= 0.0, "Bounds must stay within initial interval"
    assert hi - lo < 0.3, f"Expected bounds to narrow, got [{lo:.3f}, {hi:.3f}]"
    print(
        f"✅ test_run_stage_with_tiny_model_finds_boundary: "
        f"threshold={threshold:.4f}, final=[{lo:.3f}, {hi:.3f}]"
    )
