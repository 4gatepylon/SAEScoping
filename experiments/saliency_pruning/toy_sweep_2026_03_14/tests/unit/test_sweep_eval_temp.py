"""Unit tests for sweep_eval_temp.py.

All tests run on CPU using a tiny nn.Module — no HuggingFace model download required.
"""
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sweep_eval_temp import (
    _build_sweep_cmd,
    _is_run_complete,
    apply_pruning,
    assert_kept_weights_unchanged,
    assert_pruned_weights_are_zero,
    assert_zero_count_geq_target,
    build_sparsity_levels,
    compute_saliency_scores,
    restore_original_weights,
    sample_pruning_probes,
    save_generations,
    save_original_weights,
)


# ---------------------------------------------------------------------------
# Shared tiny model fixture
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Two-layer MLP with no bias; serves as a stand-in for a real LM in tests."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(16, 16, bias=False)
        self.fc2 = nn.Linear(16, 8, bias=False)


@pytest.fixture()
def tiny_model() -> _TinyModel:
    return _TinyModel()


def _make_saliency_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Uniform-random saliency tensor per parameter (same shape)."""
    torch.manual_seed(0)
    return {name: torch.rand_like(param.data) for name, param in model.named_parameters()}


# ---------------------------------------------------------------------------
# build_sparsity_levels
# ---------------------------------------------------------------------------


def test_build_sparsity_levels_uniform_grid() -> None:
    """precision=0.05 should produce exactly 21 levels in [0, 1]."""
    levels = build_sparsity_levels(0.05, None)
    assert len(levels) == 21, f"❌ Expected 21 levels, got {len(levels)}"
    assert levels[0] == 0.0, f"❌ First level should be 0.0, got {levels[0]}"
    assert levels[-1] == 1.0, f"❌ Last level should be 1.0, got {levels[-1]}"
    assert levels == sorted(levels), "❌ Levels should be sorted"
    print("✅ build_sparsity_levels: uniform grid")


def test_build_sparsity_levels_explicit_overrides_precision() -> None:
    """Explicit sparsity_levels_str overrides precision."""
    levels = build_sparsity_levels(0.05, "0.9,0.1,0.5")
    assert levels == [0.1, 0.5, 0.9], f"❌ Got {levels}"
    print("✅ build_sparsity_levels: explicit CSV overrides precision")


def test_build_sparsity_levels_coarse_grid() -> None:
    """precision=0.1 should produce 11 levels."""
    levels = build_sparsity_levels(0.1, None)
    assert len(levels) == 11, f"❌ Expected 11, got {len(levels)}"
    print("✅ build_sparsity_levels: coarse grid")


def test_build_sparsity_levels_explicit_unsorted_becomes_sorted() -> None:
    """Unsorted input is returned sorted."""
    levels = build_sparsity_levels(0.05, "0.7,0.3,0.0,1.0")
    assert levels == sorted(levels), "❌ Output is not sorted"
    print("✅ build_sparsity_levels: explicit unsorted becomes sorted")


# ---------------------------------------------------------------------------
# save_generations / _is_run_complete
# ---------------------------------------------------------------------------


def test_save_generations_creates_valid_json(tmp_path: Path) -> None:
    """save_generations writes a valid JSON file with the right content."""
    conversations = [
        [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}],
        [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}],
    ]
    save_generations(conversations, tmp_path, sparsity=0.25)
    expected = tmp_path / "generations_sparsity_0.2500.json"
    assert expected.exists(), f"❌ File not created: {expected}"
    loaded = json.loads(expected.read_text())
    assert loaded == conversations, "❌ Loaded content does not match saved content"
    print("✅ save_generations: creates valid JSON file")


def test_is_run_complete_false_when_nonexistent(tmp_path: Path) -> None:
    """Returns False for a directory that does not exist."""
    assert not _is_run_complete(tmp_path / "does_not_exist"), "❌ Should be False for nonexistent dir"
    print("✅ _is_run_complete: False for nonexistent dir")


def test_is_run_complete_false_when_empty(tmp_path: Path) -> None:
    """Returns False for an existing but empty directory."""
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()
    assert not _is_run_complete(run_dir), "❌ Should be False for empty dir"
    print("✅ _is_run_complete: False for empty dir")


def test_is_run_complete_true_after_save_generations(tmp_path: Path) -> None:
    """Returns True once save_generations has been called."""
    conversations = [[{"role": "user", "content": "Hi"}]]
    save_generations(conversations, tmp_path, sparsity=0.0)
    assert _is_run_complete(tmp_path), "❌ Should be True after saving generations"
    print("✅ _is_run_complete: True after save_generations")


# ---------------------------------------------------------------------------
# _build_sweep_cmd
# ---------------------------------------------------------------------------


def test_build_sweep_cmd_contains_required_flags(tmp_path: Path) -> None:
    """_build_sweep_cmd output contains saliency-path, saliency-type, and run-name flags."""
    sf = tmp_path / "test.safetensors"
    run_output_dir = tmp_path / "out"
    common_kwargs = dict(
        model_id="google/gemma-2-9b-it",
        dataset_name="4gate/StemQAMixture",
        dataset_subset="biology",
        n_samples=512,
        batch_size=4,
        n_generation_samples=32,
        max_seq_len=1024,
        max_new_tokens=256,
        precision=0.05,
        sparsity_levels=None,
        seed=42,
        wandb_project="test-project",
        no_generation=False,
    )
    cmd = _build_sweep_cmd(sf, "taylor", "my_run", run_output_dir, common_kwargs)
    cmd_str = " ".join(str(c) for c in cmd)
    assert "--saliency-path" in cmd_str, "❌ Missing --saliency-path"
    assert "--saliency-type" in cmd_str, "❌ Missing --saliency-type"
    assert "taylor" in cmd_str, "❌ Missing saliency-type value"
    assert "--wandb-run-name" in cmd_str, "❌ Missing --wandb-run-name"
    assert "my_run" in cmd_str, "❌ Missing run name value"
    assert "--no-generation" not in cmd_str, "❌ Should not have --no-generation when False"
    print("✅ _build_sweep_cmd: contains required flags")


def test_build_sweep_cmd_includes_no_generation_flag(tmp_path: Path) -> None:
    """--no-generation is forwarded when common_kwargs['no_generation'] is True."""
    sf = tmp_path / "test.safetensors"
    run_output_dir = tmp_path / "out"
    common_kwargs = dict(
        model_id="m", dataset_name="d", dataset_subset="s",
        n_samples=4, batch_size=4, n_generation_samples=4,
        max_seq_len=64, max_new_tokens=16, precision=0.5,
        sparsity_levels=None, seed=0, wandb_project="p", no_generation=True,
    )
    cmd = _build_sweep_cmd(sf, "gradient", "r", run_output_dir, common_kwargs)
    assert "--no-generation" in cmd, "❌ --no-generation flag not included"
    print("✅ _build_sweep_cmd: no-generation flag forwarded")


# ---------------------------------------------------------------------------
# compute_saliency_scores
# ---------------------------------------------------------------------------


def test_compute_saliency_gradient_equals_abs_of_tensor(tiny_model: _TinyModel) -> None:
    """Gradient saliency score is the absolute value of the saliency tensor."""
    saliency_tensors = _make_saliency_tensors(tiny_model)
    scores = compute_saliency_scores(tiny_model, saliency_tensors, "gradient")
    for name, score in scores.items():
        expected = saliency_tensors[name].float().abs()
        assert torch.allclose(score, expected), f"❌ Gradient score mismatch for '{name}'"
    print("✅ compute_saliency_scores: gradient = |tensor|")


def test_compute_saliency_taylor_equals_abs_grad_times_weight(tiny_model: _TinyModel) -> None:
    """Taylor saliency score is |grad_tensor * weight|."""
    saliency_tensors = _make_saliency_tensors(tiny_model)
    scores = compute_saliency_scores(tiny_model, saliency_tensors, "taylor")
    for name, param in tiny_model.named_parameters():
        expected = (saliency_tensors[name].float() * param.data.float()).abs()
        assert torch.allclose(scores[name], expected), f"❌ Taylor score mismatch for '{name}'"
    print("✅ compute_saliency_scores: taylor = |grad * weight|")


def test_compute_saliency_unknown_type_raises(tiny_model: _TinyModel) -> None:
    """Unknown saliency type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown saliency_type"):
        compute_saliency_scores(tiny_model, {}, "unsupported_type")
    print("✅ compute_saliency_scores: raises on unknown type")


def test_compute_saliency_skips_missing_tensors(tiny_model: _TinyModel) -> None:
    """Parameters absent from saliency_tensors are silently skipped."""
    scores = compute_saliency_scores(tiny_model, {}, "gradient")
    assert scores == {}, "❌ Should return empty dict when no tensors provided"
    print("✅ compute_saliency_scores: skips missing tensors")


# ---------------------------------------------------------------------------
# save_original_weights / restore_original_weights
# ---------------------------------------------------------------------------


def test_save_original_weights_clones_to_cpu(tiny_model: _TinyModel) -> None:
    """Saved weights are CPU tensors that do not share storage with the model."""
    original = save_original_weights(tiny_model)
    for name, param in tiny_model.named_parameters():
        assert name in original, f"❌ Missing key '{name}'"
        saved = original[name]
        assert saved.device.type == "cpu", f"❌ Saved tensor not on CPU: {saved.device}"
        assert torch.equal(saved, param.data.cpu()), f"❌ Saved value differs for '{name}'"
        # Mutate model and verify saved copy is unaffected
        param.data.fill_(999.0)
        assert not torch.equal(saved, param.data.cpu()), "❌ Saved tensor is an alias (not a clone)"
    print("✅ save_original_weights: CPU clone, independent of model")


def test_restore_original_weights_exact_round_trip(tiny_model: _TinyModel) -> None:
    """After modification, restore_original_weights returns model to original state."""
    original = save_original_weights(tiny_model)
    for param in tiny_model.parameters():
        param.data.zero_()
    restore_original_weights(tiny_model, original)
    for name, param in tiny_model.named_parameters():
        assert torch.equal(param.data.cpu(), original[name]), f"❌ Restore failed for '{name}'"
    print("✅ restore_original_weights: exact round-trip")


# ---------------------------------------------------------------------------
# sample_pruning_probes
# ---------------------------------------------------------------------------


def test_sample_pruning_probes_returns_correct_shapes(tiny_model: _TinyModel) -> None:
    """sample_pruning_probes returns (indices, values) with expected shapes per param."""
    saliency_scores = _make_saliency_tensors(tiny_model)
    rng = torch.Generator()
    rng.manual_seed(0)
    n_probe = 10
    probe = sample_pruning_probes(tiny_model, saliency_scores, n_probe, rng)
    for name in saliency_scores:
        assert name in probe, f"❌ Missing probe for '{name}'"
        idx, vals = probe[name]
        assert idx.shape == (n_probe,), f"❌ idx shape wrong for '{name}': {idx.shape}"
        assert vals.shape == (n_probe,), f"❌ vals shape wrong for '{name}': {vals.shape}"
    print("✅ sample_pruning_probes: correct shapes")


def test_sample_pruning_probes_values_match_model(tiny_model: _TinyModel) -> None:
    """Probed values match the actual model parameter values at those indices."""
    saliency_scores = _make_saliency_tensors(tiny_model)
    rng = torch.Generator()
    rng.manual_seed(7)
    probe = sample_pruning_probes(tiny_model, saliency_scores, 8, rng)
    for name, param in tiny_model.named_parameters():
        if name not in probe:
            continue
        idx, vals = probe[name]
        actual = param.data.float().cpu().flatten()[idx]
        assert torch.allclose(vals, actual), f"❌ Probe values don't match param for '{name}'"
    print("✅ sample_pruning_probes: values match model parameters")


def test_sample_pruning_probes_skips_params_not_in_saliency(tiny_model: _TinyModel) -> None:
    """Parameters absent from saliency_scores are not included in the probe."""
    rng = torch.Generator()
    rng.manual_seed(0)
    probe = sample_pruning_probes(tiny_model, {}, 8, rng)
    assert probe == {}, "❌ Should return empty probe when no saliency scores provided"
    print("✅ sample_pruning_probes: skips params not in saliency_scores")


# ---------------------------------------------------------------------------
# assert_kept_weights_unchanged
# ---------------------------------------------------------------------------


def test_assert_kept_weights_unchanged_passes(tiny_model: _TinyModel) -> None:
    """No assertion when kept weights are genuinely unchanged."""
    saliency_scores = _make_saliency_tensors(tiny_model)
    rng = torch.Generator()
    rng.manual_seed(0)
    pre_probe = sample_pruning_probes(tiny_model, saliency_scores, 16, rng)
    masks = {name: torch.ones_like(s, dtype=torch.bool) for name, s in saliency_scores.items()}
    assert_kept_weights_unchanged(tiny_model, saliency_scores, masks, pre_probe)
    print("✅ assert_kept_weights_unchanged: passes when weights unchanged")


def test_assert_kept_weights_unchanged_fails_on_modification(tiny_model: _TinyModel) -> None:
    """AssertionError raised if a kept weight is modified after the probe."""
    saliency_scores = _make_saliency_tensors(tiny_model)
    rng = torch.Generator()
    rng.manual_seed(0)
    pre_probe = sample_pruning_probes(tiny_model, saliency_scores, 64, rng)
    # Keep all weights but then modify the model
    masks = {name: torch.ones_like(s, dtype=torch.bool) for name, s in saliency_scores.items()}
    for param in tiny_model.parameters():
        param.data.fill_(999.0)
    with pytest.raises(AssertionError, match="kept weights changed"):
        assert_kept_weights_unchanged(tiny_model, saliency_scores, masks, pre_probe)
    print("✅ assert_kept_weights_unchanged: raises AssertionError on modification")


# ---------------------------------------------------------------------------
# assert_pruned_weights_are_zero
# ---------------------------------------------------------------------------


def test_assert_pruned_weights_are_zero_passes(tiny_model: _TinyModel) -> None:
    """No assertion when all pruned weights are zero."""
    saliency_scores = _make_saliency_tensors(tiny_model)
    masks = {name: torch.ones_like(s, dtype=torch.bool) for name, s in saliency_scores.items()}
    assert_pruned_weights_are_zero(tiny_model, saliency_scores, masks)
    print("✅ assert_pruned_weights_are_zero: passes when no weights pruned")


def test_assert_pruned_weights_are_zero_fails_when_nonzero(tiny_model: _TinyModel) -> None:
    """AssertionError raised if a pruned position is nonzero."""
    saliency_scores = _make_saliency_tensors(tiny_model)
    # Mark all as pruned (mask=False) but leave weights nonzero
    masks = {name: torch.zeros_like(s, dtype=torch.bool) for name, s in saliency_scores.items()}
    with pytest.raises(AssertionError, match="pruned weights in"):
        assert_pruned_weights_are_zero(tiny_model, saliency_scores, masks)
    print("✅ assert_pruned_weights_are_zero: raises AssertionError on nonzero pruned weight")


# ---------------------------------------------------------------------------
# assert_zero_count_geq_target
# ---------------------------------------------------------------------------


def test_assert_zero_count_geq_target_passes(tiny_model: _TinyModel) -> None:
    """No assertion when zero count >= n_prune."""
    saliency_scores = _make_saliency_tensors(tiny_model)
    for param in tiny_model.parameters():
        param.data.zero_()
    total = sum(s.numel() for s in saliency_scores.values())
    assert_zero_count_geq_target(tiny_model, saliency_scores, n_prune=total, sparsity_fraction=1.0)
    print("✅ assert_zero_count_geq_target: passes when all weights zeroed")


def test_assert_zero_count_geq_target_fails_when_too_few_zeros(tiny_model: _TinyModel) -> None:
    """AssertionError raised when zero count < n_prune."""
    saliency_scores = _make_saliency_tensors(tiny_model)
    total = sum(s.numel() for s in saliency_scores.values())
    with pytest.raises(AssertionError, match="zeros found"):
        assert_zero_count_geq_target(
            tiny_model, saliency_scores, n_prune=total, sparsity_fraction=1.0
        )
    print("✅ assert_zero_count_geq_target: raises when too few zeros")


# ---------------------------------------------------------------------------
# apply_pruning end-to-end
# ---------------------------------------------------------------------------


def test_apply_pruning_zero_sparsity_is_noop(tiny_model: _TinyModel) -> None:
    """sparsity=0.0 returns 0 and does not touch any weights."""
    saliency_scores = _make_saliency_tensors(tiny_model)
    original = {n: p.data.clone() for n, p in tiny_model.named_parameters()}
    n_zeroed = apply_pruning(tiny_model, saliency_scores, 0.0)
    assert n_zeroed == 0, f"❌ Expected 0 weights zeroed, got {n_zeroed}"
    for name, param in tiny_model.named_parameters():
        assert torch.equal(param.data, original[name]), f"❌ Weights changed for '{name}'"
    print("✅ apply_pruning: sparsity=0.0 is a no-op")


def test_apply_pruning_full_sparsity_zeros_all(tiny_model: _TinyModel) -> None:
    """sparsity=1.0 zeros all scored weights."""
    saliency_scores = _make_saliency_tensors(tiny_model)
    total = sum(s.numel() for s in saliency_scores.values())
    n_zeroed = apply_pruning(tiny_model, saliency_scores, 1.0)
    for name, param in tiny_model.named_parameters():
        if name in saliency_scores:
            assert param.data.count_nonzero().item() == 0, f"❌ Weights still nonzero in '{name}'"
    assert n_zeroed >= int(0.99 * total), f"❌ Only {n_zeroed}/{total} zeroed at sparsity=1.0"
    print("✅ apply_pruning: sparsity=1.0 zeros all scored weights")


def test_apply_pruning_half_sparsity_actual_close_to_target(tiny_model: _TinyModel) -> None:
    """apply_pruning at sparsity=0.5 zeros approximately 50% of weights."""
    saliency_scores = _make_saliency_tensors(tiny_model)
    total = sum(s.numel() for s in saliency_scores.values())
    n_zeroed = apply_pruning(tiny_model, saliency_scores, 0.5)
    actual_frac = n_zeroed / total
    assert 0.45 <= actual_frac <= 0.55, (
        f"❌ Expected ~50% zeroed, got {actual_frac:.1%} ({n_zeroed}/{total})"
    )
    print(f"✅ apply_pruning: sparsity=0.5 zeroed {actual_frac:.1%} of weights")


def test_apply_pruning_assertions_pass_on_valid_pruning(tiny_model: _TinyModel) -> None:
    """apply_pruning runs without any assertion errors on normal inputs."""
    saliency_scores = _make_saliency_tensors(tiny_model)
    apply_pruning(tiny_model, saliency_scores, 0.3)
    apply_pruning(tiny_model, saliency_scores, 0.7)
    print("✅ apply_pruning: all internal assertions pass")


def test_apply_pruning_restore_weights_cycle(tiny_model: _TinyModel) -> None:
    """apply_pruning followed by restore_original_weights returns model to original state."""
    saliency_scores = _make_saliency_tensors(tiny_model)
    original_weights = save_original_weights(tiny_model)
    apply_pruning(tiny_model, saliency_scores, 0.6)
    restore_original_weights(tiny_model, original_weights)
    for name, param in tiny_model.named_parameters():
        assert torch.equal(param.data.cpu(), original_weights[name]), (
            f"❌ Weights not restored for '{name}'"
        )
    print("✅ apply_pruning + restore: full round-trip")
