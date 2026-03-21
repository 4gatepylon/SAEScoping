"""Integration test for the pruning pipeline in sweep_eval_temp.py.

Uses Qwen2.5-Math-1.5B-Instruct truncated to 1 transformer layer so the
test runs in seconds on CPU.  Run with:

    python tests/test_sweep_pruning.py
"""
import warnings

import torch
from transformers import AutoModelForCausalLM

from sweep_eval_temp import (
    apply_pruning,
    compute_saliency_scores,
    restore_original_weights,
    save_original_weights,
)

_MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"


def _load_tiny_model() -> AutoModelForCausalLM:
    """Load the model and strip it down to a single transformer layer for speed."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        warnings.warn("⚠️ CUDA not available, running on CPU — this may be slow")
    model = AutoModelForCausalLM.from_pretrained(_MODEL_ID, device_map=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # Keep only the last transformer layer to make the test fast
    model.model.layers = model.model.layers[-1:]
    return model


def _fake_saliency(model: AutoModelForCausalLM, seed: int) -> dict[str, torch.Tensor]:
    """Random saliency tensors for every named parameter regardless of requires_grad.

    In production the saliency map is computed (with requires_grad=True) and
    saved to a .safetensors file BEFORE the sweep freezes the model.  This
    helper simulates that file so tests do not depend on requires_grad state.
    """
    torch.manual_seed(seed)
    return {name: torch.rand_like(param.data) for name, param in model.named_parameters()}


def test_save_restore_weights() -> None:
    print("=" * 80)
    print("Integration test: save_original_weights / restore_original_weights")
    model = _load_tiny_model()

    original = save_original_weights(model)
    assert len(original) > 0, "❌ No weights saved"

    # Corrupt the model weights
    for param in model.parameters():
        param.data.zero_()

    # Restore and verify
    restore_original_weights(model, original)
    for name, param in model.named_parameters():
        assert torch.equal(param.data.cpu(), original[name]), f"❌ Mismatch after restore for '{name}'"

    print("✅ save_original_weights / restore_original_weights: exact round-trip on real model")


def test_compute_saliency_scores_gradient() -> None:
    print("=" * 80)
    print("Integration test: compute_saliency_scores (gradient)")
    model = _load_tiny_model()

    saliency_tensors = _fake_saliency(model, seed=0)
    scores = compute_saliency_scores(model, saliency_tensors, "gradient")

    assert len(scores) == len(saliency_tensors), (
        f"❌ Expected {len(saliency_tensors)} scores, got {len(scores)}"
    )
    for name, score in scores.items():
        assert (score >= 0).all(), f"❌ Negative gradient saliency score for '{name}'"

    print(f"✅ compute_saliency_scores gradient: {len(scores)} tensors, all non-negative")


def test_compute_saliency_scores_taylor() -> None:
    print("=" * 80)
    print("Integration test: compute_saliency_scores (taylor)")
    model = _load_tiny_model()

    saliency_tensors = _fake_saliency(model, seed=1)
    scores = compute_saliency_scores(model, saliency_tensors, "taylor")

    assert len(scores) == len(saliency_tensors), (
        f"❌ Expected {len(saliency_tensors)} scores, got {len(scores)}"
    )
    for name, score in scores.items():
        assert (score >= 0).all(), f"❌ Negative Taylor saliency score for '{name}'"

    print(f"✅ compute_saliency_scores taylor: {len(scores)} tensors, all non-negative")


def test_apply_pruning_50_percent_on_real_model() -> None:
    print("=" * 80)
    print("Integration test: apply_pruning at 50% on real model")
    model = _load_tiny_model()

    saliency_tensors = _fake_saliency(model, seed=42)
    saliency_scores = compute_saliency_scores(model, saliency_tensors, "taylor")
    total_scored = sum(s.numel() for s in saliency_scores.values())

    n_zeroed = apply_pruning(model, saliency_scores, sparsity_fraction=0.5)
    actual_frac = n_zeroed / total_scored

    assert 0.45 <= actual_frac <= 0.55, (
        f"❌ Expected ~50% zeroed, got {actual_frac:.1%} ({n_zeroed}/{total_scored})"
    )
    print(f"✅ apply_pruning 50% on real model: zeroed {actual_frac:.1%} ({n_zeroed:,}/{total_scored:,})")


def test_apply_pruning_all_assertions_pass_on_real_model() -> None:
    print("=" * 80)
    print("Integration test: apply_pruning all internal assertions pass")
    model = _load_tiny_model()

    saliency_tensors = _fake_saliency(model, seed=7)
    saliency_scores = compute_saliency_scores(model, saliency_tensors, "gradient")

    # Snapshot original weights once; restore from this before each level so
    # every iteration runs on clean unpruned weights, not cumulative pruning.
    original_weights = save_original_weights(model)
    for sparsity in (0.0, 0.1, 0.5, 0.9, 1.0):
        restore_original_weights(model, original_weights)
        n = apply_pruning(model, saliency_scores, sparsity_fraction=sparsity)
        print(f"  sparsity={sparsity:.0%}: zeroed {n:,} weights — ✅ assertions passed")

    print("✅ apply_pruning: all internal assertions pass at all sparsity levels")


def test_full_prune_restore_cycle_on_real_model() -> None:
    print("=" * 80)
    print("Integration test: prune → restore cycle on real model")
    model = _load_tiny_model()
    original_weights = save_original_weights(model)

    saliency_tensors = _fake_saliency(model, seed=99)
    saliency_scores = compute_saliency_scores(model, saliency_tensors, "taylor")

    for sparsity in (0.3, 0.7):
        apply_pruning(model, saliency_scores, sparsity_fraction=sparsity)
        restore_original_weights(model, original_weights)
        for name, param in model.named_parameters():
            assert torch.equal(param.data.cpu(), original_weights[name]), (
                f"❌ Weights not restored for '{name}' after sparsity={sparsity}"
            )

    print("✅ prune → restore cycle: model returns to original weights correctly")


def main() -> None:
    test_save_restore_weights()
    test_compute_saliency_scores_gradient()
    test_compute_saliency_scores_taylor()
    test_apply_pruning_50_percent_on_real_model()
    test_apply_pruning_all_assertions_pass_on_real_model()
    test_full_prune_restore_cycle_on_real_model()
    print("\n" + "=" * 80)
    print("✅ All integration tests passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
