"""
Unit tests for prune.py

Uses a tiny random model on CUDA to test:
- Saliency score computation (gradient and taylor)
- Weight zeroing at various sparsity levels
- Save/restore round-trip
- Parameter regex filtering
- Edge cases (0%, 100%, empty saliency map)
"""

import re
import tempfile

import pytest
import torch
from torch import nn
from safetensors.torch import save_file
from transformers import PreTrainedModel, PretrainedConfig

from prune import (
    apply_pruning,
    compute_saliency_scores,
    load_saliency_map,
    prune_model,
    restore_original_weights,
    save_original_weights,
)


# ---------------------------------------------------------------------------
# Tiny model fixture
# ---------------------------------------------------------------------------


class _TinyConfig(PretrainedConfig):
    model_type = "tiny_test"

    def __init__(self, hidden_size: int = 16, num_layers: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers


class _TinyModel(PreTrainedModel):
    config_class = _TinyConfig

    def __init__(self, config: _TinyConfig):
        super().__init__(config)
        layers = []
        for _ in range(config.num_layers):
            layers.append(nn.Linear(config.hidden_size, config.hidden_size, bias=False))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.post_init()


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tiny_model():
    torch.manual_seed(42)
    config = _TinyConfig(hidden_size=16, num_layers=2)
    model = _TinyModel(config).to(DEVICE)
    model.eval()
    return model


@pytest.fixture
def fake_saliency(tiny_model):
    """Create a deterministic saliency map matching the model's params."""
    torch.manual_seed(123)
    return {
        name: torch.rand(param.shape)
        for name, param in tiny_model.named_parameters()
    }


@pytest.fixture
def saliency_file(fake_saliency, tmp_path):
    """Save fake saliency to a .safetensors file."""
    path = tmp_path / "saliency.safetensors"
    save_file(fake_saliency, str(path))
    return path


# ---------------------------------------------------------------------------
# Tests: saliency scoring
# ---------------------------------------------------------------------------


class TestComputeSaliencyScores:
    def test_gradient_mode_is_abs_grad(self, tiny_model, fake_saliency):
        scores = compute_saliency_scores(
            tiny_model, fake_saliency, "gradient"
        )
        assert len(scores) == len(list(tiny_model.named_parameters()))
        for name, score in scores.items():
            expected = fake_saliency[name].float().to(DEVICE).abs()
            assert torch.allclose(score, expected), f"❌ {name}: gradient scores mismatch"
        print("✅ gradient mode returns |grad|")

    def test_taylor_mode_is_abs_grad_times_weight(self, tiny_model, fake_saliency):
        scores = compute_saliency_scores(
            tiny_model, fake_saliency, "taylor"
        )
        for name, param in tiny_model.named_parameters():
            grad = fake_saliency[name].float().to(DEVICE)
            expected = (grad * param.data.float()).abs()
            assert torch.allclose(scores[name], expected), f"❌ {name}: taylor scores mismatch"
        print("✅ taylor mode returns |grad * weight|")

    def test_invalid_saliency_type_raises(self, tiny_model, fake_saliency):
        with pytest.raises(ValueError, match="Unknown saliency_type"):
            compute_saliency_scores(tiny_model, fake_saliency, "invalid")
        print("✅ invalid saliency type raises ValueError")

    def test_param_regex_filters_params(self, tiny_model, fake_saliency):
        scores = compute_saliency_scores(
            tiny_model, fake_saliency, "gradient", param_regex=r"head"
        )
        assert len(scores) == 1, f"❌ Expected 1 scored param, got {len(scores)}"
        assert "head.weight" in scores
        print("✅ param_regex filters to matching params only")

    def test_param_regex_no_match_returns_empty(self, tiny_model, fake_saliency):
        scores = compute_saliency_scores(
            tiny_model, fake_saliency, "gradient", param_regex=r"nonexistent"
        )
        assert len(scores) == 0
        print("✅ param_regex with no matches returns empty dict")

    def test_missing_saliency_key_skipped(self, tiny_model):
        partial_saliency = {"head.weight": torch.rand(16, 16)}
        scores = compute_saliency_scores(
            tiny_model, partial_saliency, "gradient"
        )
        assert len(scores) == 1
        assert "head.weight" in scores
        print("✅ params not in saliency map are skipped")


# ---------------------------------------------------------------------------
# Tests: save / restore
# ---------------------------------------------------------------------------


class TestSaveRestore:
    def test_round_trip(self, tiny_model):
        original = save_original_weights(tiny_model)
        # Corrupt weights
        for p in tiny_model.parameters():
            p.data.zero_()
        # Restore
        restore_original_weights(tiny_model, original)
        for name, param in tiny_model.named_parameters():
            assert torch.allclose(
                param.data.cpu(), original[name]
            ), f"❌ {name}: restore failed"
        print("✅ save/restore round-trip preserves weights")

    def test_saved_weights_are_on_cpu(self, tiny_model):
        original = save_original_weights(tiny_model)
        for name, tensor in original.items():
            assert tensor.device == torch.device("cpu"), f"❌ {name} not on CPU"
        print("✅ saved weights are on CPU")

    def test_saved_weights_are_clones(self, tiny_model):
        original = save_original_weights(tiny_model)
        # Modify model — should NOT change saved copy
        for p in tiny_model.parameters():
            p.data.add_(1.0)
        for name, param in tiny_model.named_parameters():
            assert not torch.allclose(
                param.data.cpu(), original[name]
            ), f"❌ {name}: saved weights should be independent clones"
        print("✅ saved weights are independent clones")


# ---------------------------------------------------------------------------
# Tests: apply_pruning
# ---------------------------------------------------------------------------


class TestApplyPruning:
    def test_zero_sparsity_changes_nothing(self, tiny_model, fake_saliency):
        original = save_original_weights(tiny_model)
        scores = compute_saliency_scores(tiny_model, fake_saliency, "gradient")
        n = apply_pruning(tiny_model, scores, 0.0)
        assert n == 0, f"❌ Expected 0 zeroed, got {n}"
        for name, param in tiny_model.named_parameters():
            assert torch.allclose(param.data.cpu(), original[name]), f"❌ {name} changed"
        print("✅ sparsity=0.0 changes nothing")

    def test_full_sparsity_zeros_all(self, tiny_model, fake_saliency):
        scores = compute_saliency_scores(tiny_model, fake_saliency, "gradient")
        total = sum(p.numel() for p in tiny_model.parameters())
        n = apply_pruning(tiny_model, scores, 1.0)
        assert n == total, f"❌ Expected {total} zeroed, got {n}"
        for name, param in tiny_model.named_parameters():
            assert (param.data == 0).all(), f"❌ {name} has non-zero values"
        print("✅ sparsity=1.0 zeros all scored weights")

    def test_half_sparsity_zeros_approximately_half(self, tiny_model, fake_saliency):
        scores = compute_saliency_scores(tiny_model, fake_saliency, "gradient")
        total = sum(s.numel() for s in scores.values())
        n = apply_pruning(tiny_model, scores, 0.5)
        # Allow ±5% tolerance due to threshold ties
        lower = int(0.45 * total)
        upper = int(0.55 * total)
        assert lower <= n <= upper, (
            f"❌ Expected ~{total // 2} zeroed, got {n} (range [{lower}, {upper}])"
        )
        print(f"✅ sparsity=0.5 zeroed {n}/{total} weights (~{n/total:.1%})")

    def test_pruning_only_touches_scored_params(self, tiny_model):
        partial_saliency = {
            "head.weight": torch.rand(16, 16).to(DEVICE)
        }
        original = save_original_weights(tiny_model)
        apply_pruning(tiny_model, partial_saliency, 0.5)
        # Non-scored params should be unchanged
        for name, param in tiny_model.named_parameters():
            if name != "head.weight":
                assert torch.allclose(
                    param.data.cpu(), original[name]
                ), f"❌ {name} was modified but not in saliency_scores"
        print("✅ pruning only touches scored params")

    def test_pruned_values_are_lowest_saliency(self, tiny_model):
        # Use a simple saliency where we know the ordering
        torch.manual_seed(999)
        # Single param test: create scores where we know which should be pruned
        name = "head.weight"
        param = dict(tiny_model.named_parameters())[name]
        saliency = {name: torch.arange(param.numel(), dtype=torch.float32).reshape(param.shape).to(DEVICE)}
        original = param.data.clone()
        apply_pruning(tiny_model, saliency, 0.5)
        # The lowest-scoring half should be zeroed
        flat_saliency = saliency[name].flatten()
        flat_param = param.data.flatten()
        n_prune = param.numel() // 2
        threshold = torch.kthvalue(flat_saliency.cpu(), n_prune).values.item()
        for i in range(param.numel()):
            if flat_saliency[i].item() <= threshold:
                assert flat_param[i].item() == 0.0, (
                    f"❌ weight at index {i} (saliency={flat_saliency[i].item()}) "
                    f"should be zeroed but is {flat_param[i].item()}"
                )
        print("✅ lowest-saliency weights are the ones zeroed")

    def test_empty_scores_returns_zero(self, tiny_model):
        n = apply_pruning(tiny_model, {}, 0.5)
        assert n == 0
        print("✅ empty saliency_scores prunes nothing")


# ---------------------------------------------------------------------------
# Tests: prune_model (high-level convenience)
# ---------------------------------------------------------------------------


class TestPruneModel:
    def test_end_to_end_from_file(self, tiny_model, saliency_file):
        total = sum(p.numel() for p in tiny_model.parameters())
        n = prune_model(tiny_model, saliency_file, 0.3, saliency_type="gradient")
        assert 0 < n < total
        print(f"✅ prune_model end-to-end zeroed {n}/{total}")

    def test_with_param_regex(self, tiny_model, saliency_file):
        original = save_original_weights(tiny_model)
        prune_model(
            tiny_model, saliency_file, 0.5,
            saliency_type="gradient", param_regex=r"head",
        )
        # layers should be untouched
        for name, param in tiny_model.named_parameters():
            if "head" not in name:
                assert torch.allclose(
                    param.data.cpu(), original[name]
                ), f"❌ {name} was pruned despite not matching regex"
        print("✅ prune_model with param_regex only touches matching params")


# ---------------------------------------------------------------------------
# Tests: load_saliency_map
# ---------------------------------------------------------------------------


class TestLoadSaliencyMap:
    def test_load_round_trip(self, fake_saliency, saliency_file):
        loaded = load_saliency_map(saliency_file)
        assert set(loaded.keys()) == set(fake_saliency.keys())
        for name in fake_saliency:
            assert torch.allclose(loaded[name], fake_saliency[name])
        print("✅ load_saliency_map round-trips correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
