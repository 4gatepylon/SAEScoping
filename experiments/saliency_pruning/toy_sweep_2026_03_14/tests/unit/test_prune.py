"""
Unit tests for prune.py

Uses a tiny random model on CUDA to test:
- compute_keep_masks: mask correctness, regex filtering, taylor vs gradient
- apply_keep_masks_streaming: weight zeroing, only scored params touched
- save/restore round-trip
- prune_model end-to-end from file
- Edge cases (0%, 100%, empty saliency map)
"""

import re
import tempfile

import pytest
import torch
from torch import nn
from safetensors.torch import save_file
from transformers import PreTrainedModel, PretrainedConfig

from sae_scoping.training.weight_pruning import (
    apply_keep_masks_streaming,
    compute_keep_masks,
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
# Tests: compute_keep_masks
# ---------------------------------------------------------------------------


class TestComputeKeepMasks:
    def test_gradient_masks_are_cpu_bool(self, fake_saliency, saliency_file):
        masks = compute_keep_masks(saliency_file, sparsity_fraction=0.5)
        assert len(masks) == len(fake_saliency)
        for name, mask in masks.items():
            assert mask.device == torch.device("cpu"), f"❌ {name} mask not on CPU"
            assert mask.dtype == torch.bool, f"❌ {name} mask dtype is {mask.dtype}"
        print("✅ gradient keep-masks are CPU bool tensors")

    def test_gradient_mask_shape_matches_saliency(self, fake_saliency, saliency_file):
        masks = compute_keep_masks(saliency_file, sparsity_fraction=0.5)
        for name, mask in masks.items():
            assert mask.shape == fake_saliency[name].shape, (
                f"❌ {name}: mask shape {mask.shape} != saliency shape {fake_saliency[name].shape}"
            )
        print("✅ keep-mask shapes match saliency tensor shapes")

    def test_sparsity_zero_keeps_all(self, saliency_file, fake_saliency):
        masks = compute_keep_masks(saliency_file, sparsity_fraction=0.0)
        for name, mask in masks.items():
            assert mask.all(), f"❌ {name}: sparsity=0 should keep all weights"
        print("✅ sparsity=0.0 keeps all weights")

    def test_sparsity_one_keeps_none(self, saliency_file):
        masks = compute_keep_masks(saliency_file, sparsity_fraction=1.0)
        for name, mask in masks.items():
            assert not mask.any(), f"❌ {name}: sparsity=1.0 should zero all weights"
        print("✅ sparsity=1.0 zeros all weights")

    def test_approximate_sparsity_fraction(self, saliency_file, fake_saliency):
        masks = compute_keep_masks(saliency_file, sparsity_fraction=0.5)
        n_total = sum(m.numel() for m in masks.values())
        n_zeroed = sum(int((~m).sum().item()) for m in masks.values())
        frac = n_zeroed / n_total
        assert 0.45 <= frac <= 0.55, (
            f"❌ Expected ~50% zeroed, got {frac:.1%} ({n_zeroed}/{n_total})"
        )
        print(f"✅ sparsity=0.5 zeroed {frac:.1%} of weights")

    def test_param_regex_filters_masks(self, saliency_file):
        masks = compute_keep_masks(saliency_file, sparsity_fraction=0.5, param_regex=r"head")
        assert len(masks) == 1, f"❌ Expected 1 mask, got {len(masks)}"
        assert "head.weight" in masks
        print("✅ param_regex restricts masks to matching params")

    def test_param_regex_no_match_returns_empty(self, saliency_file):
        masks = compute_keep_masks(saliency_file, sparsity_fraction=0.5, param_regex=r"nonexistent")
        assert len(masks) == 0
        print("✅ param_regex with no matches returns empty dict")

    def test_taylor_requires_weights(self, saliency_file):
        with pytest.raises(ValueError, match="param_weights_cpu"):
            compute_keep_masks(saliency_file, sparsity_fraction=0.5, saliency_type="taylor")
        print("✅ taylor saliency without param_weights_cpu raises ValueError")

    def test_taylor_masks_differ_from_gradient(self, tiny_model, saliency_file):
        param_weights_cpu = {
            name: param.data.cpu() for name, param in tiny_model.named_parameters()
        }
        grad_masks = compute_keep_masks(saliency_file, sparsity_fraction=0.5)
        taylor_masks = compute_keep_masks(
            saliency_file, sparsity_fraction=0.5,
            saliency_type="taylor", param_weights_cpu=param_weights_cpu,
        )
        any_differ = any(
            not torch.equal(grad_masks[n], taylor_masks[n]) for n in grad_masks
        )
        assert any_differ, "❌ gradient and taylor masks should differ (weights != 1)"
        print("✅ taylor and gradient masks differ as expected")

    def test_invalid_saliency_type_raises(self, saliency_file):
        with pytest.raises(ValueError, match="Unknown saliency_type"):
            compute_keep_masks(saliency_file, sparsity_fraction=0.5, saliency_type="invalid")
        print("✅ invalid saliency_type raises ValueError")


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
# Tests: apply_keep_masks_streaming
# ---------------------------------------------------------------------------


class TestApplyKeepMasksStreaming:
    def test_all_true_mask_changes_nothing(self, tiny_model):
        original = save_original_weights(tiny_model)
        all_keep = {
            name: torch.ones(param.shape, dtype=torch.bool)
            for name, param in tiny_model.named_parameters()
        }
        n = apply_keep_masks_streaming(tiny_model, all_keep)
        assert n == 0, f"❌ Expected 0 zeroed, got {n}"
        for name, param in tiny_model.named_parameters():
            assert torch.allclose(param.data.cpu(), original[name]), f"❌ {name} changed"
        print("✅ all-True mask changes nothing and returns 0")

    def test_all_false_mask_zeros_everything(self, tiny_model):
        total = sum(p.numel() for p in tiny_model.parameters())
        all_zero = {
            name: torch.zeros(param.shape, dtype=torch.bool)
            for name, param in tiny_model.named_parameters()
        }
        n = apply_keep_masks_streaming(tiny_model, all_zero)
        assert n == total, f"❌ Expected {total} zeroed, got {n}"
        for name, param in tiny_model.named_parameters():
            assert (param.data == 0).all(), f"❌ {name} has non-zero values"
        print("✅ all-False mask zeros all weights")

    def test_only_masked_params_touched(self, tiny_model):
        original = save_original_weights(tiny_model)
        partial_mask = {
            "head.weight": torch.zeros(16, 16, dtype=torch.bool)
        }
        apply_keep_masks_streaming(tiny_model, partial_mask)
        for name, param in tiny_model.named_parameters():
            if name != "head.weight":
                assert torch.allclose(
                    param.data.cpu(), original[name]
                ), f"❌ {name} was modified but not in mask dict"
        print("✅ only params present in mask dict are touched")

    def test_return_count_matches_false_entries(self, tiny_model):
        masks = {
            name: torch.ones(param.shape, dtype=torch.bool)
            for name, param in tiny_model.named_parameters()
        }
        # Zero out exactly 10 entries in head.weight
        masks["head.weight"].view(-1)[:10] = False
        n = apply_keep_masks_streaming(tiny_model, masks)
        assert n == 10, f"❌ Expected 10 zeroed, got {n}"
        print("✅ return count equals number of False mask entries")

    def test_empty_mask_dict_returns_zero(self, tiny_model):
        n = apply_keep_masks_streaming(tiny_model, {})
        assert n == 0
        print("✅ empty mask dict returns 0 and leaves model unchanged")

    def test_lowest_saliency_values_zeroed(self, tiny_model, saliency_file):
        name = "head.weight"
        param = dict(tiny_model.named_parameters())[name]
        # Saliency = ascending integers so we know the exact ordering
        ascending = {name: torch.arange(param.numel(), dtype=torch.float32).reshape(param.shape)}
        from safetensors.torch import save_file
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            ordered_path = f.name
        try:
            save_file(ascending, ordered_path)
            masks = compute_keep_masks(ordered_path, sparsity_fraction=0.5, param_regex=r"head")
            apply_keep_masks_streaming(tiny_model, masks)
            flat_param = param.data.flatten().cpu()
            # Bottom half (indices 0..127) should be zeroed; top half kept
            n = param.numel()
            for i in range(n // 2):
                assert flat_param[i].item() == 0.0, (
                    f"❌ index {i} (low saliency) not zeroed: {flat_param[i].item()}"
                )
        finally:
            os.unlink(ordered_path)
        print("✅ lowest-saliency weights are the ones zeroed")


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
