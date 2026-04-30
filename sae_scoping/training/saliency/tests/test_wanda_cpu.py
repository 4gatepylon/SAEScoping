"""
CPU unit tests for Wanda pruning using a tiny Gemma2 model.

No GPU needed. Runs in <30s.

Usage:
  python -m pytest sae_scoping/training/saliency/tests/test_wanda_cpu.py -v
"""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@pytest.fixture(scope="module")
def tiny_gemma2():
    """Create a tiny Gemma2 model on CPU for testing."""
    config = AutoConfig.from_pretrained("google/gemma-2-2b-it")
    # Shrink to tiny size
    config.num_hidden_layers = 2
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    # Keep real vocab_size so tokenizer IDs are valid
    # config.vocab_size is inherited from gemma-2-2b-it (256000)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    """Load the real tokenizer (lightweight, just for chat template)."""
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    return tok


@pytest.fixture(scope="module")
def calibration_texts(tokenizer):
    """Generate synthetic calibration texts."""
    texts = []
    for i in range(4):
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": f"What is topic {i}?"},
                {"role": "assistant", "content": f"Topic {i} is about things."},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return texts


class TestComputeWandaSaliency:
    def test_returns_dict(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.wanda import compute_wanda_saliency

        saliency = compute_wanda_saliency(
            tiny_gemma2,
            tokenizer,
            calibration_texts,
            max_seq_len=64,
        )
        assert isinstance(saliency, dict)
        assert len(saliency) > 0

    def test_keys_are_weight_params(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.wanda import compute_wanda_saliency

        saliency = compute_wanda_saliency(
            tiny_gemma2,
            tokenizer,
            calibration_texts,
            max_seq_len=64,
        )
        # All keys should end with .weight and correspond to linear layers
        for key in saliency:
            assert key.endswith(".weight"), f"Key {key} doesn't end with .weight"

    def test_shapes_match_weights(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.wanda import compute_wanda_saliency, _find_linear_layers

        saliency = compute_wanda_saliency(
            tiny_gemma2,
            tokenizer,
            calibration_texts,
            max_seq_len=64,
        )
        linear_layers = _find_linear_layers(tiny_gemma2)
        for name, scores in saliency.items():
            layer_name = name.removesuffix(".weight")
            assert layer_name in linear_layers, f"{layer_name} not found in model"
            assert scores.shape == linear_layers[layer_name].weight.shape

    def test_scores_nonnegative(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.wanda import compute_wanda_saliency

        saliency = compute_wanda_saliency(
            tiny_gemma2,
            tokenizer,
            calibration_texts,
            max_seq_len=64,
        )
        for name, scores in saliency.items():
            assert (scores >= 0).all(), f"Negative scores in {name}"

    def test_scores_on_cpu(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.wanda import compute_wanda_saliency

        saliency = compute_wanda_saliency(
            tiny_gemma2,
            tokenizer,
            calibration_texts,
            max_seq_len=64,
        )
        for name, scores in saliency.items():
            assert scores.device == torch.device("cpu")


class TestComputeWandaMasks:
    def test_mask_shapes(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.wanda import compute_wanda_saliency, compute_wanda_masks

        saliency = compute_wanda_saliency(
            tiny_gemma2,
            tokenizer,
            calibration_texts,
            max_seq_len=64,
        )
        masks = compute_wanda_masks(saliency, sparsity=0.5)
        for name in saliency:
            assert name in masks
            assert masks[name].shape == saliency[name].shape

    def test_masks_are_bool(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.wanda import compute_wanda_saliency, compute_wanda_masks

        saliency = compute_wanda_saliency(
            tiny_gemma2,
            tokenizer,
            calibration_texts,
            max_seq_len=64,
        )
        masks = compute_wanda_masks(saliency, sparsity=0.3)
        for name, mask in masks.items():
            assert mask.dtype == torch.bool

    def test_per_row_sparsity(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.wanda import compute_wanda_saliency, compute_wanda_masks

        target = 0.4
        saliency = compute_wanda_saliency(
            tiny_gemma2,
            tokenizer,
            calibration_texts,
            max_seq_len=64,
        )
        masks = compute_wanda_masks(saliency, sparsity=target)
        for name, mask in masks.items():
            if mask.ndim == 2 and mask.shape[1] > 1:
                # Each row should have the same number of zeros
                zeros_per_row = (~mask).sum(dim=1).float()
                expected_zeros = int(mask.shape[1] * target)
                assert (zeros_per_row == expected_zeros).all(), f"{name}: not all rows have {expected_zeros} zeros"

    def test_zero_sparsity_keeps_all(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.wanda import compute_wanda_saliency, compute_wanda_masks

        saliency = compute_wanda_saliency(
            tiny_gemma2,
            tokenizer,
            calibration_texts,
            max_seq_len=64,
        )
        masks = compute_wanda_masks(saliency, sparsity=0.0)
        for name, mask in masks.items():
            assert mask.all(), f"{name} has False values at sparsity=0"


class TestPruneWanda:
    def test_end_to_end(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.wanda import prune_wanda

        # Clone model to avoid mutating the fixture
        import copy

        model = copy.deepcopy(tiny_gemma2)

        zeros_before = sum((p.data == 0).sum().item() for p in model.parameters())
        n_zeroed = prune_wanda(model, tokenizer, calibration_texts, sparsity=0.5, max_seq_len=64)
        zeros_after = sum((p.data == 0).sum().item() for p in model.parameters())

        assert n_zeroed > 0
        assert zeros_after > zeros_before

    def test_return_masks(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.wanda import prune_wanda
        import copy

        model = copy.deepcopy(tiny_gemma2)

        result = prune_wanda(
            model,
            tokenizer,
            calibration_texts,
            sparsity=0.3,
            max_seq_len=64,
            return_masks=True,
        )
        assert isinstance(result, tuple)
        n_zeroed, masks = result
        assert n_zeroed > 0
        assert isinstance(masks, dict)
        assert len(masks) > 0
        for name, m in masks.items():
            assert m.dtype == torch.bool
            assert m.device == torch.device("cpu")

    def test_model_still_runs(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.wanda import prune_wanda
        import copy

        model = copy.deepcopy(tiny_gemma2)

        prune_wanda(model, tokenizer, calibration_texts, sparsity=0.5, max_seq_len=64)

        # Forward pass should work
        inputs = tokenizer("Hello", return_tensors="pt", max_length=16, truncation=True)
        with torch.no_grad():
            out = model(**inputs)
        assert out.logits.shape[0] == 1
        assert not torch.isnan(out.logits).any()
