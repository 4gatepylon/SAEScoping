"""
CPU unit tests for SparseLLM pruning using a tiny Gemma2 model.

No GPU needed. Runs in <30s.

Usage:
  python -m pytest sae_scoping/training/saliency/tests/test_sparse_llm_cpu.py -v
"""
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@pytest.fixture(scope="module")
def tiny_gemma2():
    """Create a tiny Gemma2 model on CPU for testing."""
    config = AutoConfig.from_pretrained("google/gemma-2-2b-it")
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
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    return tok


@pytest.fixture(scope="module")
def calibration_texts(tokenizer):
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


class TestComputeSparseLLMMasks:
    def test_returns_dict_of_masks(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.sparse_llm import compute_sparse_llm_masks

        masks = compute_sparse_llm_masks(
            tiny_gemma2, tokenizer, calibration_texts,
            sparsity=0.3, n_iterations=1, max_seq_len=64,
        )
        assert isinstance(masks, dict)
        assert len(masks) > 0

    def test_masks_are_binary(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.sparse_llm import compute_sparse_llm_masks

        masks = compute_sparse_llm_masks(
            tiny_gemma2, tokenizer, calibration_texts,
            sparsity=0.3, n_iterations=1, max_seq_len=64,
        )
        for name, mask in masks.items():
            unique = mask.unique()
            assert all(v in (0.0, 1.0) for v in unique), (
                f"{name} has non-binary values: {unique}"
            )

    def test_masks_have_correct_keys(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.sparse_llm import compute_sparse_llm_masks

        masks = compute_sparse_llm_masks(
            tiny_gemma2, tokenizer, calibration_texts,
            sparsity=0.3, n_iterations=1, max_seq_len=64,
        )
        param_names = {n for n, _ in tiny_gemma2.named_parameters()}
        for key in masks:
            assert key.endswith(".weight")
            assert key in param_names, f"{key} not in model params"

    def test_masks_have_correct_shapes(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.sparse_llm import compute_sparse_llm_masks

        masks = compute_sparse_llm_masks(
            tiny_gemma2, tokenizer, calibration_texts,
            sparsity=0.3, n_iterations=1, max_seq_len=64,
        )
        params = dict(tiny_gemma2.named_parameters())
        for name, mask in masks.items():
            assert mask.shape == params[name].shape, (
                f"{name}: mask {mask.shape} != param {params[name].shape}"
            )

    def test_nonzero_sparsity(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.sparse_llm import compute_sparse_llm_masks

        masks = compute_sparse_llm_masks(
            tiny_gemma2, tokenizer, calibration_texts,
            sparsity=0.5, n_iterations=1, max_seq_len=64,
        )
        total_zeros = sum((m == 0).sum().item() for m in masks.values())
        total_elements = sum(m.numel() for m in masks.values())
        assert total_zeros > 0, "No zeros produced at sparsity=0.5"
        actual_sparsity = total_zeros / total_elements
        # Actual sparsity should be in the right ballpark
        assert 0.1 < actual_sparsity < 0.9, f"Actual sparsity {actual_sparsity} seems wrong"


class TestPruneSparseLLM:
    def test_end_to_end(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.sparse_llm import prune_sparse_llm
        import copy
        model = copy.deepcopy(tiny_gemma2)

        zeros_before = sum((p.data == 0).sum().item() for p in model.parameters())
        n_zeroed = prune_sparse_llm(
            model, tokenizer, calibration_texts,
            sparsity=0.3, n_iterations=1, max_seq_len=64,
        )
        zeros_after = sum((p.data == 0).sum().item() for p in model.parameters())

        assert n_zeroed > 0
        assert zeros_after > zeros_before

    def test_return_masks(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.sparse_llm import prune_sparse_llm
        import copy
        model = copy.deepcopy(tiny_gemma2)

        result = prune_sparse_llm(
            model, tokenizer, calibration_texts,
            sparsity=0.3, n_iterations=1, max_seq_len=64,
            return_masks=True,
        )
        n_zeroed, masks = result
        assert n_zeroed > 0
        assert isinstance(masks, dict)
        for name, m in masks.items():
            assert m.device == torch.device("cpu")

    def test_model_still_runs(self, tiny_gemma2, tokenizer, calibration_texts):
        from sae_scoping.training.saliency.sparse_llm import prune_sparse_llm
        import copy
        model = copy.deepcopy(tiny_gemma2)

        prune_sparse_llm(
            model, tokenizer, calibration_texts,
            sparsity=0.3, n_iterations=1, max_seq_len=64,
        )

        inputs = tokenizer("Hello", return_tensors="pt", max_length=16, truncation=True)
        with torch.no_grad():
            out = model(**inputs)
        assert out.logits.shape[0] == 1
        assert not torch.isnan(out.logits).any()
