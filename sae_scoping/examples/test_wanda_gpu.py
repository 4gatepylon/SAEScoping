"""
GPU integration tests for Wanda pruning on a real model.

Requires CUDA and downloads ~5GB (gemma-2-2b-it). Run with:

    CUDA_VISIBLE_DEVICES=0 python -m pytest sae_scoping/examples/test_wanda_gpu.py -v

Or as a standalone script:

    CUDA_VISIBLE_DEVICES=0 python sae_scoping/examples/test_wanda_gpu.py
"""
from __future__ import annotations

import sys

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_scoping.datasets.qa_datasets import load_qa_dataset, format_as_sft_text
from sae_scoping.evaluation.loss import compute_loss, count_zeros
from sae_scoping.training.saliency.wanda import (
    compute_wanda_saliency,
    compute_wanda_masks,
    apply_masks_to_model,
    prune_wanda,
)

MODEL_ID = "google/gemma-2-2b-it"
DATASET = "4gate/StemQAMixture"
SUBSET = "biology"
N_CALIBRATION = 8
N_EVAL = 16


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def model(device):
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager",
    )


@pytest.fixture(scope="module")
def calib_texts(tokenizer):
    ds = load_qa_dataset(DATASET, SUBSET, split="train", n=N_CALIBRATION, seed=42)
    return format_as_sft_text(ds, tokenizer)


@pytest.fixture(scope="module")
def eval_texts(tokenizer):
    ds = load_qa_dataset(DATASET, SUBSET, split="train", n=N_EVAL, seed=123)
    return format_as_sft_text(ds, tokenizer)


@pytest.fixture(scope="module")
def saliency(model, tokenizer, calib_texts):
    return compute_wanda_saliency(model, tokenizer, calib_texts, max_seq_len=512)


@pytest.fixture(scope="module")
def baseline_zeros(model):
    z, _ = count_zeros(model)
    return z


@requires_cuda
class TestSaliencyShapes:
    def test_nonempty(self, saliency):
        assert len(saliency) > 0

    def test_shapes_match_weights(self, saliency, model):
        params = dict(model.named_parameters())
        for name, scores in saliency.items():
            assert name in params, f"{name} not a model parameter"
            assert scores.shape == params[name].shape

    def test_scores_nonnegative(self, saliency):
        for name, scores in saliency.items():
            assert (scores >= 0).all(), f"Negative scores in {name}"


@requires_cuda
class TestMasks:
    def test_per_row_sparsity(self, saliency):
        masks = compute_wanda_masks(saliency, sparsity=0.3)
        for name, mask in masks.items():
            assert mask.dtype == torch.bool
            if mask.ndim == 2:
                row_sp = 1.0 - mask.float().mean(dim=1)
                assert (row_sp - 0.3).abs().mean().item() < 0.05

    def test_zero_sparsity_keeps_all(self, saliency):
        masks = compute_wanda_masks(saliency, sparsity=0.0)
        for mask in masks.values():
            assert mask.all()

    def test_high_sparsity_removes_most(self, saliency):
        masks = compute_wanda_masks(saliency, sparsity=0.9)
        for mask in masks.values():
            if mask.ndim == 2:
                kept = mask.float().mean().item()
                assert kept < 0.15


@requires_cuda
class TestPruneEndToEnd:
    def test_zeros_increase(self, model, tokenizer, calib_texts, baseline_zeros, device):
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager",
        )
        n_zeroed = prune_wanda(m, tokenizer, calib_texts, sparsity=0.5, max_seq_len=512)
        assert n_zeroed > 0
        post_zeros, _ = count_zeros(m)
        assert post_zeros > baseline_zeros
        del m
        torch.cuda.empty_cache()

    def test_loss_finite_after_pruning(self, model, tokenizer, calib_texts, eval_texts, device):
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager",
        )
        prune_wanda(m, tokenizer, calib_texts, sparsity=0.5, max_seq_len=512)
        loss = compute_loss(m, tokenizer, eval_texts, max_seq_len=512)
        assert 0 < loss < float("inf")
        del m
        torch.cuda.empty_cache()

    def test_generation_works(self, model, tokenizer, calib_texts, device):
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager",
        )
        prune_wanda(m, tokenizer, calib_texts, sparsity=0.5, max_seq_len=512)
        inputs = tokenizer("What is photosynthesis?", return_tensors="pt").to(device)
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=30, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        assert len(text) > 10
        del m
        torch.cuda.empty_cache()

    def test_return_masks_for_pgd(self, model, tokenizer, calib_texts, device):
        m = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager",
        )
        result = prune_wanda(
            m, tokenizer, calib_texts, sparsity=0.3,
            max_seq_len=512, return_masks=True,
        )
        assert isinstance(result, tuple)
        n_z, masks = result
        assert n_z > 0
        for mask in masks.values():
            assert mask.dtype == torch.bool
            assert mask.device == torch.device("cpu")
        del m, masks
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
