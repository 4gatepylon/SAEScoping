"""
CPU unit tests for all three unlearning methods using a tiny Gemma2 model.

No GPU needed. Runs in <60s.

Usage:
  python -m pytest sae_scoping/training/unlearning/tests/test_unlearning_cpu.py -v
"""
import copy

import pytest
import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@pytest.fixture(scope="module")
def tiny_gemma2():
    config = AutoConfig.from_pretrained("google/gemma-2-2b-it")
    config.num_hidden_layers = 2
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    # Keep real vocab_size so tokenizer IDs are valid
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _make_dataset(tokenizer, domain: str, n: int = 8, max_length: int = 64) -> Dataset:
    """Create a tiny formatted and tokenized dataset."""
    rows = []
    for i in range(n):
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": f"What is {domain} topic {i}?"},
                {"role": "assistant", "content": f"This is about {domain} topic {i}."},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        tok = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
        rows.append({
            "text": text,
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels": tok["input_ids"],
        })
    return Dataset.from_list(rows)


@pytest.fixture(scope="module")
def forget_dataset(tokenizer):
    return _make_dataset(tokenizer, "math", n=8)


@pytest.fixture(scope="module")
def retain_dataset(tokenizer):
    return _make_dataset(tokenizer, "biology", n=8)


def _compute_loss(model, tokenizer, texts, max_len=64):
    """Compute mean loss on a list of texts."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for text in texts:
            tok = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
            labels = tok["input_ids"].clone()
            out = model(**tok, labels=labels)
            total += out.loss.item()
            n += 1
    return total / max(n, 1)


def _params_changed(model_before_state, model_after) -> bool:
    """Check if any parameters changed."""
    for name, param in model_after.named_parameters():
        before = model_before_state[name]
        if not torch.allclose(param.data.cpu(), before, atol=1e-6):
            return True
    return False


# =========================================================================
# Gradient Difference tests
# =========================================================================


class TestGradientDiff:
    def test_modifies_model(self, tiny_gemma2, tokenizer, forget_dataset, retain_dataset):
        from sae_scoping.training.unlearning.gradient_diff import unlearn_gradient_diff

        model = copy.deepcopy(tiny_gemma2)
        before_state = {n: p.data.cpu().clone() for n, p in model.named_parameters()}

        unlearn_gradient_diff(
            model, tokenizer, forget_dataset, retain_dataset,
            max_steps=5, max_length=64, batch_size=2,
        )

        assert _params_changed(before_state, model), "Model parameters should change after unlearning"

    def test_forget_loss_increases(self, tiny_gemma2, tokenizer, forget_dataset, retain_dataset):
        from sae_scoping.training.unlearning.gradient_diff import unlearn_gradient_diff

        model = copy.deepcopy(tiny_gemma2)
        loss_before = _compute_loss(model, tokenizer, forget_dataset["text"][:4])

        unlearn_gradient_diff(
            model, tokenizer, forget_dataset, retain_dataset,
            max_steps=20, max_length=64, batch_size=2, learning_rate=1e-3,
        )

        loss_after = _compute_loss(model, tokenizer, forget_dataset["text"][:4])
        assert loss_after > loss_before, (
            f"Forget loss should increase: {loss_before:.4f} -> {loss_after:.4f}"
        )

    def test_model_still_runs(self, tiny_gemma2, tokenizer, forget_dataset, retain_dataset):
        from sae_scoping.training.unlearning.gradient_diff import unlearn_gradient_diff

        model = copy.deepcopy(tiny_gemma2)
        unlearn_gradient_diff(
            model, tokenizer, forget_dataset, retain_dataset,
            max_steps=5, max_length=64, batch_size=2,
        )

        tok = tokenizer("Hello", return_tensors="pt", max_length=16, truncation=True)
        with torch.no_grad():
            out = model(**tok)
        assert not torch.isnan(out.logits).any(), "Output has NaN after unlearning"


# =========================================================================
# NPO tests
# =========================================================================


class TestNPO:
    def test_modifies_model(self, tiny_gemma2, tokenizer, forget_dataset, retain_dataset):
        from sae_scoping.training.unlearning.npo import unlearn_npo

        model = copy.deepcopy(tiny_gemma2)
        before_state = {n: p.data.cpu().clone() for n, p in model.named_parameters()}

        unlearn_npo(
            model, tokenizer, forget_dataset, retain_dataset=retain_dataset,
            max_steps=5, max_length=64, batch_size=2,
        )

        assert _params_changed(before_state, model), "Model parameters should change"

    def test_forget_loss_increases(self, tiny_gemma2, tokenizer, forget_dataset, retain_dataset):
        from sae_scoping.training.unlearning.npo import unlearn_npo

        model = copy.deepcopy(tiny_gemma2)
        loss_before = _compute_loss(model, tokenizer, forget_dataset["text"][:4])

        unlearn_npo(
            model, tokenizer, forget_dataset, retain_dataset=retain_dataset,
            max_steps=20, max_length=64, batch_size=2, learning_rate=1e-3,
        )

        loss_after = _compute_loss(model, tokenizer, forget_dataset["text"][:4])
        assert loss_after > loss_before, (
            f"Forget loss should increase: {loss_before:.4f} -> {loss_after:.4f}"
        )

    def test_works_without_retain(self, tiny_gemma2, tokenizer, forget_dataset):
        from sae_scoping.training.unlearning.npo import unlearn_npo

        model = copy.deepcopy(tiny_gemma2)
        # NPO without retain dataset should still work (pure NPO, no KL)
        unlearn_npo(
            model, tokenizer, forget_dataset, retain_dataset=None,
            max_steps=5, max_length=64, batch_size=2,
        )
        tok = tokenizer("Hello", return_tensors="pt", max_length=16, truncation=True)
        with torch.no_grad():
            out = model(**tok)
        assert not torch.isnan(out.logits).any()

    def test_model_still_runs(self, tiny_gemma2, tokenizer, forget_dataset, retain_dataset):
        from sae_scoping.training.unlearning.npo import unlearn_npo

        model = copy.deepcopy(tiny_gemma2)
        unlearn_npo(
            model, tokenizer, forget_dataset, retain_dataset=retain_dataset,
            max_steps=5, max_length=64, batch_size=2,
        )
        tok = tokenizer("Hello", return_tensors="pt", max_length=16, truncation=True)
        with torch.no_grad():
            out = model(**tok)
        assert not torch.isnan(out.logits).any()


# =========================================================================
# RMU tests
# =========================================================================


class TestRMU:
    def test_only_update_layers_change(self, tiny_gemma2, tokenizer, forget_dataset, retain_dataset):
        from sae_scoping.training.unlearning.rmu import unlearn_rmu

        model = copy.deepcopy(tiny_gemma2)
        before_state = {n: p.data.cpu().clone() for n, p in model.named_parameters()}

        # Hook layer 1, update only layer 0 — with param_ids=None to update all params in layer 0
        unlearn_rmu(
            model, tokenizer, forget_dataset, retain_dataset,
            hook_layer_id=1, update_layer_ids=[0], param_ids=None,
            max_steps=5, max_length=64, steering_coeff=5.0, alpha=1.0,
        )

        changed_names = []
        for name, param in model.named_parameters():
            if not torch.allclose(param.data.cpu(), before_state[name], atol=1e-6):
                changed_names.append(name)

        assert len(changed_names) > 0, "Some parameters should change"
        for name in changed_names:
            assert "layers.0." in name, (
                f"Parameter {name} changed but is not in update layer 0"
            )

    def test_forget_loss_increases(self, tiny_gemma2, tokenizer, forget_dataset, retain_dataset):
        from sae_scoping.training.unlearning.rmu import unlearn_rmu

        model = copy.deepcopy(tiny_gemma2)
        loss_before = _compute_loss(model, tokenizer, forget_dataset["text"][:4])

        unlearn_rmu(
            model, tokenizer, forget_dataset, retain_dataset,
            hook_layer_id=1, update_layer_ids=[0, 1], param_ids=None,
            max_steps=100, max_length=64,
            steering_coeff=100.0, alpha=1.0, learning_rate=1e-2,
        )

        loss_after = _compute_loss(model, tokenizer, forget_dataset["text"][:4])
        assert loss_after > loss_before, (
            f"Forget loss should increase: {loss_before:.4f} -> {loss_after:.4f}"
        )

    def test_model_still_runs(self, tiny_gemma2, tokenizer, forget_dataset, retain_dataset):
        from sae_scoping.training.unlearning.rmu import unlearn_rmu

        model = copy.deepcopy(tiny_gemma2)
        unlearn_rmu(
            model, tokenizer, forget_dataset, retain_dataset,
            hook_layer_id=1, update_layer_ids=[1], param_ids=None,
            max_steps=5, max_length=64, steering_coeff=5.0, alpha=1.0,
        )
        tok = tokenizer("Hello", return_tensors="pt", max_length=16, truncation=True)
        with torch.no_grad():
            out = model(**tok)
        assert not torch.isnan(out.logits).any()

    def test_all_params_unfrozen_after(self, tiny_gemma2, tokenizer, forget_dataset, retain_dataset):
        from sae_scoping.training.unlearning.rmu import unlearn_rmu

        model = copy.deepcopy(tiny_gemma2)
        unlearn_rmu(
            model, tokenizer, forget_dataset, retain_dataset,
            hook_layer_id=0, update_layer_ids=[0], param_ids=None,
            max_steps=3, max_length=64, steering_coeff=5.0, alpha=1.0,
        )
        for name, param in model.named_parameters():
            assert param.requires_grad, f"{name} is still frozen after RMU"
