"""Tiny model factories for unit tests.

All factories create models from config (randomly initialised) — no HuggingFace
weight downloads required.  Models are CPU-only and have a single transformer
layer so they are fast to instantiate and run backward passes on.

Key exports
-----------
make_tiny_qwen2()          – Qwen2ForCausalLM (1 layer, hidden_size=64)
make_tiny_llama()          – LlamaForCausalLM (1 layer, hidden_size=64)
TINY_HF_FACTORIES          – ordered dict of name → factory, for parametrisation
make_known_saliency_fixture() – (model, saliency) with analytically known pruning set
make_taylor_vs_gradient_fixture() – (model, grad_saliency, taylor_saliency) where
                                    the two criteria prune different weights
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

# ---------------------------------------------------------------------------
# Shared config for all tiny HF models
# ---------------------------------------------------------------------------

_SHARED_CONFIG: dict = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=32,
    tie_word_embeddings=False,  # avoid tied-weight complications in hook tests
)


def make_tiny_qwen2() -> Qwen2ForCausalLM:
    """1-layer Qwen2 model with randomly initialised weights (CPU, no download)."""
    config = Qwen2Config(**_SHARED_CONFIG)
    return Qwen2ForCausalLM(config)


def make_tiny_llama() -> LlamaForCausalLM:
    """1-layer Llama model with randomly initialised weights (CPU, no download)."""
    config = LlamaConfig(**_SHARED_CONFIG)
    return LlamaForCausalLM(config)


# Ordered dict used to parametrise tests over architectures.
TINY_HF_FACTORIES: dict[str, callable] = {
    "qwen2": make_tiny_qwen2,
    "llama": make_tiny_llama,
}


# ---------------------------------------------------------------------------
# Synthetic backward-pass helper
# ---------------------------------------------------------------------------


def run_single_synthetic_step(
    model: nn.Module,
    seq_len: int = 8,
    batch_size: int = 2,
    vocab_size: int = 256,
) -> None:
    """Run one forward+backward step with random input_ids/labels.

    Clears model._ema_seen (if present) before the step so that each call
    represents a fresh training step in GradCollectTrainer's convention.
    """
    if hasattr(model, "_ema_seen"):
        model._ema_seen.clear()
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    outputs = model(input_ids=input_ids, labels=input_ids)
    outputs.loss.backward()


# ---------------------------------------------------------------------------
# Known-saliency fixture
# ---------------------------------------------------------------------------


class _TwoGroupModel(nn.Module):
    """Model with two parameter groups of analytically known saliency."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(42)
        # 16 params — saliency will be set to 100.0 (always kept)
        self.important = nn.Linear(4, 4, bias=False)
        # 8 params — saliency will be set to 0.001 (always pruned first)
        self.unimportant = nn.Linear(4, 2, bias=False)
        nn.init.normal_(self.important.weight)
        nn.init.normal_(self.unimportant.weight)


def make_known_saliency_fixture() -> tuple[_TwoGroupModel, dict[str, torch.Tensor]]:
    """Return (model, saliency_scores) where the pruning set is analytically known.

    ``important.weight``   → saliency = 100.0  (16 params, always kept)
    ``unimportant.weight`` → saliency = 0.001  (8 params, always pruned first)

    At sparsity = 8/24 ≈ 0.333, *exactly* ``unimportant.weight`` should be zeroed.
    The invariant ``assert_pruning_is_lowest_saliency`` must hold at every sparsity.
    """
    model = _TwoGroupModel()
    saliency: dict[str, torch.Tensor] = {
        "important.weight": torch.full_like(model.important.weight, 100.0),
        "unimportant.weight": torch.full_like(model.unimportant.weight, 0.001),
    }
    return model, saliency


# ---------------------------------------------------------------------------
# Taylor-vs-gradient divergence fixture
# ---------------------------------------------------------------------------


class _SingleLinearModel(nn.Module):
    """1 × 2 weight matrix used to construct a Taylor vs. gradient divergence case."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 1, bias=False)


def make_taylor_vs_gradient_fixture() -> tuple[
    _SingleLinearModel,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    """Return (model, grad_saliency, taylor_saliency) where the two criteria prune
    different individual weights in ``fc.weight``.

    Weight layout (1 × 2 matrix, flattened to [w0, w1]):
        w0 = 10.0,  w1 = 0.1

    Gradient saliency (EMA |grad| simulated manually):
        g0 = 0.5,   g1 = 2.0

    Gradient scoring: prune w0 (score 0.5 < 2.0)
    Taylor scoring:   |g * w| = [|0.5 * 10.0|, |2.0 * 0.1|] = [5.0, 0.2]
                      prune w1 (score 0.2 < 5.0)

    The two pruning decisions are thus *opposite*.
    """
    model = _SingleLinearModel()
    with torch.no_grad():
        model.fc.weight.data = torch.tensor([[10.0, 0.1]])

    grad_saliency: dict[str, torch.Tensor] = {
        "fc.weight": torch.tensor([[0.5, 2.0]]),
    }
    taylor_saliency: dict[str, torch.Tensor] = {
        "fc.weight": (grad_saliency["fc.weight"].abs() * model.fc.weight.data.abs()),
    }
    return model, grad_saliency, taylor_saliency
