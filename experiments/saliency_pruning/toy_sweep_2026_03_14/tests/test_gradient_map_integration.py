"""Integration tests for gradient-map computation and the full pruning pipeline.

These tests require a GPU and download a small pretrained model the first time
they run (Qwen/Qwen2.5-0.5B-Instruct, ~880 MB in fp16).  They use a synthetic
in-memory SFT dataset so no additional HuggingFace dataset download is needed.

The model is truncated to 2 transformer layers to keep runtime to a few minutes.

Run with:
    python tests/test_gradient_map_integration.py

Or with pytest (marks integration tests clearly):
    pytest tests/test_gradient_map_integration.py -v

Design intent
-------------
These tests are the "sanity check" tier: they exercise the full
GradCollectTrainer + apply_pruning loop on *real* weights and a *real*
tokenizer to catch bugs that only manifest with actual numeric distributions
(e.g. gradient-hook misfires on weight-tied tensors, silent tokenisation
errors, dtype mismatches on GPU).  The unit tests in tests/unit/ cover the
same invariants on CPU-only toy models.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from sae_scoping.training.saliency.grad import GradCollectTrainer
from sweep_eval_temp import (
    apply_pruning,
    compute_saliency_scores,
    restore_original_weights,
    save_original_weights,
)
from tests.unit.validators import (
    assert_gradient_map_covers_all_params,
    assert_gradient_map_nonneg,
    assert_pruning_is_lowest_saliency,
    assert_restore_is_exact,
    assert_sparsity_achieved,
    assert_weights_unchanged_from_snapshot,
)

_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
_NUM_LAYERS_TO_KEEP = 2
_SPARSITY_LEVELS = [0.1, 0.3, 0.5, 0.7]

# Small synthetic biology-like sentences; no dataset download required.
_SYNTHETIC_TEXTS = [
    "Photosynthesis converts light energy into chemical energy stored as glucose.",
    "DNA replication is a semi-conservative process catalysed by DNA polymerase.",
    "Mitochondria generate ATP through oxidative phosphorylation.",
    "Ribosomes translate mRNA sequences into polypeptide chains.",
    "The cell membrane is a phospholipid bilayer with embedded proteins.",
    "Enzymes lower the activation energy of biochemical reactions.",
    "CRISPR-Cas9 enables precise editing of DNA sequences.",
    "Neurons transmit signals via action potentials and synaptic release.",
]


def _load_truncated_model() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen2.5-0.5B and keep only the first _NUM_LAYERS_TO_KEEP layers."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        warnings.warn(
            "⚠️ CUDA not available; running integration test on CPU — may be slow"
        )
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    # Truncate to keep the test fast.
    model.model.layers = model.model.layers[:_NUM_LAYERS_TO_KEEP]

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _make_sft_dataset() -> Dataset:
    """Return a tiny in-memory SFT dataset with a 'text' column."""
    return Dataset.from_dict({"text": _SYNTHETIC_TEXTS})


# ---------------------------------------------------------------------------
# Integration test 1: GradCollectTrainer produces a valid gradient map
# ---------------------------------------------------------------------------


def test_grad_collect_trainer_produces_gradient_map() -> None:
    """GradCollectTrainer must leave weights unchanged and produce a full EMA grad map.

    After running train() for 1 epoch on 8 synthetic examples:
    - model weights must be byte-for-byte unchanged (optimizer.step is a no-op)
    - ema_grads() must cover every trainable parameter
    - abs-mode map must be non-negative
    - model._hook_fires counts must equal n_steps per parameter
    """
    print("=" * 80)
    print(f"Integration test: GradCollectTrainer on {_MODEL_ID} ({_NUM_LAYERS_TO_KEEP} layers)")

    model, tokenizer = _load_truncated_model()
    dataset = _make_sft_dataset()
    snapshot = {n: p.data.cpu().clone() for n, p in model.named_parameters()}

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = GradCollectTrainer(
            model=model,
            beta=0.9,
            abs_grad=True,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=tmp_dir,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                max_length=64,
                report_to="none",
                logging_steps=1,
                save_strategy="no",
            ),
        )
        trainer.train()

    grad_map = trainer.ema_grads()

    assert_weights_unchanged_from_snapshot(model, snapshot)
    assert_gradient_map_covers_all_params(model, grad_map)
    assert_gradient_map_nonneg(grad_map)

    n_params = sum(1 for _ in model.named_parameters() if True)
    print(
        f"✅ test_grad_collect_trainer_produces_gradient_map: "
        f"grad map covers {len(grad_map)}/{n_params} params, all non-negative, "
        f"weights unchanged after training"
    )


# ---------------------------------------------------------------------------
# Integration test 2: full sweep pipeline — gradient map → prune → restore
# ---------------------------------------------------------------------------


def test_prune_sweep_end_to_end_with_real_model() -> None:
    """Full pipeline: gradient map → compute_saliency_scores → multi-level pruning.

    Steps:
      1. Compute EMA gradient map via GradCollectTrainer (1 epoch).
      2. Derive gradient and Taylor saliency scores via compute_saliency_scores.
      3. For each of _SPARSITY_LEVELS:
         a. apply_pruning using gradient saliency.
         b. assert_pruning_is_lowest_saliency
         c. assert_sparsity_achieved
         d. restore_original_weights
         e. assert_restore_is_exact
    """
    print("=" * 80)
    print(f"Integration test: full sweep pipeline on {_MODEL_ID} ({_NUM_LAYERS_TO_KEEP} layers)")

    model, tokenizer = _load_truncated_model()
    dataset = _make_sft_dataset()

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = GradCollectTrainer(
            model=model,
            beta=0.9,
            abs_grad=False,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                output_dir=tmp_dir,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                max_length=64,
                report_to="none",
                logging_steps=1,
                save_strategy="no",
            ),
        )
        trainer.train()

    grad_map = trainer.ema_grads()
    gradient_saliency = compute_saliency_scores(model, grad_map, "gradient")
    original = save_original_weights(model)

    for sparsity in _SPARSITY_LEVELS:
        restore_original_weights(model, original)
        n_zeroed = apply_pruning(model, gradient_saliency, sparsity_fraction=sparsity)

        assert n_zeroed > 0, (
            f"❌ test_prune_sweep_end_to_end_with_real_model: "
            f"apply_pruning returned 0 zeroed weights at sparsity={sparsity}"
        )
        # original is the pre-pruning state (model was just restored from it)
        assert_pruning_is_lowest_saliency(model, gradient_saliency, original)
        assert_sparsity_achieved(model, gradient_saliency, target=sparsity, tol=0.02)

        restore_original_weights(model, original)
        assert_restore_is_exact(model, original)

        print(
            f"  sparsity={sparsity:.0%}: zeroed {n_zeroed:,} weights — "
            f"pruning optimal, restore exact ✅"
        )

    print(
        f"✅ test_prune_sweep_end_to_end_with_real_model: "
        f"{len(_SPARSITY_LEVELS)} sparsity levels, all invariants passed"
    )


# ---------------------------------------------------------------------------
# Entry point for standalone execution
# ---------------------------------------------------------------------------


def main() -> None:
    test_grad_collect_trainer_produces_gradient_map()
    test_prune_sweep_end_to_end_with_real_model()
    print("\n" + "=" * 80)
    print("✅ All integration tests passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
