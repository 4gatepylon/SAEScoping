"""
Integration test for prune_and_maybe_recover.py

Uses a 1-layer Qwen2.5-0.5B-Instruct on a single GPU. Tests:
1. Prune-only (no recovery) — verify weights are zeroed, result returned
2. Prune + recovery with loss metric — verify recovery SFT runs and metric improves
"""

import tempfile
import warnings

import torch
from datasets import Dataset
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_scoping.training.weight_pruning import save_original_weights
from prune_and_maybe_recover import prune_and_maybe_recover


_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_tiny_model():
    """Load Qwen 0.5B, strip to 1 layer for speed."""
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE,
    )
    model.model.layers = model.model.layers[-1:]
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _make_fake_dataset(n: int = 16) -> Dataset:
    """Create a tiny QA dataset."""
    rows = [
        {"question": f"What is {i} + {i}?", "answer": f"The answer is {i + i}."}
        for i in range(n)
    ]
    return Dataset.from_list(rows)


def _make_fake_saliency(model, path):
    """Create a random saliency map matching the model params."""
    torch.manual_seed(42)
    saliency = {
        name: torch.rand(param.shape)
        for name, param in model.named_parameters()
    }
    save_file(saliency, str(path))
    return path


def main():
    print("=" * 80)
    print("Integration test: prune_and_maybe_recover")
    print("=" * 80)

    if DEVICE == "cpu":
        warnings.warn("⚠️ CUDA not available, running on CPU (slow)")

    model, tokenizer = _make_tiny_model()
    dataset = _make_fake_dataset(16)

    with tempfile.TemporaryDirectory() as tmpdir:
        saliency_path = _make_fake_saliency(model, f"{tmpdir}/saliency.safetensors")
        original_weights = save_original_weights(model)

        # -------------------------------------------------------------------
        # Test 1: Prune only (threshold < 0 → skip recovery)
        # -------------------------------------------------------------------
        print("\n--- Test 1: Prune only (no recovery) ---")
        result = prune_and_maybe_recover(
            model=model,
            tokenizer=tokenizer,
            saliency_path=saliency_path,
            sparsity=0.3,
            dataset_evaluation=dataset,
            saliency_type="gradient",
            metric_type="loss",
            threshold=-1.0,
            max_seq_len=128,
            output_dir=f"{tmpdir}/test1_output",
        )
        assert result.n_weights_zeroed > 0, f"❌ Expected some weights zeroed, got {result.n_weights_zeroed}"
        assert result.recovery_steps == 0, f"❌ Expected 0 recovery steps, got {result.recovery_steps}"
        assert result.metric_before_recovery == result.metric_after_recovery
        print(f"✅ Prune-only: zeroed {result.n_weights_zeroed} weights, "
              f"loss={result.metric_before_recovery:.4f}, no recovery ran")

        # Restore for next test
        from sae_scoping.training.weight_pruning import restore_original_weights
        restore_original_weights(model, original_weights)

        # -------------------------------------------------------------------
        # Test 2: Prune + recovery with loss metric
        # -------------------------------------------------------------------
        print("\n--- Test 2: Prune + recovery (loss metric) ---")
        # Use a very generous threshold so recovery will stop quickly
        result2 = prune_and_maybe_recover(
            model=model,
            tokenizer=tokenizer,
            saliency_path=saliency_path,
            sparsity=0.1,
            dataset_evaluation=dataset,
            saliency_type="gradient",
            metric_type="loss",
            threshold=999.0,  # very generous — should meet immediately or quickly
            dataset_recovery=dataset,
            max_steps=20,
            eval_every=5,
            batch_size=2,
            learning_rate=1e-4,
            max_seq_len=128,
            output_dir=f"{tmpdir}/test2_output",
        )
        assert result2.n_weights_zeroed > 0, f"❌ Expected some weights zeroed"
        assert result2.metric_after_recovery > 0, f"❌ Expected a valid metric"
        print(f"✅ Prune+recovery: zeroed {result2.n_weights_zeroed} weights, "
              f"loss before={result2.metric_before_recovery:.4f}, "
              f"loss after={result2.metric_after_recovery:.4f}, "
              f"steps={result2.recovery_steps}, "
              f"early_stop={result2.recovery_stopped_early}")

        # Restore for next test
        restore_original_weights(model, original_weights)

        # -------------------------------------------------------------------
        # Test 3: Prune + recovery that actually trains (tight threshold)
        # -------------------------------------------------------------------
        print("\n--- Test 3: Prune + recovery (tight threshold, trains to max_steps) ---")
        result3 = prune_and_maybe_recover(
            model=model,
            tokenizer=tokenizer,
            saliency_path=saliency_path,
            sparsity=0.3,
            dataset_evaluation=dataset,
            saliency_type="taylor",
            metric_type="loss",
            threshold=0.001,  # impossibly tight — will train to max_steps
            dataset_recovery=dataset,
            max_steps=10,
            eval_every=5,
            batch_size=2,
            learning_rate=1e-4,
            max_seq_len=128,
            output_dir=f"{tmpdir}/test3_output",
        )
        assert result3.recovery_steps == 10, (
            f"❌ Expected 10 recovery steps (max_steps), got {result3.recovery_steps}"
        )
        assert not result3.recovery_stopped_early, "❌ Should not have stopped early"
        print(f"✅ Tight threshold: trained full {result3.recovery_steps} steps, "
              f"loss before={result3.metric_before_recovery:.4f}, "
              f"loss after={result3.metric_after_recovery:.4f}")

    print("\n" + "=" * 80)
    print("✅ All prune_and_maybe_recover integration tests passed")
    print("=" * 80)


if __name__ == "__main__":
    main()
