"""
Integration test for prune_and_maybe_recover_sweep.py

Uses a 1-layer Qwen2.5-0.5B-Instruct on CPU (or GPU if available).

The model is stripped to 1 layer and saved to a temp directory so that
run_sweep_step can reload it from disk on every binary search step —
matching exactly the production use-case where weights must be re-loaded
because pruning destroys them in-place.

Tests
-----
1. Prune-only sweep (threshold=-1)
   All steps succeed (no quality bar). Binary search advances lo toward k_max.
   best_sparsity is not None. Per-step JSON and sweep_result.json are written.

2. Sweep with easy loss threshold (threshold=999.0)
   Recovery is triggered (threshold not met post-prune) but the very first
   eval step (step 5) should satisfy loss<=999. best_sparsity not None.

3. Sweep with impossible loss threshold (threshold=0.001)
   No step can possibly satisfy loss<=0.001 even after recovery.
   best_sparsity is None. All SweepStepResult.is_success are False.

4. Give-up rule fires on impossible threshold
   A give-up rule says "give up if loss > 10 after 5 recovery steps".
   Recovery gives up quickly. gave_up=True for all steps.
"""

import tempfile
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from datasets import Dataset
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from prune_and_maybe_recover_sweep import (
    GiveUpThreshold,
    SweepResult,
    prune_and_maybe_recover_sweep,
)


_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Shared settings for all tests — kept tiny so tests run fast on CPU.
_K_MIN = 0.1
_K_MAX = 0.5
_MAX_STEPS_SWEEP = 3
_MAX_STEPS_RECOVERY = 10
_EVAL_EVERY = 5
_BATCH_SIZE = 2
_MAX_SEQ_LEN = 64
_N_DATASET = 8


def _make_tiny_model_dir(save_dir: str) -> str:
    """Load Qwen2.5-0.5B, strip to 1 layer, update config, save to save_dir."""
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID, torch_dtype=torch.bfloat16,
    )
    model.model.layers = nn.ModuleList([model.model.layers[-1]])
    model.config.num_hidden_layers = 1
    model.save_pretrained(save_dir)

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(save_dir)

    return save_dir


def _make_saliency(model_dir: str, saliency_path: str) -> str:
    """Create a random saliency map matching the saved model's parameters."""
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    torch.manual_seed(42)
    saliency = {
        name: torch.rand(param.shape)
        for name, param in model.named_parameters()
    }
    save_file(saliency, saliency_path)
    del model
    return saliency_path


def _make_fake_dataset(n: int = _N_DATASET) -> Dataset:
    """Create a tiny synthetic QA dataset."""
    rows = [
        {"question": f"What is {i} + {i}?", "answer": f"The answer is {i + i}."}
        for i in range(n)
    ]
    return Dataset.from_list(rows)


def _run_sweep(
    model_dir: str,
    saliency_path: str,
    output_dir: str,
    threshold: float,
    give_up_thresholds: list[GiveUpThreshold] | None = None,
) -> SweepResult:
    """Helper: run prune_and_maybe_recover_sweep with shared settings."""
    dataset = _make_fake_dataset()
    return prune_and_maybe_recover_sweep(
        model_name_or_path=model_dir,
        saliency_path=saliency_path,
        dataset_evaluation=dataset,
        k_min=_K_MIN,
        k_max=_K_MAX,
        metric_type="loss",
        threshold=threshold,
        dataset_recovery=dataset if threshold >= 0.0 else None,
        max_steps_recovery=_MAX_STEPS_RECOVERY,
        eval_every=_EVAL_EVERY,
        batch_size=_BATCH_SIZE,
        max_seq_len=_MAX_SEQ_LEN,
        max_steps_sweep=_MAX_STEPS_SWEEP,
        give_up_thresholds=give_up_thresholds or [],
        output_dir=output_dir,
        device=DEVICE,
        wandb_project=None,
    )


def _assert_json_files_written(output_dir: str, n_steps: int) -> None:
    """Check that per-step JSON and sweep_result.json exist."""
    root = Path(output_dir)
    assert (root / "sweep_result.json").exists(), "sweep_result.json not written"
    for i in range(n_steps):
        step_json = root / f"step_{i:03d}_result.json"
        assert step_json.exists(), f"{step_json.name} not written"


def main() -> None:
    print("=" * 80)
    print("Integration test: prune_and_maybe_recover_sweep")
    print("=" * 80)

    if DEVICE == "cpu":
        warnings.warn("⚠️  CUDA not available; running on CPU (slow but valid)")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("\nSetting up: loading Qwen2.5-0.5B and saving 1-layer version...")
        model_dir = _make_tiny_model_dir(f"{tmpdir}/model")
        saliency_path = _make_saliency(model_dir, f"{tmpdir}/saliency.safetensors")
        print("Model + saliency ready.")

        # ---------------------------------------------------------------
        # Test 1: Prune-only (threshold=-1, no recovery)
        # ---------------------------------------------------------------
        print("\n--- Test 1: Prune-only sweep (no quality threshold) ---")
        result1 = _run_sweep(
            model_dir=model_dir,
            saliency_path=saliency_path,
            output_dir=f"{tmpdir}/test1",
            threshold=-1.0,
        )
        assert result1.best_sparsity is not None, (
            f"❌ Test 1: best_sparsity should be found when threshold=-1, got None"
        )
        assert len(result1.steps) == _MAX_STEPS_SWEEP, (
            f"❌ Test 1: expected {_MAX_STEPS_SWEEP} steps, got {len(result1.steps)}"
        )
        assert all(s.is_success for s in result1.steps), (
            "❌ Test 1: all steps should succeed when threshold=-1"
        )
        assert all(s.recovery_steps == 0 for s in result1.steps), (
            "❌ Test 1: no recovery steps should run when threshold=-1"
        )
        _assert_json_files_written(f"{tmpdir}/test1", _MAX_STEPS_SWEEP)
        # Binary search should push lo toward k_max when all succeed
        assert result1.best_sparsity > (_K_MIN + _K_MAX) / 2.0, (
            f"❌ Test 1: best_sparsity={result1.best_sparsity:.4f} should exceed midpoint "
            f"when all steps succeed"
        )
        print(
            f"✅ Test 1: prune-only, {len(result1.steps)} steps, "
            f"best_sparsity={result1.best_sparsity:.4f}"
        )

        # ---------------------------------------------------------------
        # Test 2: Easy loss threshold (should succeed with brief recovery)
        # ---------------------------------------------------------------
        print("\n--- Test 2: Sweep with easy loss threshold (999.0) ---")
        result2 = _run_sweep(
            model_dir=model_dir,
            saliency_path=saliency_path,
            output_dir=f"{tmpdir}/test2",
            threshold=999.0,
        )
        assert result2.best_sparsity is not None, (
            "❌ Test 2: best_sparsity should be found with threshold=999"
        )
        assert all(s.is_success for s in result2.steps), (
            "❌ Test 2: all steps should succeed with threshold=999"
        )
        assert all(s.n_weights_zeroed > 0 for s in result2.steps), (
            "❌ Test 2: every step must prune some weights"
        )
        _assert_json_files_written(f"{tmpdir}/test2", _MAX_STEPS_SWEEP)
        print(
            f"✅ Test 2: easy threshold, {len(result2.steps)} steps, "
            f"best_sparsity={result2.best_sparsity:.4f}"
        )

        # ---------------------------------------------------------------
        # Test 3: Impossible threshold (all steps fail)
        # ---------------------------------------------------------------
        print("\n--- Test 3: Sweep with impossible loss threshold (0.001) ---")
        result3 = _run_sweep(
            model_dir=model_dir,
            saliency_path=saliency_path,
            output_dir=f"{tmpdir}/test3",
            threshold=0.001,
        )
        assert result3.best_sparsity is None, (
            f"❌ Test 3: best_sparsity should be None when all steps fail, "
            f"got {result3.best_sparsity}"
        )
        assert all(not s.is_success for s in result3.steps), (
            "❌ Test 3: all steps should fail with threshold=0.001"
        )
        _assert_json_files_written(f"{tmpdir}/test3", _MAX_STEPS_SWEEP)
        print(
            f"✅ Test 3: impossible threshold, {len(result3.steps)} steps all failed, "
            f"best_sparsity=None"
        )

        # ---------------------------------------------------------------
        # Test 4: Give-up rules fire
        # ---------------------------------------------------------------
        print("\n--- Test 4: Give-up rules fire on impossible threshold ---")
        # threshold=0.001: "give up if loss > 0.001 after N steps", which always fires
        # since a pruned 1-layer LLM's loss is always much larger than 0.001.
        give_up = [GiveUpThreshold(steps=_EVAL_EVERY, threshold=0.001)]
        result4 = _run_sweep(
            model_dir=model_dir,
            saliency_path=saliency_path,
            output_dir=f"{tmpdir}/test4",
            threshold=0.001,
            give_up_thresholds=give_up,
        )
        # Loss of a 1-layer LLM will never be <=0.001, so give-up fires at step 5
        assert all(s.gave_up for s in result4.steps), (
            "❌ Test 4: all steps should have gave_up=True with a firing give-up rule "
            f"(got {[s.gave_up for s in result4.steps]})"
        )
        assert all(s.recovery_steps <= _EVAL_EVERY for s in result4.steps), (
            "❌ Test 4: recovery should stop by the give-up step at most"
        )
        print(
            f"✅ Test 4: give-up fired on all {len(result4.steps)} steps, "
            f"recovery_steps={[s.recovery_steps for s in result4.steps]}"
        )

    print("\n" + "=" * 80)
    print("✅ All prune_and_maybe_recover_sweep integration tests passed")
    print("=" * 80)


if __name__ == "__main__":
    main()
