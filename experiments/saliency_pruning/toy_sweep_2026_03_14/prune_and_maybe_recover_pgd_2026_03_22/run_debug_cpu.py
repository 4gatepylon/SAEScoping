"""CPU smoke test for PGD recovery with a tiny model.

Creates a 1-layer Llama model, generates a matching fake saliency file,
and runs prune_and_maybe_recover end-to-end on CPU.  Takes ~30 seconds.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from datasets import Dataset
from safetensors.torch import save_file
from transformers import AutoTokenizer

from prune_and_maybe_recover import prune_and_maybe_recover

# Re-use the tiny model factory from unit tests
from tests.unit.model_factories import make_tiny_llama


def main() -> None:
    print("Creating tiny Llama model (1 layer, hidden_size=64)...")
    model = make_tiny_llama()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Use a real tokenizer (tiny model's vocab_size=256, but tokenizer
    # just needs to produce input_ids; we truncate to max_length anyway).
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build a fake saliency file that matches the model's parameter names.
    saliency: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        saliency[name] = torch.rand_like(param, dtype=torch.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        saliency_path = Path(tmpdir) / "fake_saliency.safetensors"
        save_file(saliency, str(saliency_path))
        print(f"Saved fake saliency to {saliency_path}")

        # Tiny dataset (question/answer pairs)
        qa_data = [
            {"question": "What is DNA?", "answer": "Deoxyribonucleic acid."},
            {"question": "What is RNA?", "answer": "Ribonucleic acid."},
            {"question": "What is ATP?", "answer": "Adenosine triphosphate."},
            {"question": "What is a cell?", "answer": "The basic unit of life."},
        ]
        dataset = Dataset.from_list(qa_data)

        output_dir = Path(tmpdir) / "recovery_output"
        output_json = Path(tmpdir) / "result.json"

        print("Running prune_and_maybe_recover (PGD, CPU, 3 steps)...")
        result = prune_and_maybe_recover(
            model=model,
            tokenizer=tokenizer,
            saliency_path=str(saliency_path),
            sparsity=0.10,
            dataset_evaluation=dataset,
            saliency_type="gradient",
            metric_type="loss",
            threshold=999.0,  # absolute threshold, very high so recovery runs
            threshold_mode="absolute",
            dataset_recovery=dataset,
            max_steps=3,
            eval_every=2,
            batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            max_seq_len=32,
            max_new_tokens=8,
            output_dir=str(output_dir),
            use_pgd=True,
            n_iterations=1,
        )

        print(f"\n✅ PGD recovery completed successfully on CPU!")
        print(f"   Sparsity: {result.sparsity}")
        print(f"   Weights zeroed: {result.n_weights_zeroed}")
        print(f"   Recovery steps: {result.recovery_steps}")
        print(f"   Metric before: {result.metric_before_recovery:.4f}")
        print(f"   Metric after:  {result.metric_after_recovery:.4f}")


if __name__ == "__main__":
    main()
