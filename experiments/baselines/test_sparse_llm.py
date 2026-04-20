"""
Integration tests for SparseLLM pruning on small Gemma models.

Usage:
  CUDA_VISIBLE_DEVICES=0 python test_sparse_llm.py
  CUDA_VISIBLE_DEVICES=0 python test_sparse_llm.py --model google/gemma-3-4b-it
"""
from __future__ import annotations

import sys
import traceback

import click
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_texts(tokenizer, dataset_name, subset, n, seed=42):
    ds = load_dataset(dataset_name, subset, split="train")
    ds = ds.shuffle(seed=seed)
    texts = []
    for i in range(min(n, len(ds))):
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": str(ds[i]["question"])},
                {"role": "assistant", "content": str(ds[i]["answer"])},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return texts


@torch.no_grad()
def _val_loss(model, tokenizer, texts, device, max_seq_len=512):
    model.eval()
    old_pad = tokenizer.padding_side
    tokenizer.padding_side = "right"
    total, n = 0.0, 0
    for t in texts:
        tok = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_seq_len)
        ids = tok["input_ids"].to(device)
        mask = tok["attention_mask"].to(device)
        labels = ids.clone()
        labels[mask == 0] = -100
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        total += out.loss.item()
        n += 1
    tokenizer.padding_side = old_pad
    return total / max(n, 1)


def _count_zeros(model):
    total, zeros = 0, 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p.data == 0).sum().item()
    return zeros, total


@click.command()
@click.option("--model", "model_id", default="google/gemma-2-2b-it")
@click.option("--dataset-name", default="4gate/StemQAMixture")
@click.option("--dataset-subset", default="biology")
@click.option("--n-calibration", default=4, help="Few samples due to SparseLLM memory")
@click.option("--n-eval", default=8)
@click.option("--device", default="cuda:0")
def main(model_id, dataset_name, dataset_subset, n_calibration, n_eval, device):
    device = torch.device(device)
    passed, failed = [], []

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device,
        attn_implementation="eager",
    )

    calib_texts = _load_texts(tokenizer, dataset_name, dataset_subset, n_calibration, seed=42)
    eval_texts = _load_texts(tokenizer, dataset_name, dataset_subset, n_eval, seed=123)

    baseline_loss = _val_loss(model, tokenizer, eval_texts, device)
    baseline_zeros, total_params = _count_zeros(model)
    print(f"Baseline: loss={baseline_loss:.4f}, zeros={baseline_zeros:,}/{total_params:,}")

    # ── Test 1: compute_sparse_llm_masks produces valid masks ─────────────
    test_name = "compute_sparse_llm_masks shapes and values"
    try:
        from sae_scoping.training.saliency.sparse_llm import compute_sparse_llm_masks
        masks = compute_sparse_llm_masks(
            model, tokenizer, calib_texts,
            sparsity=0.3, n_iterations=2,  # fewer iters for speed
            max_seq_len=512,
        )
        assert len(masks) > 0, "No masks produced"
        for name, mask in masks.items():
            assert name.endswith(".weight"), f"Expected .weight suffix, got {name}"
            # Masks should be 0/1 float
            unique_vals = mask.unique()
            assert all(v in (0.0, 1.0) for v in unique_vals), (
                f"Mask {name} has non-binary values: {unique_vals}"
            )
        print(f"  PASS: {test_name} ({len(masks)} masks)")
        passed.append(test_name)
    except Exception as e:
        print(f"  FAIL: {test_name}: {e}")
        traceback.print_exc()
        failed.append(test_name)

    # ── Test 2: prune_sparse_llm end-to-end ───────────────────────────────
    test_name = "prune_sparse_llm end-to-end at 30%"
    try:
        from sae_scoping.training.saliency.sparse_llm import prune_sparse_llm
        # Reload model
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager",
        )
        n_zeroed = prune_sparse_llm(
            model, tokenizer, calib_texts,
            sparsity=0.3, n_iterations=2, max_seq_len=512,
        )
        assert n_zeroed > 0, "No weights zeroed"
        post_zeros, _ = _count_zeros(model)
        assert post_zeros > baseline_zeros, (
            f"Expected more zeros: {post_zeros} <= {baseline_zeros}"
        )
        print(f"  Zeroed {n_zeroed:,}, total zeros {post_zeros:,}/{total_params:,}")
        print(f"  PASS: {test_name}")
        passed.append(test_name)
    except Exception as e:
        print(f"  FAIL: {test_name}: {e}")
        traceback.print_exc()
        failed.append(test_name)

    # ── Test 3: pruned model runs and loss is finite ──────────────────────
    test_name = "pruned model loss is finite"
    try:
        pruned_loss = _val_loss(model, tokenizer, eval_texts, device)
        print(f"  Pruned loss: {pruned_loss:.4f} (baseline: {baseline_loss:.4f})")
        assert pruned_loss < float("inf"), "Loss is inf"
        assert not torch.isnan(torch.tensor(pruned_loss)), "Loss is NaN"
        print(f"  PASS: {test_name}")
        passed.append(test_name)
    except Exception as e:
        print(f"  FAIL: {test_name}: {e}")
        traceback.print_exc()
        failed.append(test_name)

    # ── Test 4: pruned model can generate ─────────────────────────────────
    test_name = "pruned model generation"
    try:
        inputs = tokenizer("What is photosynthesis?", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        assert len(text) > 10, f"Generation too short: {repr(text)}"
        print(f"  Generated: {text[:100]}...")
        print(f"  PASS: {test_name}")
        passed.append(test_name)
    except Exception as e:
        print(f"  FAIL: {test_name}: {e}")
        traceback.print_exc()
        failed.append(test_name)

    # ── Test 5: return_masks for PGD ──────────────────────────────────────
    test_name = "return_masks for PGD"
    try:
        model2 = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager",
        )
        result = prune_sparse_llm(
            model2, tokenizer, calib_texts,
            sparsity=0.3, n_iterations=1, max_seq_len=512,
            return_masks=True,
        )
        n_z, masks = result
        assert isinstance(n_z, int) and n_z > 0
        assert isinstance(masks, dict) and len(masks) > 0
        for name, m in masks.items():
            assert m.device == torch.device("cpu"), "Mask should be on CPU"
        del model2, masks
        torch.cuda.empty_cache()
        print(f"  PASS: {test_name}")
        passed.append(test_name)
    except Exception as e:
        print(f"  FAIL: {test_name}: {e}")
        traceback.print_exc()
        failed.append(test_name)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
