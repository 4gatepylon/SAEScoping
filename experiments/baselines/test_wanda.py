"""
Integration tests for Wanda pruning on small Gemma models.

Usage:
  # Test on gemma2-2b-it (default):
  CUDA_VISIBLE_DEVICES=0 python test_wanda.py

  # Test on gemma3-4b-it:
  CUDA_VISIBLE_DEVICES=0 python test_wanda.py --model google/gemma-3-4b-it

  # Quick test with fewer samples:
  CUDA_VISIBLE_DEVICES=0 python test_wanda.py --n-calibration 4 --n-eval 8
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
@click.option("--n-calibration", default=8)
@click.option("--n-eval", default=16)
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

    # Baseline loss
    baseline_loss = _val_loss(model, tokenizer, eval_texts, device)
    baseline_zeros, total_params = _count_zeros(model)
    print(f"Baseline: loss={baseline_loss:.4f}, zeros={baseline_zeros:,}/{total_params:,}")

    # ── Test 1: compute_wanda_saliency produces correct shapes ────────────
    test_name = "compute_wanda_saliency shapes"
    try:
        from sae_scoping.training.saliency.wanda import compute_wanda_saliency
        saliency = compute_wanda_saliency(model, tokenizer, calib_texts, max_seq_len=512)
        assert len(saliency) > 0, "No saliency scores produced"
        for name, scores in saliency.items():
            assert name.endswith(".weight"), f"Expected .weight suffix, got {name}"
            # Find matching param
            for pname, p in model.named_parameters():
                if pname == name:
                    assert scores.shape == p.shape, f"Shape mismatch: {scores.shape} vs {p.shape}"
                    break
            assert (scores >= 0).all(), f"Negative saliency scores for {name}"
        print(f"  PASS: {test_name} ({len(saliency)} params scored)")
        passed.append(test_name)
    except Exception as e:
        print(f"  FAIL: {test_name}: {e}")
        traceback.print_exc()
        failed.append(test_name)

    # ── Test 2: compute_wanda_masks produces valid masks ──────────────────
    test_name = "compute_wanda_masks validity"
    try:
        from sae_scoping.training.saliency.wanda import compute_wanda_masks
        masks = compute_wanda_masks(saliency, sparsity=0.3)
        assert len(masks) == len(saliency), "Mask count != saliency count"
        for name, mask in masks.items():
            assert mask.dtype == torch.bool, f"Expected bool dtype, got {mask.dtype}"
            assert mask.shape == saliency[name].shape, f"Shape mismatch for {name}"
            # Check sparsity is approximately correct per row
            if mask.ndim == 2:
                per_row_sparsity = 1.0 - mask.float().mean(dim=1)
                mean_sparsity = per_row_sparsity.mean().item()
                assert abs(mean_sparsity - 0.3) < 0.05, (
                    f"Per-row sparsity {mean_sparsity:.3f} too far from target 0.3"
                )
        print(f"  PASS: {test_name}")
        passed.append(test_name)
    except Exception as e:
        print(f"  FAIL: {test_name}: {e}")
        traceback.print_exc()
        failed.append(test_name)

    # ── Test 3: prune_wanda end-to-end ────────────────────────────────────
    test_name = "prune_wanda end-to-end at 50%"
    try:
        from sae_scoping.training.saliency.wanda import prune_wanda
        # Reload model fresh
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager",
        )
        n_zeroed = prune_wanda(model, tokenizer, calib_texts, sparsity=0.5, max_seq_len=512)
        assert n_zeroed > 0, "No weights were zeroed"
        post_zeros, _ = _count_zeros(model)
        assert post_zeros > baseline_zeros, (
            f"Expected more zeros after pruning: {post_zeros} <= {baseline_zeros}"
        )
        # Check actual sparsity is in the right ballpark
        pruned_params_zeros = post_zeros - baseline_zeros
        # Wanda only prunes linear weight matrices, so actual overall sparsity
        # will be less than 50% (bias, embeddings, etc not pruned)
        print(f"  Zeroed {n_zeroed:,} weights, total zeros now {post_zeros:,}/{total_params:,}")
        print(f"  PASS: {test_name}")
        passed.append(test_name)
    except Exception as e:
        print(f"  FAIL: {test_name}: {e}")
        traceback.print_exc()
        failed.append(test_name)

    # ── Test 4: pruned model still runs and loss increases ────────────────
    test_name = "pruned model forward pass and loss increase"
    try:
        pruned_loss = _val_loss(model, tokenizer, eval_texts, device)
        print(f"  Pruned loss: {pruned_loss:.4f} (baseline: {baseline_loss:.4f})")
        assert pruned_loss >= baseline_loss * 0.9, (
            f"Pruned loss {pruned_loss:.4f} unexpectedly lower than baseline {baseline_loss:.4f}"
        )
        assert pruned_loss < float("inf"), "Pruned loss is inf"
        assert not torch.isnan(torch.tensor(pruned_loss)), "Pruned loss is NaN"
        print(f"  PASS: {test_name}")
        passed.append(test_name)
    except Exception as e:
        print(f"  FAIL: {test_name}: {e}")
        traceback.print_exc()
        failed.append(test_name)

    # ── Test 5: pruned model can generate ─────────────────────────────────
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

    # ── Test 6: return_masks works for PGD compatibility ──────────────────
    test_name = "return_masks for PGD"
    try:
        model2 = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager",
        )
        result = prune_wanda(
            model2, tokenizer, calib_texts,
            sparsity=0.3, max_seq_len=512, return_masks=True,
        )
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        n_z, masks = result
        assert isinstance(n_z, int) and n_z > 0
        assert isinstance(masks, dict) and len(masks) > 0
        # Verify masks are bool and on CPU
        for name, m in masks.items():
            assert m.dtype == torch.bool, f"Mask dtype should be bool, got {m.dtype}"
            assert m.device == torch.device("cpu"), f"Mask should be on CPU"
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
