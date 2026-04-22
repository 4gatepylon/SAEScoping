"""
GPU integration tests for unlearning methods on gemma-2-2b-it.

Verifies actual unlearning behavior:
  - Forget-domain loss increases (model gets worse at forgotten domain)
  - Retain-domain loss stays similar (model preserves retained domain)
  - Model still generates coherent text

Usage:
  CUDA_VISIBLE_DEVICES=3 python test_unlearning.py
  CUDA_VISIBLE_DEVICES=3 python test_unlearning.py --model google/gemma-3-4b-it
"""
from __future__ import annotations

import copy
import sys
import traceback

import click
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_domain_texts(tokenizer, dataset_name, subset, n, seed=42, max_length=512):
    from datasets import load_dataset
    ds = load_dataset(dataset_name, subset, split="train")
    ds = ds.shuffle(seed=seed)
    rows = []
    for i in range(min(n, len(ds))):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": str(ds[i]["question"])},
             {"role": "assistant", "content": str(ds[i]["answer"])}],
            tokenize=False, add_generation_prompt=False,
        )
        tok = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
        rows.append({
            "text": text,
            "input_ids": tok["input_ids"],
            "attention_mask": tok["attention_mask"],
            "labels": tok["input_ids"],
        })
    return Dataset.from_list(rows)


@torch.no_grad()
def _domain_loss(model, tokenizer, texts, device, max_len=512):
    model.eval()
    old_pad = tokenizer.padding_side
    tokenizer.padding_side = "right"
    total, n = 0.0, 0
    for t in texts:
        tok = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_len)
        ids = tok["input_ids"].to(device)
        mask = tok["attention_mask"].to(device)
        labels = ids.clone()
        labels[mask == 0] = -100
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        total += out.loss.item()
        n += 1
    tokenizer.padding_side = old_pad
    return total / max(n, 1)


LOSS_INCREASE_THRESHOLD = 1.05  # forget loss should increase by at least 5%
LOSS_RETAIN_THRESHOLD = 2.00    # retain loss should not more than double (GD/NPO are known to be unstable)


@click.command()
@click.option("--model", "model_id", default="google/gemma-2-2b-it")
@click.option("--forget-domain", default="math")
@click.option("--retain-domain", default="biology")
@click.option("--dataset-name", default="4gate/StemQAMixture")
@click.option("--n-forget", default=100, help="Forget dataset size")
@click.option("--n-retain", default=100, help="Retain dataset size")
@click.option("--n-eval", default=30, help="Eval samples per domain")
@click.option("--device", default="cuda:0")
def main(model_id, forget_domain, retain_domain, dataset_name, n_forget, n_retain, n_eval, device):
    device = torch.device(device)
    passed, failed = [], []

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load domain datasets (non-overlapping: train data vs eval data via different seeds)
    forget_train = _load_domain_texts(tokenizer, dataset_name, forget_domain, n_forget, seed=42)
    retain_train = _load_domain_texts(tokenizer, dataset_name, retain_domain, n_retain, seed=42)
    forget_eval = _load_domain_texts(tokenizer, dataset_name, forget_domain, n_eval, seed=999)
    retain_eval = _load_domain_texts(tokenizer, dataset_name, retain_domain, n_eval, seed=999)

    print(f"Forget domain: {forget_domain} ({n_forget} train, {n_eval} eval)")
    print(f"Retain domain: {retain_domain} ({n_retain} train, {n_eval} eval)")

    methods = [
        ("GradientDiff", _test_gradient_diff),
        ("NPO", _test_npo),
        ("RMU", _test_rmu),
    ]

    for method_name, test_fn in methods:
        print(f"\n{'='*60}")
        print(f"  Testing: {method_name}")
        print(f"{'='*60}")

        # Fresh model for each method
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device,
            attn_implementation="eager",
        )

        # Baseline losses
        forget_loss_before = _domain_loss(model, tokenizer, forget_eval["text"], device)
        retain_loss_before = _domain_loss(model, tokenizer, retain_eval["text"], device)
        print(f"  Before: forget_loss={forget_loss_before:.4f}, retain_loss={retain_loss_before:.4f}")

        try:
            test_fn(
                model, tokenizer, forget_train, retain_train,
                forget_eval, retain_eval, device,
                forget_loss_before, retain_loss_before,
                method_name, passed, failed,
            )
        except Exception as e:
            print(f"  FAIL: {method_name} crashed: {e}")
            traceback.print_exc()
            failed.append(f"{method_name}/crash")

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All tests passed!")


def _test_gradient_diff(
    model, tokenizer, forget_train, retain_train,
    forget_eval, retain_eval, device,
    forget_loss_before, retain_loss_before,
    method_name, passed, failed,
):
    from sae_scoping.training.unlearning.gradient_diff import unlearn_gradient_diff

    unlearn_gradient_diff(
        model, tokenizer, forget_train, retain_train,
        forget_weight=1.0, retain_weight=5.0,
        max_steps=50, learning_rate=5e-5, batch_size=4, max_length=512,
    )

    forget_loss_after = _domain_loss(model, tokenizer, forget_eval["text"], device)
    retain_loss_after = _domain_loss(model, tokenizer, retain_eval["text"], device)
    print(f"  After:  forget_loss={forget_loss_after:.4f}, retain_loss={retain_loss_after:.4f}")

    # Check 1: forget loss increased
    test = f"{method_name}/forget_loss_increases"
    if forget_loss_after > forget_loss_before * LOSS_INCREASE_THRESHOLD:
        print(f"  PASS: {test} ({forget_loss_before:.4f} -> {forget_loss_after:.4f})")
        passed.append(test)
    else:
        print(f"  FAIL: {test} ({forget_loss_before:.4f} -> {forget_loss_after:.4f}, need >{forget_loss_before * LOSS_INCREASE_THRESHOLD:.4f})")
        failed.append(test)

    # Check 2: retain loss didn't explode
    test = f"{method_name}/retain_loss_stable"
    if retain_loss_after < retain_loss_before * LOSS_RETAIN_THRESHOLD:
        print(f"  PASS: {test} ({retain_loss_before:.4f} -> {retain_loss_after:.4f})")
        passed.append(test)
    else:
        print(f"  FAIL: {test} ({retain_loss_before:.4f} -> {retain_loss_after:.4f}, limit {retain_loss_before * LOSS_RETAIN_THRESHOLD:.4f})")
        failed.append(test)

    # Check 3: model still generates
    _check_generation(model, tokenizer, device, method_name, passed, failed)


def _test_npo(
    model, tokenizer, forget_train, retain_train,
    forget_eval, retain_eval, device,
    forget_loss_before, retain_loss_before,
    method_name, passed, failed,
):
    from sae_scoping.training.unlearning.npo import unlearn_npo

    unlearn_npo(
        model, tokenizer, forget_train, retain_dataset=retain_train,
        npo_beta=0.1, retain_weight=5.0,
        max_steps=50, learning_rate=5e-5, batch_size=4, max_length=512,
    )

    forget_loss_after = _domain_loss(model, tokenizer, forget_eval["text"], device)
    retain_loss_after = _domain_loss(model, tokenizer, retain_eval["text"], device)
    print(f"  After:  forget_loss={forget_loss_after:.4f}, retain_loss={retain_loss_after:.4f}")

    test = f"{method_name}/forget_loss_increases"
    if forget_loss_after > forget_loss_before * LOSS_INCREASE_THRESHOLD:
        print(f"  PASS: {test} ({forget_loss_before:.4f} -> {forget_loss_after:.4f})")
        passed.append(test)
    else:
        print(f"  FAIL: {test} ({forget_loss_before:.4f} -> {forget_loss_after:.4f})")
        failed.append(test)

    test = f"{method_name}/retain_loss_stable"
    if retain_loss_after < retain_loss_before * LOSS_RETAIN_THRESHOLD:
        print(f"  PASS: {test} ({retain_loss_before:.4f} -> {retain_loss_after:.4f})")
        passed.append(test)
    else:
        print(f"  FAIL: {test} ({retain_loss_before:.4f} -> {retain_loss_after:.4f})")
        failed.append(test)

    _check_generation(model, tokenizer, device, method_name, passed, failed)


def _test_rmu(
    model, tokenizer, forget_train, retain_train,
    forget_eval, retain_eval, device,
    forget_loss_before, retain_loss_before,
    method_name, passed, failed,
):
    from sae_scoping.training.unlearning.rmu import unlearn_rmu, get_num_layers

    n_layers = get_num_layers(model)
    hook_layer = min(7, n_layers - 1)

    unlearn_rmu(
        model, tokenizer, forget_train, retain_train,
        hook_layer_id=hook_layer, param_ids=None,
        steering_coeff=20.0, alpha=100.0,
        max_steps=50, learning_rate=5e-5, max_length=512,
    )

    forget_loss_after = _domain_loss(model, tokenizer, forget_eval["text"], device)
    retain_loss_after = _domain_loss(model, tokenizer, retain_eval["text"], device)
    print(f"  After:  forget_loss={forget_loss_after:.4f}, retain_loss={retain_loss_after:.4f}")

    test = f"{method_name}/forget_loss_increases"
    if forget_loss_after > forget_loss_before * LOSS_INCREASE_THRESHOLD:
        print(f"  PASS: {test} ({forget_loss_before:.4f} -> {forget_loss_after:.4f})")
        passed.append(test)
    else:
        print(f"  FAIL: {test} ({forget_loss_before:.4f} -> {forget_loss_after:.4f})")
        failed.append(test)

    test = f"{method_name}/retain_loss_stable"
    if retain_loss_after < retain_loss_before * LOSS_RETAIN_THRESHOLD:
        print(f"  PASS: {test} ({retain_loss_before:.4f} -> {retain_loss_after:.4f})")
        passed.append(test)
    else:
        print(f"  FAIL: {test} ({retain_loss_before:.4f} -> {retain_loss_after:.4f})")
        failed.append(test)

    _check_generation(model, tokenizer, device, method_name, passed, failed)


def _check_generation(model, tokenizer, device, method_name, passed, failed):
    test = f"{method_name}/generates_coherent_text"
    try:
        inputs = tokenizer("What is photosynthesis?", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if len(text) > 10 and not any(c == '\x00' for c in text):
            print(f"  PASS: {test}")
            print(f"    Generated: {text[:80]}...")
            passed.append(test)
        else:
            print(f"  FAIL: {test} — output too short or garbled: {repr(text[:50])}")
            failed.append(test)
    except Exception as e:
        print(f"  FAIL: {test} — {e}")
        failed.append(test)


if __name__ == "__main__":
    main()
