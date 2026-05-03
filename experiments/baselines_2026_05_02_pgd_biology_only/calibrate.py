"""Calibration step: compute Wanda saliency map + vanilla baseline scores.

Inputs:
    --step-spec: path to a pre-compiled StepSpec YAML (contains all config)
    --no-wandb: disable W&B logging (optional override)
    Environment: $SAESCOPING_ARTIFACTS_LOCATION

Outputs (under $SAESCOPING_ARTIFACTS_LOCATION/{artifacts_subdir}/saliency_maps/{model}/{scope_domain}/):
    wanda_saliency.safetensors — per-weight saliency scores
    vanilla_scores.json — LLM judge + loss at zero sparsity (baseline for early stopping)
    calibration_sweep.json — per-sparsity metrics from the threshold sweep
    metadata.json — step configuration snapshot

Side effects:
    Logs calibration sweep metrics to W&B (if enabled) under calibration/ namespace.

Idempotency:
    Skips if wanda_saliency.safetensors already exists (unless --no-cache).

Failure mode:
    Partial outputs are left on disk. Re-running detects missing vanilla_scores.json
    and re-computes from the cached saliency map.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interface import CalibrateStep, ModelConfig, StepSpec, _slash_safe

# Imports from the library
from sae_scoping.training.saliency.wanda import (
    apply_masks_to_model,
    compute_wanda_masks,
    compute_wanda_saliency,
)
from sae_scoping.utils.cache import load_or_compute_safetensors
from sae_scoping.utils.model_loading import load_model_and_tokenizer


# ── Internals ─────────────────────────────────────────────────────────────


def _resolve_artifacts_root(spec: StepSpec) -> Path:
    base = os.environ.get("SAESCOPING_ARTIFACTS_LOCATION")
    if not base:
        raise click.ClickException("SAESCOPING_ARTIFACTS_LOCATION not set.")
    root = Path(base) / spec.artifacts_subdir
    root.mkdir(parents=True, exist_ok=True)
    return root


def _compute_loss(model, tokenizer, texts: list[str], max_seq_len: int, batch_size: int) -> float:
    """Compute mean cross-entropy loss on a list of texts."""
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    n_batches = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            padding=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        total_loss += out.loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def _count_model_sparsity(model) -> tuple[float, float]:
    """Return (overall_sparsity, nn_linear_sparsity)."""
    total_params = 0
    total_zeros = 0
    linear_params = 0
    linear_zeros = 0
    for name, param in model.named_parameters():
        n = param.numel()
        z = int((param.data == 0).sum().item())
        total_params += n
        total_zeros += z
        if ".weight" in name and param.dim() == 2:
            linear_params += n
            linear_zeros += z
    overall = total_zeros / max(total_params, 1)
    linear = linear_zeros / max(linear_params, 1)
    return overall, linear


def _run_calibration_sweep(
    model,
    tokenizer,
    saliency_map: dict[str, torch.Tensor],
    eval_texts: list[str],
    sparsities: list[float],
    max_seq_len: int,
    eval_batch_size: int,
    spec: StepSpec,
    scope_domain: str,
    wandb_run=None,
) -> list[dict]:
    """Apply masks at each sparsity threshold, evaluate, restore weights."""
    import copy

    results = []
    for sparsity in sparsities:
        print(f"\n[calibrate] Sweep sparsity={sparsity:.2f}")
        if sparsity == 0.0:
            loss = _compute_loss(model, tokenizer, eval_texts, max_seq_len, eval_batch_size)
            model_sp, linear_sp = _count_model_sparsity(model)
            entry = {
                "sparsity_threshold": sparsity,
                "loss": loss,
                "model_sparsity": model_sp,
                "nn_linear_sparsity": linear_sp,
            }
            if spec.llm_judge.enabled:
                scores = _run_llm_judge_all_domains(model, tokenizer, spec, scope_domain, max_seq_len)
                entry["llm_judge"] = scores
        else:
            masks = compute_wanda_masks(saliency_map, sparsity)
            # Save original weights, apply masks, eval, restore
            original_state = {name: param.data.clone() for name, param in model.named_parameters() if name in masks}
            apply_masks_to_model(model, masks)
            loss = _compute_loss(model, tokenizer, eval_texts, max_seq_len, eval_batch_size)
            model_sp, linear_sp = _count_model_sparsity(model)
            entry = {
                "sparsity_threshold": sparsity,
                "loss": loss,
                "model_sparsity": model_sp,
                "nn_linear_sparsity": linear_sp,
            }
            # Restore
            for name, data in original_state.items():
                param = dict(model.named_parameters())[name]
                param.data.copy_(data)

        results.append(entry)
        print(f"  loss={entry['loss']:.4f} model_sp={entry.get('model_sparsity', 0):.4f} linear_sp={entry.get('nn_linear_sparsity', 0):.4f}")

        if wandb_run is not None:
            wandb_run.log(
                {f"calibration/{k}": v for k, v in entry.items() if isinstance(v, (int, float))},
                step=int(sparsity * 100),
            )

    return results


def _run_llm_judge_all_domains(model, tokenizer, spec: StepSpec, scope_domain: str, max_seq_len: int) -> dict:
    """Run OneClickLLMJudgeScopingEval on all domains and return scores dict."""
    from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval

    judge_cfg = spec.llm_judge
    all_domains = spec.scope_domains

    domain_questions: dict[str, list[str]] = {}
    domain_answers: dict[str, list[str]] = {}
    for domain in all_domains:
        ds = load_dataset(spec.dataset_name, domain, split=judge_cfg.split)
        ds_subset = ds.select(range(min(judge_cfg.n_samples, len(ds))))
        domain_questions[domain] = [str(r["question"]) for r in ds_subset]
        domain_answers[domain] = [str(r["answer"]) for r in ds_subset]

    evaluator = OneClickLLMJudgeScopingEval(
        train_domain=scope_domain,
        judge_model=judge_cfg.judge_model,
        n_samples=judge_cfg.n_samples,
        generation_kwargs={"do_sample": False, "max_new_tokens": max_seq_len},
    )
    scores, _ = evaluator.evaluate(
        model,
        tokenizer,
        domain_questions=domain_questions,
        domain_answers=domain_answers,
    )
    return scores


def _write_metadata(output_dir: Path, model_config: ModelConfig, spec: StepSpec, scope_domain: str) -> None:
    meta = {
        "model_id": model_config.model_id,
        "scope_domain": scope_domain,
        "calibration_batch_size": model_config.calibration_batch_size,
        "calibration_max_seq_len": model_config.calibration_max_seq_len,
        "n_calibration": spec.n_calibration,
        "calibration_sweep_sparsities": spec.calibration_sweep_sparsities,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────


@click.command()
@click.option("--step-spec", required=True, type=click.Path(exists=True))
@click.option("--no-wandb", is_flag=True, default=False)
def main(step_spec: str, no_wandb: bool) -> None:
    """Run calibration: compute saliency map + vanilla baselines."""
    spec = StepSpec.from_yaml(step_spec)
    assert isinstance(spec.step, CalibrateStep)
    mc = spec.model_config
    scope_domain = spec.step.scope_domain
    model_safe = _slash_safe(mc.model_id)

    artifacts_root = _resolve_artifacts_root(spec)
    output_dir = artifacts_root / "saliency_maps" / model_safe / scope_domain
    saliency_path = output_dir / "wanda_saliency.safetensors"
    vanilla_path = output_dir / "vanilla_scores.json"
    sweep_path = output_dir / "calibration_sweep.json"

    # Idempotency: skip if all outputs exist
    if saliency_path.exists() and vanilla_path.exists() and sweep_path.exists():
        if not spec.no_cache:
            print(f"[calibrate] All outputs exist, skipping: {output_dir}")
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    # W&B setup
    wandb_run = None
    if spec.wandb.enabled and not no_wandb:
        import wandb

        os.environ["WANDB_DIR"] = str(artifacts_root / "wandb")
        wandb_run = wandb.init(
            project=spec.wandb.project,
            name=f"calibrate__{mc.model_id.split('/')[-1]}__{scope_domain}",
            config={"model_id": mc.model_id, "scope_domain": scope_domain, "step_type": "calibrate"},
            tags=["calibrate", scope_domain, mc.model_id.split("/")[-1]],
        )

    print(f"[calibrate] Model: {mc.model_id}")
    print(f"[calibrate] Scope domain: {scope_domain}")
    print(f"[calibrate] Output: {output_dir}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(mc.model_id, device=spec.device)

    # Load calibration data
    ds_train = load_dataset(spec.dataset_name, scope_domain, split="train")
    n_cal = min(spec.n_calibration, len(ds_train))
    calib_texts = [str(r["question"]) + "\n" + str(r["answer"]) for r in ds_train.select(range(n_cal))]
    print(f"[calibrate] Calibration texts: {len(calib_texts)}")

    # Compute or load saliency map
    saliency_map = load_or_compute_safetensors(
        path=saliency_path,
        compute_fn=lambda: compute_wanda_saliency(
            model,
            tokenizer,
            calib_texts,
            max_seq_len=mc.calibration_max_seq_len,
            batch_size=mc.calibration_batch_size,
        ),
        no_cache=spec.no_cache,
        label="Wanda saliency",
    )

    # Load eval data for the sweep
    ds_val = load_dataset(spec.dataset_name, scope_domain, split="validation")
    n_eval = min(spec.n_eval, len(ds_val))
    eval_texts = [str(r["question"]) + "\n" + str(r["answer"]) for r in ds_val.select(range(n_eval))]

    # Run calibration sweep
    sweep_results = _run_calibration_sweep(
        model=model,
        tokenizer=tokenizer,
        saliency_map=saliency_map,
        eval_texts=eval_texts,
        sparsities=spec.calibration_sweep_sparsities,
        max_seq_len=mc.calibration_max_seq_len,
        eval_batch_size=mc.wrapper.eval_batch_size,
        spec=spec,
        scope_domain=scope_domain,
        wandb_run=wandb_run,
    )

    # Save calibration sweep results
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"[calibrate] Saved calibration sweep → {sweep_path}")

    # Extract and save vanilla scores (sparsity=0.0 entry)
    vanilla_entry = next((r for r in sweep_results if r["sparsity_threshold"] == 0.0), None)
    if vanilla_entry is not None:
        vanilla_scores = {"loss": vanilla_entry["loss"]}
        if "llm_judge" in vanilla_entry:
            vanilla_scores.update(vanilla_entry["llm_judge"])
        with open(vanilla_path, "w") as f:
            json.dump(vanilla_scores, f, indent=2)
        print(f"[calibrate] Saved vanilla scores → {vanilla_path}")
    else:
        print("[calibrate] WARNING: no sparsity=0.0 in sweep, cannot save vanilla scores")

    # Save metadata
    _write_metadata(output_dir, mc, spec, scope_domain)

    if wandb_run is not None:
        wandb_run.finish()

    print(f"[calibrate] Done: {output_dir}")


if __name__ == "__main__":
    main()
