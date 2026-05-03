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
import torch.nn as nn
from datasets import load_dataset
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))

from interface import CalibrateStep, ModelConfig, StepSpec, _slash_safe
from utils import maybe_init_wandb, resolve_artifacts_root

from sae_scoping.datasets.qa_datasets import format_as_sft_text
from sae_scoping.evaluation.loss import compute_loss, count_zeros
from sae_scoping.training.saliency.wanda import (
    apply_masks_to_model,
    compute_wanda_masks,
    compute_wanda_saliency,
)
from sae_scoping.utils.cache import load_or_compute_safetensors
from sae_scoping.utils.model_loading import load_model_and_tokenizer


# ── Internals ─────────────────────────────────────────────────────────────


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
    results = []
    for sparsity in sparsities:
        print(f"\n[calibrate] Sweep sparsity={sparsity:.2f}")
        # NOTE: mask is zero at sparsity X -> weights are zero at sparsity Y
        # for all Y >= X
        if sparsity > 0.0:
            masks = compute_wanda_masks(saliency_map, sparsity)
            apply_masks_to_model(model, masks)

        loss = compute_loss(model, tokenizer, eval_texts, max_seq_len=max_seq_len, batch_size=eval_batch_size)
        zeros, total = count_zeros(model)
        lin_zeros, lin_total = count_zeros(model, wanda_prunable_only=True)
        entry = {
            "sparsity_threshold": sparsity,
            "loss": loss,
            "model_sparsity": (zeros / total) if total > 0 else 0.0,
            "nn_linear_sparsity": (lin_zeros / lin_total) if lin_total > 0 else 0.0,
        }
        if sparsity == 0.0 and spec.llm_judge.enabled:
            entry["llm_judge"] = _run_llm_judge_all_domains(model, tokenizer, spec, scope_domain, max_seq_len)

        results.append(entry)
        print(f"  loss={entry['loss']:.4f} model_sp={entry['model_sparsity']:.4f} linear_sp={entry['nn_linear_sparsity']:.4f}")

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
    selection_amount = min(judge_cfg.n_samples, 10_000)
    domain_datasets = {
        domain: load_dataset(spec.dataset_name, domain, split=judge_cfg.split).select(range(selection_amount)) for domain in spec.scope_domains
    }
    evaluator = OneClickLLMJudgeScopingEval(
        train_domain=scope_domain,
        judge_model=judge_cfg.judge_model,
        n_samples=judge_cfg.n_samples,
        generation_kwargs={"do_sample": False, "max_new_tokens": max_seq_len},
        domain_datasets=domain_datasets,
    )
    scores, _ = evaluator.evaluate(model, tokenizer)
    return scores


# ── Dry-run ───────────────────────────────────────────────────────────────


def _dry_run_calibrate(spec: StepSpec, output_dir: Path, saliency_path: Path, vanilla_path: Path, sweep_path: Path) -> None:
    """Dry-run: load model to CPU, write random saliency map + stub scores."""
    from transformers import AutoModelForCausalLM

    mc = spec.model_cfg
    print(f"[calibrate][dry-run] Loading {mc.model_id} to CPU (weights only, no GPU)...")
    model = AutoModelForCausalLM.from_pretrained(mc.model_id, torch_dtype=torch.bfloat16, device_map="cpu")

    # Random saliency map with correct tensor names and shapes
    saliency_map: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            key = f"{name}.weight"
            saliency_map[key] = torch.rand_like(module.weight.data, device="cpu")
    save_file(saliency_map, str(saliency_path))
    print(f"[calibrate][dry-run] Wrote random saliency ({len(saliency_map)} tensors) → {saliency_path}")

    # Stub vanilla scores
    with open(vanilla_path, "w") as f:
        json.dump({"loss": 0.0, "dry_run": True}, f, indent=2)
    print(f"[calibrate][dry-run] Wrote stub vanilla scores → {vanilla_path}")

    # Stub calibration sweep
    sweep_results = [{"sparsity_threshold": sp, "loss": 0.0, "dry_run": True} for sp in spec.calibration_sweep_sparsities]
    with open(sweep_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"[calibrate][dry-run] Wrote stub sweep → {sweep_path}")

    _write_metadata(output_dir, mc, spec, spec.step.scope_domain)
    (output_dir / "COMPLETED").write_text("step_id=calibrate\n")
    del model
    print(f"[calibrate][dry-run] Done: {output_dir}")


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
    """Run calibration: compute saliency map + vanilla baselines.

    CONTRACT: exit code must be 0 iff the step fully succeeded.
    The scheduler treats any non-zero exit as FAILED and skips all dependents.
    """
    spec = StepSpec.from_yaml(step_spec)
    if spec.dry_run:
        print("[calibrate] *** DRY RUN MODE ***")
    assert isinstance(spec.step, CalibrateStep)
    mc = spec.model_cfg
    scope_domain = spec.step.scope_domain
    model_safe = _slash_safe(mc.model_id)

    artifacts_root = resolve_artifacts_root(spec)
    output_dir = artifacts_root / "saliency_maps" / model_safe / scope_domain
    saliency_path = output_dir / "wanda_saliency.safetensors"
    vanilla_path = output_dir / "vanilla_scores.json"
    sweep_path = output_dir / "calibration_sweep.json"

    # Idempotency: skip only if COMPLETED marker exists (written last by previous run)
    if (output_dir / "COMPLETED").exists() and not spec.no_cache:
        print(f"[calibrate] COMPLETED marker found, skipping: {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if spec.dry_run:
        _dry_run_calibrate(spec, output_dir, saliency_path, vanilla_path, sweep_path)
        return

    # W&B setup
    wandb_run = maybe_init_wandb(
        spec,
        artifacts_root,
        name=f"calibrate__{mc.model_id.split('/')[-1]}__{scope_domain}",
        no_wandb=no_wandb,
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
    calib_texts = format_as_sft_text(ds_train.select(range(n_cal)), tokenizer)
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
    eval_texts = format_as_sft_text(ds_val.select(range(n_eval)), tokenizer)

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

    (output_dir / "COMPLETED").write_text("step_id=calibrate\n")
    print(f"[calibrate] Done: {output_dir}")


if __name__ == "__main__":
    main()
