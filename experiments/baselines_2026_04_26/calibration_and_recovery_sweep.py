"""
Calibration, pruning, and recovery sweep for saliency-based baselines.

Pipeline per grid point (method × model × domain × sparsity):
  1. Calibrate — compute saliency scores (cached to disk as safetensors)
  2. Prune    — threshold saliency into boolean keep-masks, zero weights
  3. Evaluate — cross-entropy loss on pruned model
  4. Recover  — PGD fine-tuning on in-domain data (optional, on by default)
  5. Evaluate — cross-entropy loss on recovered model

Elicitation (adversarial fine-tuning on OOD data) is NOT done here.
The elicitation sweep consumes the artifacts this script produces
(masks + recovered model checkpoints). See "Artifact Interface" below.

Artifact layout on disk:
  {artifact_dir}/
    {model_slug}/
      {domain}/
        wanda_saliency.safetensors        <- from library cache convention
        random_saliency.safetensors
        ema_grads.safetensors
        taylor_saliency.safetensors
        gradient_saliency.safetensors
        {method}/
          result.json                      <- MethodDomainResult (pydantic)
          sp_0.30/
            masks.safetensors              <- boolean keep-masks (True=keep)
            metrics.json                   <- SparsityResult (pydantic)
            recovered/                     <- HF checkpoint (model + tokenizer)
          sp_0.50/
            ...
    manifest.json                          <- SweepManifest (top-level index)

Artifact Interface (for elicitation_sweep.py):
  1. Load manifest:
       manifest = SweepManifest.load(artifact_dir / "manifest.json")
  2. For each entry in manifest.entries, for each sparsity result:
       masks = load_file(result.masks_path)  # dict[str, Tensor], bool dtype
       model = AutoModelForCausalLM.from_pretrained(result.recovered_model_dir)
       tokenizer = AutoTokenizer.from_pretrained(result.recovered_model_dir)
  3. Run PGD elicitation:
       trainer = PGDSFTTrainer(masks=masks, model=model, ...)
       trainer.train()

Usage:
  # Full grid on GPUs 1,6:
  python calibration_and_recovery_sweep.py launch --gpus 1,6

  # Eval-only (skip recovery training):
  python calibration_and_recovery_sweep.py launch --gpus 1,6 --no-recovery

  # Dry run (show jobs without executing):
  python calibration_and_recovery_sweep.py launch --gpus 1,6 --dry-run

  # Single worker (called by launcher, or run manually for debugging):
  python calibration_and_recovery_sweep.py worker \
      --method wanda --model-id google/gemma-2-9b-it --domain biology
"""
from __future__ import annotations

import gc
import json
import sys
import time  # BUG TODO(adriano): unused import, delete
import traceback
from pathlib import Path

import click

from helpers import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATASET,
    DEFAULT_SPARSITY_LEVELS,
    DOMAINS,
    METHODS,
    MODELS,
    MethodDomainResult,
    SparsityResult,
    SweepManifest,
    make_sft_config,
    model_slug as _model_slug,
    saliency_path as _saliency_path,
    sparsity_dir as _sparsity_dir,
)


# ============================================================================
# Worker logic
# ============================================================================


def run_worker(
    method: str,
    model_id: str,
    domain: str,
    sparsity_levels: list[float],
    artifact_dir: Path,
    dataset_name: str = DEFAULT_DATASET,
    n_calibration: int = 128,
    n_recovery: int = 500,
    n_eval_train: int = 200,
    n_eval_test: int = 200,
    max_seq_len: int = 1024,
    recovery: bool = True,
    sft_overrides: dict | None = None,
) -> MethodDomainResult:
    """Run calibration + pruning + optional recovery for one (method, model, domain).

    Sweeps across sparsity_levels sequentially, reusing the model in GPU memory.
    """
    # Heavy imports deferred to here so that the `launch` and `--dry-run` paths
    # (which only build JobSpecs) don't pay the multi-second torch/transformers
    # import cost.
    import torch
    from datasets import load_dataset as hf_load_dataset
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sae_scoping.datasets.qa_datasets import format_as_sft_dataset, format_as_sft_text
    from sae_scoping.evaluation.loss import compute_loss
    from sae_scoping.training.saliency.dispatch import compute_saliency, masks_for_sparsity
    from sae_scoping.training.saliency.wanda import apply_masks_to_model
    from sae_scoping.training.weight_pruning import restore_original_weights, save_original_weights

    device = torch.device("cuda:0")

    print(f"\n{'=' * 70}")
    print(f"  Worker: {method} / {model_id} / {domain}")
    print(f"  Sparsity levels: {sparsity_levels}")
    print(f"  Recovery: {'ON' if recovery else 'OFF'}")
    print(f"{'=' * 70}\n")

    # --- Load model ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="eager",  # BUG TODO(adriano): was dtype= which may be silently ignored on older transformers
    )

    # --- Load data (non-overlapping splits from a single shuffle) ---
    ds = hf_load_dataset(dataset_name, domain, split="train").shuffle(seed=42)
    total_needed = n_calibration + n_recovery + n_eval_train + n_eval_test
    if len(ds) < total_needed:
        raise ValueError(
            f"{dataset_name}/{domain} has {len(ds)} rows, need {total_needed}"
        )

    i = 0
    calib_ds = ds.select(range(i, i + n_calibration)); i += n_calibration
    recovery_ds = ds.select(range(i, i + n_recovery)); i += n_recovery
    eval_train_ds = ds.select(range(i, i + n_eval_train)); i += n_eval_train
    eval_test_ds = ds.select(range(i, i + n_eval_test)); i += n_eval_test

    calib_texts = format_as_sft_text(calib_ds, tokenizer)
    eval_train_texts = format_as_sft_text(eval_train_ds, tokenizer)
    eval_test_texts = format_as_sft_text(eval_test_ds, tokenizer)
    recovery_sft_ds = format_as_sft_dataset(recovery_ds, tokenizer) if recovery else None

    # --- Calibrate (cached via library convention) ---
    saliency_data = compute_saliency(
        method, model, tokenizer, calib_texts, max_seq_len,
        cache_dir=artifact_dir,
        model_id=model_id,
        dataset_subset=domain,
        dataset_name=dataset_name,
    )
    sal_path = _saliency_path(artifact_dir, model_id, domain, method)

    # --- Save original weights so we can restore between sparsity levels ---
    original_weights = save_original_weights(model)

    # --- Sweep sparsity levels ---
    sparsity_results: list[SparsityResult] = []

    for sp in sparsity_levels:
        print(f"\n--- {method} / {_model_slug(model_id)} / {domain} @ {sp:.0%} ---")

        sp_dir = _sparsity_dir(artifact_dir, model_id, domain, method, sp)
        sp_dir.mkdir(parents=True, exist_ok=True)

        restore_original_weights(model, original_weights)

        # Compute and save masks
        masks = masks_for_sparsity(method, saliency_data, sp)
        masks_path = sp_dir / "masks.safetensors"
        save_file({k: v.contiguous() for k, v in masks.items()}, str(masks_path))

        # Prune
        n_zeroed = apply_masks_to_model(model, masks)
        print(f"  Pruned: {n_zeroed:,} weights zeroed")

        # Eval pruned model
        pruned_train_loss = compute_loss(model, tokenizer, eval_train_texts, max_seq_len)
        pruned_test_loss = compute_loss(model, tokenizer, eval_test_texts, max_seq_len)
        print(f"  Pruned loss: train={pruned_train_loss:.4f} test={pruned_test_loss:.4f}")

        result = SparsityResult(
            sparsity=sp,
            masks_path=masks_path,
            pruned_loss_train=pruned_train_loss,
            pruned_loss_test=pruned_test_loss,
        )

        # Recovery (PGD SFT on in-domain data)
        if recovery and recovery_sft_ds is not None:
            try:
                from sae_scoping.training.pgd_trainer import PGDSFTTrainer

                recovered_dir = sp_dir / "recovered"

                # SFT config resolved from sft_defaults.yaml with optional
                # CLI overrides. Reserved fields (output_dir, save_strategy,
                # report_to, dataset_text_field) are set automatically.
                recovery_args = make_sft_config(
                    phase="recovery",
                    model_id=model_id,
                    output_dir=sp_dir / "_recovery_checkpoints",
                    overrides=sft_overrides,
                )

                trainer = PGDSFTTrainer(
                    masks=masks,
                    model=model,
                    processing_class=tokenizer,
                    train_dataset=recovery_sft_ds,
                    args=recovery_args,
                )
                trainer.train()

                model.save_pretrained(str(recovered_dir))
                tokenizer.save_pretrained(str(recovered_dir))

                recovered_train_loss = compute_loss(
                    model, tokenizer, eval_train_texts, max_seq_len,
                )
                recovered_test_loss = compute_loss(
                    model, tokenizer, eval_test_texts, max_seq_len,
                )
                print(
                    f"  Recovered loss: train={recovered_train_loss:.4f}"
                    f" test={recovered_test_loss:.4f}"
                )

                result.recovered_model_dir = recovered_dir
                result.recovered_loss_train = recovered_train_loss
                result.recovered_loss_test = recovered_test_loss

                del trainer
                gc.collect()
                torch.cuda.empty_cache()

            except Exception:
                print(f"  Recovery FAILED:\n{traceback.format_exc()}")

        # Save per-sparsity metrics
        (sp_dir / "metrics.json").write_text(result.model_dump_json(indent=2))
        sparsity_results.append(result)

    # --- Save per-worker result ---
    method_result = MethodDomainResult(
        method=method,
        model_id=model_id,
        domain=domain,
        dataset_name=dataset_name,
        saliency_path=sal_path,
        results=sparsity_results,
    )
    result_path = artifact_dir / _model_slug(model_id) / domain / method / "result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(method_result.model_dump_json(indent=2))

    del original_weights, model
    gc.collect()
    torch.cuda.empty_cache()

    return method_result


# ============================================================================
# CLI
# ============================================================================


@click.group()
def cli():
    """Calibration, pruning, and recovery sweep for saliency baselines."""


@cli.command()
@click.option("--method", type=click.Choice(METHODS), required=True)
@click.option("--model-id", required=True)
@click.option("--domain", required=True)
@click.option(
    "--sparsity-levels",
    default=",".join(str(s) for s in DEFAULT_SPARSITY_LEVELS),
    help="Comma-separated sparsity fractions",
)
@click.option("--artifact-dir", type=click.Path(path_type=Path), default=DEFAULT_ARTIFACT_DIR)
@click.option("--dataset-name", default=DEFAULT_DATASET)
@click.option("--no-recovery", is_flag=True, help="Skip recovery training (eval-only)")
@click.option("--n-calibration", default=128)
@click.option("--n-recovery", default=500, help="Samples for recovery SFT")
@click.option("--max-seq-len", default=1024)
@click.option(
    "--sft-overrides", default=None,
    help="JSON string of SFTConfig overrides (e.g. '{\"learning_rate\": 3e-5}')",
)
def worker(
    method, model_id, domain, sparsity_levels, artifact_dir,
    dataset_name, no_recovery, n_calibration, n_recovery, max_seq_len,
    sft_overrides,
):
    """Run one (method, model, domain) sweep. Called by `launch` or manually."""
    levels = sorted(float(s.strip()) for s in sparsity_levels.split(","))
    overrides = json.loads(sft_overrides) if sft_overrides else None
    run_worker(
        method=method,
        model_id=model_id,
        domain=domain,
        sparsity_levels=levels,
        artifact_dir=artifact_dir,
        dataset_name=dataset_name,
        recovery=not no_recovery,
        n_calibration=n_calibration,
        n_recovery=n_recovery,
        max_seq_len=max_seq_len,
        sft_overrides=overrides,
    )


@cli.command()
@click.option("--gpus", required=True, help="Comma-separated GPU IDs (e.g. 1,6)")
@click.option("--methods", default=",".join(METHODS), help="Comma-separated methods")
@click.option("--models", default=",".join(MODELS), help="Comma-separated HF model IDs")
@click.option("--domains", default=",".join(DOMAINS), help="Comma-separated domain names")
@click.option(
    "--sparsity-levels",
    default=",".join(str(s) for s in DEFAULT_SPARSITY_LEVELS),
    help="Comma-separated sparsity fractions",
)
@click.option("--artifact-dir", type=click.Path(path_type=Path), default=DEFAULT_ARTIFACT_DIR)
@click.option("--dataset-name", default=DEFAULT_DATASET)
@click.option("--no-recovery", is_flag=True, help="Skip recovery training (eval-only)")
@click.option("--dry-run", is_flag=True, help="Print jobs without executing")
@click.option("--log-dir", type=click.Path(path_type=Path), default=Path("./sweep_logs"))
@click.option(
    "--sft-overrides", default=None,
    help="JSON string of SFTConfig overrides, forwarded to all workers",
)
def launch(
    gpus, methods, models, domains, sparsity_levels, artifact_dir,
    dataset_name, no_recovery, dry_run, log_dir, sft_overrides,
):
    """Build the full experiment grid and launch via the generic scheduler."""
    from sae_scoping.infrastructure.scheduler import JobSpec, run_jobs

    gpu_ids = [int(g.strip()) for g in gpus.split(",")]
    method_list = [m.strip() for m in methods.split(",")]
    model_list = [m.strip() for m in models.split(",")]
    domain_list = [d.strip() for d in domains.split(",")]

    jobs: list[JobSpec] = []
    for mdl in model_list:
        for dom in domain_list:
            for meth in method_list:
                name = f"{meth}/{_model_slug(mdl)}/{dom}"
                cmd = [
                    sys.executable, __file__, "worker",
                    "--method", meth,
                    "--model-id", mdl,
                    "--domain", dom,
                    "--sparsity-levels", sparsity_levels,
                    "--artifact-dir", str(artifact_dir.resolve()),
                    "--dataset-name", dataset_name,
                ]
                if no_recovery:
                    cmd.append("--no-recovery")
                if sft_overrides:
                    cmd += ["--sft-overrides", sft_overrides]
                jobs.append(JobSpec(command=cmd, n_gpus=1, name=name))

    n_methods = len(method_list)
    n_models = len(model_list)
    n_domains = len(domain_list)
    print(f"Grid: {n_methods} methods × {n_models} models × {n_domains} domains = {len(jobs)} jobs")
    print(f"GPUs: {gpu_ids}")

    results = run_jobs(jobs, gpu_ids=gpu_ids, log_dir=str(log_dir), dry_run=dry_run)

    if not dry_run:
        manifest = SweepManifest(artifact_dir=artifact_dir)
        for mdl in model_list:
            for dom in domain_list:
                for meth in method_list:
                    rp = artifact_dir / _model_slug(mdl) / dom / meth / "result.json"
                    if rp.exists():
                        entry = MethodDomainResult.model_validate_json(rp.read_text())
                        manifest.entries.append(entry)
                    else:
                        print(f"  WARNING: missing result for {meth}/{mdl}/{dom}")
        manifest.save()
        print(f"\nManifest: {artifact_dir / 'manifest.json'}")
        print(f"  {len(manifest.entries)} entries written")

    return results


if __name__ == "__main__":
    cli()
