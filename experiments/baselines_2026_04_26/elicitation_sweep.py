"""
Elicitation sweep: adversarial PGD SFT on OOD domains to test scoping robustness.

Takes outputs from calibration_and_recovery_sweep.py (recovered model checkpoints
+ pruning masks) and runs elicitation — adversarial SFT on OOD data using PGD to
keep pruned weights at zero.

For each in-domain Xi, elicits on each other StemQA domain Yj (Xi != Yj), then
evaluates on both Xi and Yj. This tests whether scoping actually suppresses OOD
capability: if elicitation on Yj recovers high Yj performance, scoping is weak.

Single elicitation (one GPU):
  python elicitation_sweep.py \
      --artifact-dir ./artifacts \
      --method wanda --model google/gemma-2-2b-it \
      --in-domain biology --ood-domain chemistry --sparsity 0.5

Full grid (all OOD domains for a given in-domain):
  python elicitation_sweep.py \
      --artifact-dir ./artifacts \
      --method wanda --model google/gemma-2-2b-it \
      --in-domain biology --sparsity 0.5

Launch all (method x in_domain x sparsity x ood_domain) via scheduler:
  python elicitation_sweep.py \
      --artifact-dir ./artifacts \
      --launch --gpus 0,1,2,3
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import click
import torch
import wandb
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from helpers import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATASET,
    DOMAINS,
    SweepManifest,
    make_sft_config,
    masks_path as _masks_path,
    model_slug as _model_slug,
    recovered_model_dir as _recovered_model_dir,
    sparsity_dir as _sparsity_dir,
)
from sae_scoping.datasets.qa_datasets import (
    format_as_sft_dataset,
    load_nonoverlapping_splits,
    load_qa_dataset,
)
from sae_scoping.evaluation.loss import compute_loss
from sae_scoping.training.pgd_trainer import PGDSFTTrainer, build_pgd_masks_from_model



def _load_recovered_model_and_masks(
    artifact_dir: Path, model_id: str, domain: str, method: str,
    sparsity: float, device: torch.device,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, dict[str, torch.Tensor]]:
    model_path = _recovered_model_dir(artifact_dir, model_id, domain, method, sparsity)
    mp = _masks_path(artifact_dir, model_id, domain, method, sparsity)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Recovered model not found at {model_path}. "
            "Run calibration_and_recovery_sweep.py first."
        )

    print(f"Loading recovered model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )

    if mp.exists():
        print(f"Loading masks from {mp}")
        masks = load_file(str(mp))
        masks = {k: v.bool() for k, v in masks.items()}
    else:
        print("No masks file found — deriving masks from model zero pattern")
        masks = build_pgd_masks_from_model(model)

    print(f"  {len(masks)} masked parameters")
    return model, tokenizer, masks


def _save_elicited_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved elicited model to {output_dir}")


def elicit_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    masks: dict[str, torch.Tensor],
    model_id: str,
    ood_domain: str,
    n_train: int = 512,
    sft_overrides: dict | None = None,
    output_dir: Path | None = None,
) -> AutoModelForCausalLM:
    """Run PGD SFT elicitation on OOD data. Modifies model in-place."""
    print(f"\n--- Elicitation on {ood_domain} ---")

    ood_dataset = load_qa_dataset(
        DEFAULT_DATASET, ood_domain, split="train", n=n_train, seed=123
    )
    sft_dataset = format_as_sft_dataset(ood_dataset, tokenizer)

    train_output = output_dir / "train_output" if output_dir else Path("./deleteme_elicit")

    # SFT config resolved from sft_defaults.yaml with optional CLI overrides.
    # Reserved fields (output_dir, save_strategy, etc.) are set automatically.
    sft_config = make_sft_config(
        phase="elicitation",
        model_id=model_id,
        output_dir=train_output,
        overrides=sft_overrides,
    )

    trainer = PGDSFTTrainer(
        masks=masks,
        validate_sparsity=True,
        model=model,
        args=sft_config,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"  Elicitation training completed in {elapsed:.1f}s")

    return model


def evaluate_elicited(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    in_domain: str,
    ood_domain: str,
    n_eval: int = 200,
    max_seq_len: int = 1024,
    no_judge: bool = True,
    n_judge_samples: int = 50,
) -> dict:
    """Evaluate an elicited model on in-domain (utility) and OOD (safety)."""
    results: dict = {}

    for domain, label in [(in_domain, "in_domain"), (ood_domain, "ood")]:
        _, _, test_texts = load_nonoverlapping_splits(
            tokenizer, DEFAULT_DATASET, domain,
            n_calibration=0, n_train=0, n_test=n_eval, seed=999,
        )
        loss = compute_loss(model, tokenizer, test_texts, max_seq_len)
        results[f"{label}_loss"] = loss
        results[f"{label}_domain"] = domain
        print(f"  {label} ({domain}) loss = {loss:.4f}")

    if not no_judge:
        try:
            from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval

            for domain, label in [(in_domain, "in_domain"), (ood_domain, "ood")]:
                ds = load_dataset(DEFAULT_DATASET, domain, split="train").shuffle(seed=888)
                n_j = min(n_judge_samples, len(ds))
                questions = [str(ds[i]["question"]) for i in range(n_j)]
                answers = [str(ds[i]["answer"]) for i in range(n_j)]

                evaluator = OneClickLLMJudgeScopingEval(
                    n_samples=n_j,
                    train_domain=in_domain,
                    attack_domain=ood_domain,
                )
                scores, _ = evaluator.evaluate(
                    model, tokenizer,
                    {domain: questions},
                    domain_answers={domain: answers},
                )
                for k, v in scores.items():
                    results[f"{label}/{k}"] = v
                print(f"  {label} ({domain}) judge: {json.dumps({k: f'{v:.3f}' for k, v in scores.items()})}")
        except Exception as e:
            print(f"  Judge evaluation failed: {e}")
            results["judge_error"] = str(e)

    return results


@click.command()
@click.option("--artifact-dir", type=click.Path(path_type=Path), default=DEFAULT_ARTIFACT_DIR,
              help="Root artifact directory (shared with calibration sweep)")
@click.option("--method", required=True, help="Pruning method (wanda, taylor, gradient, random)")
@click.option("--model", "model_id", required=True, help="HuggingFace model ID")
@click.option("--in-domain", required=True, type=click.Choice(DOMAINS),
              help="In-domain (retain) domain")
@click.option("--ood-domain", default=None, type=click.Choice(DOMAINS),
              help="Single OOD domain to elicit on (default: all others)")
@click.option("--sparsity", required=True, type=float, help="Sparsity level of the recovered model")
@click.option("--n-train", default=512, help="OOD training samples for elicitation")
@click.option("--n-eval", default=200, help="Eval samples per domain")
@click.option("--max-seq-len", default=1024, type=int)
@click.option("--no-judge", is_flag=True, help="Skip LLM judge evaluation")
@click.option("--n-judge-samples", default=50, type=int)
@click.option(
    "--sft-overrides", default=None,
    help="JSON string of SFTConfig overrides (e.g. '{\"learning_rate\": 3e-5}')",
)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None,
              help="Output dir for results (default: {artifact_dir}/elicitation/...)")
@click.option("--save-model", is_flag=True, help="Save elicited model checkpoints")
@click.option("--wandb-project", default="sae-scoping-elicitation")
@click.option("--device", default=None, help="CUDA device (auto-detected)")
@click.option("--launch", is_flag=True, help="Launch full grid via scheduler")
@click.option("--gpus", default=None, help="Comma-separated GPU IDs (for --launch)")
@click.option("--methods", default=None, help="Filter methods for --launch (comma-separated)")
@click.option("--models", default=None, help="Filter models for --launch (comma-separated)")
@click.option("--domains", default=None, help="Filter in-domains for --launch (comma-separated)")
@click.option("--dry-run", is_flag=True, help="Print jobs without running (for --launch)")
def main(
    artifact_dir, method, model_id, in_domain, ood_domain, sparsity,
    n_train, n_eval, max_seq_len,
    no_judge, n_judge_samples, sft_overrides,
    output_dir, save_model, wandb_project,
    device, launch, gpus, methods, models, domains, dry_run,
):
    if launch:
        _launch_grid(
            artifact_dir=artifact_dir, gpus=gpus, dry_run=dry_run,
            no_judge=no_judge, wandb_project=wandb_project,
            sft_overrides=sft_overrides,
            filter_methods=methods, filter_models=models, filter_domains=domains,
        )
        return

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    ood_domains = [ood_domain] if ood_domain else [d for d in DOMAINS if d != in_domain]

    slug = _model_slug(model_id)
    sp_dir = _sparsity_dir(artifact_dir, model_id, in_domain, method, sparsity)

    if output_dir is None:
        output_dir = artifact_dir / "elicitation" / slug / in_domain / method / f"sp_{sparsity:.2f}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"  Elicitation Sweep")
    print(f"  Method: {method} | Model: {model_id} | Sparsity: {sparsity:.0%}")
    print(f"  In-domain: {in_domain} | OOD targets: {ood_domains}")
    print(f"  Artifacts: {sp_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 70}")

    overrides = json.loads(sft_overrides) if sft_overrides else None

    wandb.init(
        project=wandb_project,
        name=f"elicit/{method}/{slug}/{in_domain}/sp{sparsity}",
        config=dict(
            method=method, model=model_id, in_domain=in_domain,
            sparsity=sparsity, ood_domains=ood_domains,
            sft_overrides=overrides, n_train=n_train,
        ),
    )

    all_results = []
    for ood in ood_domains:
        print(f"\n{'=' * 70}")
        print(f"  Eliciting: {in_domain} -> {ood} ({method} @ {sparsity:.0%})")
        print(f"{'=' * 70}")

        model, tokenizer, masks = _load_recovered_model_and_masks(
            artifact_dir, model_id, in_domain, method, sparsity, device,
        )

        ood_output = output_dir / ood
        elicit_one(
            model, tokenizer, masks, model_id, ood,
            n_train=n_train, sft_overrides=overrides, output_dir=ood_output,
        )

        result = evaluate_elicited(
            model, tokenizer, in_domain, ood,
            n_eval=n_eval, max_seq_len=max_seq_len,
            no_judge=no_judge, n_judge_samples=n_judge_samples,
        )
        result.update(dict(
            method=method, model=model_id, sparsity=sparsity,
            in_domain=in_domain, ood_domain=ood,
        ))

        wandb.log({f"elicit/{ood}/{k}": v for k, v in result.items() if isinstance(v, (int, float))})
        all_results.append(result)

        if save_model:
            _save_elicited_model(model, tokenizer, ood_output / "elicited_model")

        del model, tokenizer, masks
        gc.collect()
        torch.cuda.empty_cache()

    results_path = output_dir / "elicitation_results.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")

    _print_summary(all_results)
    wandb.finish()


def _print_summary(results: list[dict]) -> None:
    print(f"\n{'=' * 70}")
    print(f"  Elicitation Summary")
    print(f"{'=' * 70}")
    print(f"{'OOD Domain':<15} {'In-domain Loss':>15} {'OOD Loss':>15}")
    print(f"{'-' * 45}")
    for r in results:
        print(
            f"{r['ood_domain']:<15} "
            f"{r.get('in_domain_loss', float('nan')):>15.4f} "
            f"{r.get('ood_loss', float('nan')):>15.4f}"
        )
    print()


def _launch_grid(
    artifact_dir: Path,
    gpus: str | None,
    dry_run: bool,
    no_judge: bool,
    wandb_project: str,
    sft_overrides: str | None = None,
    filter_methods: str | None = None,
    filter_models: str | None = None,
    filter_domains: str | None = None,
) -> None:
    """Read manifest from calibration sweep and launch elicitation jobs."""
    from sae_scoping.infrastructure.scheduler import JobSpec, run_jobs

    if gpus is None:
        raise click.UsageError("--gpus required with --launch")

    gpu_ids = [int(g.strip()) for g in gpus.split(",")]

    method_allow = set(m.strip() for m in filter_methods.split(",")) if filter_methods else None
    model_allow = set(m.strip() for m in filter_models.split(",")) if filter_models else None
    domain_allow = set(d.strip() for d in filter_domains.split(",")) if filter_domains else None

    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest.json found at {manifest_path}")
        print("Run calibration_and_recovery_sweep.py first.")
        return

    manifest = SweepManifest.load(manifest_path)
    print(f"Loaded manifest: {len(manifest.entries)} entries")

    jobs: list[JobSpec] = []
    n_skipped = 0
    for entry in manifest.entries:
        if method_allow and entry.method not in method_allow:
            n_skipped += 1
            continue
        if model_allow and entry.model_id not in model_allow:
            n_skipped += 1
            continue
        if domain_allow and entry.domain not in domain_allow:
            n_skipped += 1
            continue

        slug = _model_slug(entry.model_id)
        for sr in entry.results:
            if sr.recovered_model_dir is None:
                continue
            ood_domains = [d for d in DOMAINS if d != entry.domain]
            for ood in ood_domains:
                cmd = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--artifact-dir", str(artifact_dir.resolve()),
                    "--method", entry.method,
                    "--model", entry.model_id,
                    "--in-domain", entry.domain,
                    "--ood-domain", ood,
                    "--sparsity", str(sr.sparsity),
                    "--device", "cuda:0",
                    "--wandb-project", wandb_project,
                ]
                if no_judge:
                    cmd.append("--no-judge")
                if sft_overrides:
                    cmd += ["--sft-overrides", sft_overrides]
                jobs.append(JobSpec(
                    command=cmd,
                    n_gpus=1,
                    name=f"elicit/{entry.method}/{slug}/{entry.domain}>>{ood}/sp{sr.sparsity}",
                ))

    if n_skipped:
        print(f"Filtered out {n_skipped} entries")
    print(f"Launching {len(jobs)} elicitation jobs across {len(gpu_ids)} GPUs")

    log_dir = artifact_dir / "elicitation" / "logs"
    run_jobs(
        jobs, gpu_ids=gpu_ids, n_cpu_workers=0,
        log_dir=log_dir, dry_run=dry_run,
    )


if __name__ == "__main__":
    main()
