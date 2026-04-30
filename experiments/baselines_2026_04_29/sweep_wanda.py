"""Run Wanda pruning sweep on a model and report loss vs sparsity.

Per-run artifacts are written under
    $artifacts_root/outputs/{run_id}/
        metadata.json                # run-level config + git sha + start time
        step_NNN/
            step_metadata.json       # sparsity, loss, zeros counts
            judgements.jsonl         # streamed by JsonlSink (--enable-llm-judge)
            inference.jsonl          # streamed by JsonlSink (--enable-llm-judge)
            scores.json              # final llm_judge/{domain}/{scope}/... dict

For the crash/durability/thread-safety guarantees of the streaming JSONL
files, see `JsonlSink` in sae_scoping.evaluation.utils.
"""

import json
import os
from pathlib import Path
from typing import Optional

import click
import torch

from sae_scoping.datasets.qa_datasets import format_as_sft_text, load_qa_dataset
from sae_scoping.evaluation.loss import compute_loss, count_zeros
from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval
from sae_scoping.evaluation.utils import JsonlSink
from sae_scoping.training.saliency.validators import MaskSubsetValidator
from sae_scoping.training.saliency.wanda import (
    apply_masks_to_model,
    compute_wanda_masks,
    compute_wanda_saliency,
)
from sae_scoping.utils.artifacts import (
    build_run_metadata,
    make_run_dir,
    make_run_id,
    make_step_dir,
    resolve_artifacts_root,
)
from sae_scoping.utils.cache import cache_path, load_or_compute_safetensors
from sae_scoping.utils.click_utils import load_yaml_config, parse_comma_separated_floats
from sae_scoping.utils.model_loading import load_model_and_tokenizer


def _load_judge_domains(
    dataset_name: str,
    domains: list[str],
    split: str,
    n: int,
    seed: int = 42,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Load `n` (question, answer) rows per domain from `split`."""
    domain_questions: dict[str, list[str]] = {}
    domain_answers: dict[str, list[str]] = {}
    for d in domains:
        ds = load_qa_dataset(dataset_name, d, split=split, n=n, seed=seed)
        domain_questions[d] = list(ds["question"])
        domain_answers[d] = list(ds["answer"])
    return domain_questions, domain_answers


@click.command()
@click.option(
    "--config",
    is_eager=True,
    expose_value=False,
    callback=load_yaml_config,
    type=click.Path(exists=True),
    help="YAML config file (CLI flags override).",
)
@click.option("--model-id", default="google/gemma-3-4b-it", show_default=True, help="HuggingFace model ID.")
@click.option("--dataset-name", default="4gate/StemQAMixture", show_default=True)
@click.option("--dataset-subset", default="biology", show_default=True)
@click.option("--n-calibration", default=128, show_default=True, help="Calibration samples.")
@click.option("--n-eval", default=64, show_default=True, help="Evaluation samples (separate from calibration).")
@click.option("--max-seq-len", default=2048, show_default=True)
@click.option("--batch-size", default=1, show_default=True, help="Batch size for calibration and eval.")
@click.option(
    "--nn-linear-sparsity",
    "-s",
    default=None,
    help="Per-row sparsity within nn.Linear layers only (embeddings/head untouched). Comma-separated for sweep (e.g. -s 0.2,0.4,0.6).",
)
@click.option(
    "--cache-dir",
    default=str(Path(os.environ.get("SAESCOPING_ARTIFACTS_LOCATION", ".")) / "cache"),
    show_default=True,
    help="Directory for cached saliency maps.",
)
@click.option("--no-cache", is_flag=True, help="Recompute saliency even if cached.")
@click.option("--low-memory", is_flag=True, help="Skip mask monotonicity validation to save CPU memory.")
@click.option("--device", default="cuda:0", show_default=True)
# ── Artifacts / per-run logging ─────────────────────────────────────────────
@click.option(
    "--artifacts-dir",
    default=None,
    help="Root for run artifacts. Defaults to $SAESCOPING_ARTIFACTS_LOCATION (or '.').",
)
# ── LLM-judge wiring (off by default; needs OPENAI_API_KEY when on) ─────────
@click.option("--enable-llm-judge", is_flag=True, help="Run LLM-judge evaluation per sparsity step.")
@click.option(
    "--judge-domains",
    default=None,
    help="Comma-separated dataset subsets to judge (e.g. 'biology,math,chemistry'). Defaults to --dataset-subset only.",
)
@click.option("--judge-n-samples", default=50, show_default=True, type=int)
@click.option("--judge-model", default="gpt-4.1-nano", show_default=True)
@click.option("--judge-split", default="validation", show_default=True, help="Dataset split to draw judge questions from.")
@click.pass_context
def main(
    ctx,
    model_id,
    dataset_name,
    dataset_subset,
    n_calibration,
    n_eval,
    max_seq_len,
    batch_size,
    nn_linear_sparsity,
    cache_dir,
    no_cache,
    low_memory,
    device,
    artifacts_dir,
    enable_llm_judge,
    judge_domains,
    judge_n_samples,
    judge_model,
    judge_split,
):
    """Run Wanda pruning sweep: compute saliency once, then evaluate at each sparsity level from low to high."""
    sparsities = parse_comma_separated_floats(nn_linear_sparsity, default=[0.5])
    print(f"Sweep nn.Linear sparsities: {[f'{s:.1%}' for s in sparsities]}")

    # ── Run-level artifacts setup ───────────────────────────────────────────
    artifacts_root = resolve_artifacts_root(artifacts_dir)
    run_id = make_run_id()
    run_dir = make_run_dir(artifacts_root, run_id)
    print(f"[artifacts] run_dir: {run_dir}")

    judge_domains_list: list[str] = (
        [d.strip() for d in judge_domains.split(",")] if judge_domains else [dataset_subset]
    )

    run_metadata = build_run_metadata(
        ctx.params,
        run_id=run_id,
        script=Path(__file__),
        sparsities_parsed=sparsities,
        artifacts_dir_resolved=str(artifacts_root),
        judge_domains_parsed=judge_domains_list if enable_llm_judge else None,
    )
    (run_dir / "metadata.json").write_text(json.dumps(run_metadata, indent=2, default=str), encoding="utf-8")

    print(f"Loading tokenizer and model: {model_id}")
    model, tokenizer = load_model_and_tokenizer(model_id, device=device)

    print(f"Loading dataset: {dataset_name}/{dataset_subset}")
    n_total = n_calibration + n_eval
    ds = load_qa_dataset(dataset_name, dataset_subset, n=n_total, seed=42)
    all_texts = format_as_sft_text(ds, tokenizer)
    calib_texts = all_texts[:n_calibration]
    eval_texts = all_texts[n_calibration:]

    # ── Pre-load LLM-judge data (once; reused per sparsity step) ─────────────
    evaluator: Optional[OneClickLLMJudgeScopingEval] = None
    judge_questions: Optional[dict[str, list[str]]] = None
    judge_answers: Optional[dict[str, list[str]]] = None
    if enable_llm_judge:
        print(f"[llm-judge] Loading {judge_split} split for domains: {judge_domains_list}")
        judge_questions, judge_answers = _load_judge_domains(
            dataset_name=dataset_name,
            domains=judge_domains_list,
            split=judge_split,
            n=judge_n_samples,
        )
        evaluator = OneClickLLMJudgeScopingEval(
            n_samples=judge_n_samples,
            judge_model=judge_model,
            train_domain=dataset_subset,
        )

    print("\n=== Baseline (pre-pruning) ===")
    baseline_loss = compute_loss(model, tokenizer, eval_texts, max_seq_len=max_seq_len, batch_size=batch_size)
    zeros_before, total_params = count_zeros(model)
    print(f"  Loss:            {baseline_loss:.4f}")
    print(f"  Model sparsity:  {zeros_before}/{total_params} ({zeros_before / total_params:.2%})")

    saliency_file = cache_path(Path(cache_dir), model_id, dataset_subset, "wanda_saliency.safetensors")
    saliency_map = load_or_compute_safetensors(
        path=saliency_file,
        compute_fn=lambda: compute_wanda_saliency(model, tokenizer, calib_texts, max_seq_len=max_seq_len, batch_size=batch_size),
        no_cache=no_cache,
        label="Wanda saliency",
    )

    linear_total = sum(t.numel() for t in saliency_map.values())

    validator = MaskSubsetValidator(enabled=not low_memory)
    results = []
    for i, sparsity in enumerate(sparsities):
        masks = compute_wanda_masks(saliency_map, sparsity)
        validator.validate_and_update(masks)
        apply_masks_to_model(model, masks)

        pruned_loss = compute_loss(model, tokenizer, eval_texts, max_seq_len=max_seq_len, batch_size=batch_size)
        zeros_after, _ = count_zeros(model)
        linear_zeros = sum(int((~m).sum().item()) for m in masks.values())
        delta = pruned_loss - baseline_loss
        results.append((sparsity, pruned_loss, delta, zeros_after, linear_zeros))

        # ── Per-step artifacts ──────────────────────────────────────────────
        # TODO(adrianoh) have some way of modularizing this out
        step_dir = make_step_dir(run_dir, i)
        step_metadata = {
            "step_idx": i,
            "nn_linear_sparsity": sparsity,
            "loss": pruned_loss,
            "loss_delta_vs_baseline": delta,
            "linear_zeros": linear_zeros,
            "linear_total": linear_total,
            "linear_sparsity": linear_zeros / linear_total,
            "model_zeros": zeros_after,
            "model_total": total_params,
            "model_sparsity": zeros_after / total_params,
            "baseline_loss": baseline_loss,
        }
        (step_dir / "step_metadata.json").write_text(
            json.dumps(step_metadata, indent=2, default=str), encoding="utf-8"
        )

        print(f"\n=== nn.Linear sparsity {sparsity:.1%} (step {i}) ===")
        print(f"  Loss:                 {pruned_loss:.4f} (delta: {delta:+.4f})")
        print(f"  nn.Linear sparsity:   {linear_zeros}/{linear_total} ({linear_zeros / linear_total:.2%})")
        print(f"  Whole-model sparsity: {zeros_after}/{total_params} ({zeros_after / total_params:.2%})")
        print(f"  Step dir:             {step_dir}")

        # ── LLM-judge per step ─────────────────────────────────────────────
        if enable_llm_judge:
            # TODO(adrianoh) have some way of modularizing this out
            assert evaluator is not None and judge_questions is not None and judge_answers is not None
            with (
                JsonlSink(step_dir / "judgements.jsonl") as j_sink,
                JsonlSink(step_dir / "inference.jsonl") as i_sink,
            ):
                scores, _df_json = evaluator.evaluate(
                    model,
                    tokenizer,
                    domain_questions=judge_questions,
                    domain_answers=judge_answers,
                    judgement_sink=j_sink,
                    inference_sink=i_sink,
                )
            (step_dir / "scores.json").write_text(
                json.dumps(scores, indent=2, default=str), encoding="utf-8"
            )
            for k, v in sorted(scores.items()):
                print(f"  llm_judge: {k:<60} {v:.3f}")

    print(f"\n{'=' * 70}")
    print(f"Summary: {model_id} on {dataset_subset}  (run_id={run_id})")
    print(f"{'nn.Linear %':>12} {'Loss':>10} {'Delta':>10} {'Linear %':>10} {'Model %':>10}")
    print(f"{'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for sparsity, loss, delta, zeros, lin_zeros in results:
        print(f"{sparsity:>12.1%} {loss:>10.4f} {delta:>+10.4f} {lin_zeros / linear_total:>10.2%} {zeros / total_params:>10.2%}")
    print(f"\nArtifacts: {run_dir}")


if __name__ == "__main__":
    main()
