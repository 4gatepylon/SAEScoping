#!/usr/bin/env python3
"""Evaluate attribution-pruned models: loss + LLM judge across StemQA domains.

Loads a pre-computed attribution_scores.pt, sweeps over sparsity levels,
and for each one reconstructs the pruned model on the fly (no saved
checkpoints needed).  Runs:

  1. Cross-entropy loss on each eval domain (via compute_loss).
  2. LLM judge (relevance, fluency, ground_truth_similarity) on each eval
     domain (via OneClickLLMJudgeScopingEval).

Designed for StemQAMixture (biology/chemistry/math/physics).  The
``--train_domain`` flag tells the judge which domain is "in_scope" and
which are "out_of_scope".

Example
-------
    CUDA_VISIBLE_DEVICES=0 python eval_attribution.py \\
        --model_name google/gemma-2-9b-it \\
        --attribution_scores_path pruned_models/gemma2_9b_it_biology/attribution_scores.pt \\
        --train_domain biology \\
        --sparsity_levels 0.1 0.3 0.5 0.63 0.8 \\
        --dtype bfloat16 \\
        --output_dir results/gemma2_9b_it_biology_eval

Smoke test (tiny, no real GPU pressure)::

    CUDA_VISIBLE_DEVICES=0 python eval_attribution.py \\
        --model_name NousResearch/Llama-3.2-1B \\
        --attribution_scores_path /tmp/smoke_attr/attribution_scores.pt \\
        --train_domain biology \\
        --sparsity_levels 0.3 \\
        --eval_domains biology \\
        --judge_n_samples 2 --loss_n_samples 4 --loss_batch_size 2 \\
        --output_dir /tmp/smoke_eval_attr

TODO(hadriano) integrate wandb.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

import importlib.util as _ilu


def _load_sibling(name, filename):
    spec = _ilu.spec_from_file_location(
        name, os.path.join(os.path.dirname(__file__), filename)
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


shared = _load_sibling("baselines_narrow_shared", "shared.py")
load_and_prune_model = shared.load_and_prune_model

_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from sae_scoping.datasets.qa_datasets import format_as_sft_text, load_qa_dataset
from sae_scoping.evaluation.loss import compute_loss, count_zeros
from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval
from sae_scoping.evaluation.utils import JsonlSink


def _load_judge_domains(
    dataset_name: str,
    domains: list[str],
    split: str,
    n: int,
    seed: int = 42,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Load ``n`` (question, answer) rows per domain from ``split``."""
    domain_questions: dict[str, list[str]] = {}
    domain_answers: dict[str, list[str]] = {}
    for d in domains:
        ds = load_qa_dataset(dataset_name, d, split=split, n=n, seed=seed)
        domain_questions[d] = list(ds["question"])
        domain_answers[d] = list(ds["answer"])
    return domain_questions, domain_answers


def _load_loss_texts(
    model_name: str,
    dataset_name: str,
    domains: list[str],
    n: int,
    split: str = "validation",
    seed: int = 42,
) -> dict[str, list[str]]:
    """Load chat-templated SFT texts per domain for loss evaluation."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts_by_domain: dict[str, list[str]] = {}
    for d in domains:
        ds = load_qa_dataset(dataset_name, d, split=split, n=n, seed=seed)
        texts_by_domain[d] = format_as_sft_text(ds, tokenizer)
    return texts_by_domain


def _eval_one_sparsity(
    *,
    model_name: str,
    attribution_scores: dict,
    sparsity: float,
    dtype: str,
    loss_texts_by_domain: dict[str, list[str]],
    loss_batch_size: int,
    evaluator: OneClickLLMJudgeScopingEval,
    domain_questions: dict[str, list[str]],
    domain_answers: dict[str, list[str]],
    output_dir: Path,
) -> dict:
    """Prune, measure loss + LLM judge, save logs. Returns stats dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, pruned_neurons, neurons_per_layer = load_and_prune_model(
        model_name, attribution_scores, sparsity, dtype=dtype,
    )

    zeros, total = count_zeros(model)
    print(f"  Model sparsity: {zeros}/{total} ({zeros / total:.2%})")

    total_neurons = sum(
        layer.mlp.gate_proj.out_features
        for layer in shared.text_decoder(model).layers
    )

    loss_by_domain: dict[str, float] = {}
    for domain, texts in loss_texts_by_domain.items():
        loss = compute_loss(
            model, tokenizer, texts,
            max_seq_len=shared.NARROW_MAX_LENGTH,
            batch_size=loss_batch_size,
        )
        loss_by_domain[domain] = loss
        print(f"  Loss ({domain}): {loss:.4f}")

    judge_scores: dict[str, float] = {}
    with (
        JsonlSink(output_dir / "judgements.jsonl") as judgement_sink,
        JsonlSink(output_dir / "inference.jsonl") as inference_sink,
    ):
        judge_scores, df_json = evaluator.evaluate(
            model, tokenizer,
            domain_questions=domain_questions,
            domain_answers=domain_answers,
            judgement_sink=judgement_sink,
            inference_sink=inference_sink,
        )

    for k, v in sorted(judge_scores.items()):
        print(f"  {k}: {v:.4f}")

    del model
    torch.cuda.empty_cache()

    stats = {
        "base_model": model_name,
        "pruning_method": "attribution",
        "neuron_sparsity": sparsity,
        "total_neurons": total_neurons,
        "neurons_pruned": len(pruned_neurons),
        "neurons_per_layer": neurons_per_layer,
        "model_zeros": zeros,
        "model_total_params": total,
        "model_sparsity": zeros / total,
        "loss_by_domain": loss_by_domain,
        "llm_judge_scores": judge_scores,
        "pruned_neurons": pruned_neurons,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    (output_dir / "judge_df.json").write_text(df_json)
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate attribution-pruned models (loss + LLM judge)"
    )

    g = p.add_argument_group("model")
    g.add_argument("--model_name", type=str, required=True)
    g.add_argument(
        "--attribution_scores_path", type=str, required=True,
        help="Path to attribution_scores.pt (from create_attribution_pruned_models.py).",
    )
    g.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"],
    )

    g = p.add_argument_group("pruning sweep")
    g.add_argument(
        "--sparsity_levels", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.63, 0.8],
    )

    g = p.add_argument_group("evaluation domains")
    g.add_argument(
        "--train_domain", type=str, required=True,
        choices=list(shared.STEMQA_CONFIGS),
        help="The domain the model was pruned on (marked in_scope for judge).",
    )
    g.add_argument(
        "--eval_domains", type=str, nargs="+", default=list(shared.STEMQA_CONFIGS),
        help="Domains to evaluate on (default: all StemQA configs).",
    )
    g.add_argument(
        "--eval_split", type=str, default="validation",
        help="HF split for evaluation data.",
    )

    g = p.add_argument_group("LLM judge")
    g.add_argument("--judge_model", type=str, default="gpt-4.1-nano")
    g.add_argument(
        "--judge_n_samples", type=int, default=50,
        help="Questions per domain for the LLM judge.",
    )

    g = p.add_argument_group("loss")
    g.add_argument(
        "--loss_n_samples", type=int, default=200,
        help="Samples per domain for loss computation.",
    )
    g.add_argument("--loss_batch_size", type=int, default=2)

    g = p.add_argument_group("output")
    g.add_argument("--output_dir", type=str, required=True)

    return p.parse_args()


def main(args=None) -> None:
    if args is None:
        args = parse_args()
    shared.validate_args(args, sparsity_attrs=("sparsity_levels",))

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 80}")
    print("Attribution-Pruning Evaluation (loss + LLM judge)")
    print(f"{'=' * 80}")
    print(f"Model:             {args.model_name}")
    print(f"Attribution scores:{args.attribution_scores_path}")
    print(f"Sparsity levels:   {args.sparsity_levels}")
    print(f"Train domain:      {args.train_domain} (in_scope)")
    print(f"Eval domains:      {args.eval_domains}")
    print(f"Eval split:        {args.eval_split}")
    print(f"Judge model:       {args.judge_model}")
    print(f"Judge n_samples:   {args.judge_n_samples}")
    print(f"Loss n_samples:    {args.loss_n_samples}")
    print(f"Dtype:             {args.dtype}")
    print(f"Output:            {args.output_dir}")
    print(f"{'=' * 80}\n")

    attribution_scores = torch.load(
        args.attribution_scores_path, map_location="cpu", weights_only=True,
    )
    print(f"Loaded attribution scores: {len(attribution_scores)} layers")

    print(f"\nLoading judge data ({args.eval_split} split, {args.judge_n_samples} samples/domain)...")
    domain_questions, domain_answers = _load_judge_domains(
        shared.STEMQA_DATASET, args.eval_domains,
        split=args.eval_split, n=args.judge_n_samples,
    )

    print(f"Loading loss texts ({args.eval_split} split, {args.loss_n_samples} samples/domain)...")
    loss_texts_by_domain = _load_loss_texts(
        args.model_name, shared.STEMQA_DATASET, args.eval_domains,
        n=args.loss_n_samples, split=args.eval_split,
    )

    evaluator = OneClickLLMJudgeScopingEval(
        n_samples=args.judge_n_samples,
        judge_model=args.judge_model,
        train_domain=args.train_domain,
    )

    all_stats = []
    for sparsity in args.sparsity_levels:
        print(f"\n{'=' * 80}")
        print(f"Sparsity {sparsity:.2%}")
        print(f"{'=' * 80}")

        sparsity_dir = output_root / f"sparsity_{sparsity}"
        stats = _eval_one_sparsity(
            model_name=args.model_name,
            attribution_scores=attribution_scores,
            sparsity=sparsity,
            dtype=args.dtype,
            loss_texts_by_domain=loss_texts_by_domain,
            loss_batch_size=args.loss_batch_size,
            evaluator=evaluator,
            domain_questions=domain_questions,
            domain_answers=domain_answers,
            output_dir=sparsity_dir,
        )
        all_stats.append(stats)

    summary = {
        "model_name": args.model_name,
        "train_domain": args.train_domain,
        "eval_domains": args.eval_domains,
        "eval_split": args.eval_split,
        "judge_model": args.judge_model,
        "judge_n_samples": args.judge_n_samples,
        "loss_n_samples": args.loss_n_samples,
        "dtype": args.dtype,
        "sparsity_results": all_stats,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'Sparsity':>10}", end="")
    for d in args.eval_domains:
        print(f"  {d + ' loss':>16}", end="")
    print()
    print(f"{'-' * 10}", end="")
    for _ in args.eval_domains:
        print(f"  {'-' * 16}", end="")
    print()
    for s in all_stats:
        print(f"{s['neuron_sparsity']:>10.2%}", end="")
        for d in args.eval_domains:
            print(f"  {s['loss_by_domain'].get(d, float('nan')):>16.4f}", end="")
        print()

    print(f"\nJudge scores per sparsity:")
    for s in all_stats:
        print(f"  sparsity={s['neuron_sparsity']:.2%}:")
        for k, v in sorted(s["llm_judge_scores"].items()):
            print(f"    {k}: {v:.4f}")

    print(f"\nAll artifacts: {args.output_dir}")


if __name__ == "__main__":
    main()
