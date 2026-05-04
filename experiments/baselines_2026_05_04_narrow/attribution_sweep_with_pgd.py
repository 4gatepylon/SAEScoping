#!/usr/bin/env python3
"""Attribution-prune at multiple sparsities, then PGD-recover each one.

Computes attribution scores once, then for each sparsity level:
  1. Loads a fresh model copy
  2. Prunes neurons by attribution
  3. Builds PGD masks via build_pgd_masks_from_model (zero = pruned)
  4. Runs PGDSFTTrainer for recovery (with native Trainer eval)
  5. Saves model + stats

Data splits are non-overlapping:
  [0, num_attribution_samples)                         → attribution calibration
  [num_attribution_samples, ... + pgd_train_samples)   → PGD training
  [... + pgd_train_samples, ... + pgd_eval_samples)    → eval

TODO(hadriano): add LLM judge evaluation per sparsity (wire OneClickLLMJudgeScopingEval)
TODO(hadriano): add W&B logging (follow sweep_wanda.py's define_metric pattern)
TODO(hadriano): migrate CLI to pydantic-yaml SweepConfig (extend or subclass the existing schema)
TODO(hadriano): read through this script end-to-end and verify correctness against upstream narrow

Smoke tests (single GPU, small steps):

    # gemma-2-2b-it
    export HF_HOME=/path/to/hf_cache
    python attribution_sweep_with_pgd.py \\
      --model_name google/gemma-2-2b-it \\
      --dataset_name 4gate/StemQAMixture --dataset_config biology \\
      --num_attribution_samples 16 --attribution_batch_size 2 \\
      --sparsity_levels 0.3 \\
      --pgd_train_samples 32 --pgd_eval_samples 16 \\
      --pgd_max_steps 10 --pgd_batch_size 1 --pgd_gradient_accumulation_steps 1 \\
      --pgd_eval_every_steps 5 --pgd_logging_steps 1 --pgd_warmup_ratio 0.0 \\
      --output_base_dir /tmp/smoke_aspgd_g2_2b

    # gemma-3-4b-it
    python attribution_sweep_with_pgd.py \\
      --model_name google/gemma-3-4b-it \\
      --dataset_name 4gate/StemQAMixture --dataset_config biology \\
      --num_attribution_samples 16 --attribution_batch_size 2 \\
      --sparsity_levels 0.3 \\
      --pgd_train_samples 32 --pgd_eval_samples 16 \\
      --pgd_max_steps 10 --pgd_batch_size 1 --pgd_gradient_accumulation_steps 1 \\
      --pgd_eval_every_steps 5 --pgd_logging_steps 1 --pgd_warmup_ratio 0.0 \\
      --output_base_dir /tmp/smoke_aspgd_g3_4b
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

# ---------------------------------------------------------------------------
# Sibling imports via importlib (no PYTHONPATH dependency)
# ---------------------------------------------------------------------------
import importlib.util as _ilu

def _load_sibling(name, filename):
    spec = _ilu.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), filename))
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

shared = _load_sibling("baselines_narrow_shared", "shared.py")
_cap = _load_sibling("create_attribution_pruned_models", "create_attribution_pruned_models.py")
compute_attribution_scores = _cap.compute_attribution_scores
prune_by_attribution = _cap.prune_by_attribution
prepare_dataloader = _cap.prepare_dataloader

# ---------------------------------------------------------------------------
# Project imports via importlib (no PYTHONPATH dependency)
# ---------------------------------------------------------------------------
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from sae_scoping.evaluation.loss import compute_loss, count_zeros
from sae_scoping.training.pgd_trainer import PGDSFTTrainer, build_pgd_masks_from_model


def _run_one_sparsity(
    *,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    attribution_scores: dict,
    sparsity: float,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    """Prune → build PGD masks → recover → save. Returns stats dict."""
    pruned_neurons, neurons_per_layer = prune_by_attribution(model, attribution_scores, sparsity)
    masks = build_pgd_masks_from_model(model)
    print(f"  PGD masks: {len(masks)} parameters with at least one zero")

    zeros_pre, total = count_zeros(model)
    print(f"  Pre-recovery sparsity: {zeros_pre}/{total} ({zeros_pre/total:.2%})")

    use_cuda = torch.cuda.is_available()
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=str(output_dir / "trl_output"),
        learning_rate=args.pgd_learning_rate,
        num_train_epochs=1,
        max_steps=args.pgd_max_steps,
        per_device_train_batch_size=args.pgd_batch_size,
        per_device_eval_batch_size=args.pgd_batch_size,
        gradient_accumulation_steps=args.pgd_gradient_accumulation_steps,
        warmup_ratio=args.pgd_warmup_ratio,
        logging_steps=args.pgd_logging_steps,
        eval_strategy="steps",
        eval_steps=args.pgd_eval_every_steps,
        max_length=shared.NARROW_MAX_LENGTH,
        bf16=use_cuda,
        report_to="none",
        save_strategy="no",
        optim="adamw_torch_fused" if use_cuda else "adamw_torch",
        gradient_checkpointing=args.pgd_gradient_checkpointing,
        use_cpu=not use_cuda,
    )

    # TODO(hadriano): wire RecoveryEvalCallback for richer per-step metrics (loss delta, sparsity tracking)
    trainer = PGDSFTTrainer(
        masks=masks,
        validate_sparsity=True,
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    print(f"  Starting PGD recovery ({args.pgd_max_steps} steps)...")
    trainer.train()

    eval_results = trainer.evaluate()
    post_loss = eval_results.get("eval_loss", float("nan"))
    zeros_post, _ = count_zeros(model)

    model.save_pretrained(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))

    stats = {
        "sparsity": sparsity,
        "neurons_pruned": len(pruned_neurons),
        "neurons_per_layer": neurons_per_layer,
        "pgd_masks_count": len(masks),
        "post_recovery_loss": post_loss,
        "pre_recovery_zeros": zeros_pre,
        "post_recovery_zeros": zeros_post,
        "total_params": total,
        "pgd_max_steps": args.pgd_max_steps,
        "pgd_learning_rate": args.pgd_learning_rate,
        "trainer_log_history": trainer.state.log_history,
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(f"  Post-recovery loss: {post_loss:.4f}  sparsity: {zeros_post}/{total} ({zeros_post/total:.2%})")
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attribution prune + PGD recovery sweep")

    g = p.add_argument_group("model / dataset")
    g.add_argument("--model_name", type=str, default="NousResearch/Llama-3.2-1B")
    g.add_argument("--dataset_name", type=str, default=shared.CODEPARROT_DATASET, choices=list(shared.SUPPORTED_DATASETS))
    g.add_argument("--dataset_config", type=str, default=None,
                   help=f"Required for {shared.STEMQA_DATASET} (one of {list(shared.STEMQA_CONFIGS)}).")

    g = p.add_argument_group("attribution pruning")
    g.add_argument("--sparsity_levels", type=float, nargs="+", default=[0.3, 0.63, 0.8])
    g.add_argument("--num_attribution_samples", type=int, default=1024)
    g.add_argument("--attribution_batch_size", type=int, default=8)

    g = p.add_argument_group("PGD recovery")
    g.add_argument("--pgd_train_samples", type=int, default=4000)
    g.add_argument("--pgd_eval_samples", type=int, default=200)
    g.add_argument("--pgd_max_steps", type=int, default=500)
    g.add_argument("--pgd_learning_rate", type=float, default=2e-5)
    g.add_argument("--pgd_batch_size", type=int, default=2)
    g.add_argument("--pgd_gradient_accumulation_steps", type=int, default=4)
    g.add_argument("--pgd_warmup_ratio", type=float, default=0.05)
    g.add_argument("--pgd_eval_every_steps", type=int, default=50)
    g.add_argument("--pgd_logging_steps", type=int, default=10)
    g.add_argument("--pgd_gradient_checkpointing", action="store_true")

    g = p.add_argument_group("output")
    g.add_argument("--output_base_dir", type=str, required=True)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    shared.validate_args(args, sparsity_attrs=("sparsity_levels",))

    os.makedirs(args.output_base_dir, exist_ok=True)
    print(f"{'='*80}")
    print(f"Attribution Prune + PGD Recovery Sweep")
    print(f"{'='*80}")
    print(f"Model:              {args.model_name}")
    print(f"Dataset:            {args.dataset_name}"
          + (f" ({args.dataset_config})" if args.dataset_config else ""))
    print(f"Sparsity levels:    {args.sparsity_levels}")
    print(f"Attribution samples: {args.num_attribution_samples}")
    print(f"PGD train samples:  {args.pgd_train_samples}")
    print(f"PGD eval samples:   {args.pgd_eval_samples}")
    print(f"PGD max steps:      {args.pgd_max_steps}")
    print(f"Output:             {args.output_base_dir}")
    print(f"{'='*80}\n")

    # --- Phase 1: compute attribution scores (once) ---

    print("Loading model for attribution...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        **shared.load_model_kwargs(args.model_name),
    )
    dataloader = prepare_dataloader(
        args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        num_samples=args.num_attribution_samples,
        batch_size=args.attribution_batch_size,
        max_length=shared.NARROW_MAX_LENGTH,
    )
    num_batches = args.num_attribution_samples // args.attribution_batch_size
    attribution_scores = compute_attribution_scores(model, dataloader, num_batches)

    attribution_file = os.path.join(args.output_base_dir, "attribution_scores.pt")
    torch.save(attribution_scores, attribution_file)
    print(f"Saved attribution scores to: {attribution_file}")

    del model
    torch.cuda.empty_cache()

    # --- Phase 2: load SFT texts for PGD (non-overlapping with attribution) ---

    train_skip = args.num_attribution_samples
    eval_skip = train_skip + args.pgd_train_samples

    print(f"\nLoading PGD training texts ({args.pgd_train_samples} samples, skip={train_skip})...")
    train_texts = shared.load_pruning_texts(
        args.model_name, args.dataset_name, args.pgd_train_samples,
        skip_samples=train_skip, dataset_config=args.dataset_config,
    )
    print(f"Loading PGD eval texts ({args.pgd_eval_samples} samples, skip={eval_skip})...")
    eval_texts = shared.load_pruning_texts(
        args.model_name, args.dataset_name, args.pgd_eval_samples,
        skip_samples=eval_skip, dataset_config=args.dataset_config,
    )

    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Phase 3: prune + recover per sparsity ---

    all_stats = []
    for sparsity in args.sparsity_levels:
        print(f"\n{'='*80}")
        print(f"Sparsity {sparsity:.2%}")
        print(f"{'='*80}")

        print("Loading fresh model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            **shared.load_model_kwargs(args.model_name),
        )

        sparsity_dir = Path(args.output_base_dir) / f"sparsity_{sparsity}"
        stats = _run_one_sparsity(
            model=model,
            tokenizer=tokenizer,
            attribution_scores=attribution_scores,
            sparsity=sparsity,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=args,
            output_dir=sparsity_dir,
        )
        all_stats.append(stats)

        del model
        torch.cuda.empty_cache()

    # --- Summary ---

    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'Sparsity':>10} {'Post-loss':>10} {'Zeros':>10}")
    print(f"{'-'*10} {'-'*10} {'-'*10}")
    for s in all_stats:
        print(f"{s['sparsity']:>10.2%} {s['post_recovery_loss']:>10.4f} {s['post_recovery_zeros']:>10}")
    print(f"\nArtifacts: {args.output_base_dir}")


if __name__ == "__main__":
    main()
