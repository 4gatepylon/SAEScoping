"""
Full SAE scoping pipeline. Supports Gemma-2-9b-it (--gemma2) and Gemma-3-12b-it (--gemma3, default).

Stages:
  1. RANK:     Compute firing rates on the chosen train domain (or load precomputed)
  2. PRUNE:    Prune SAE neurons with firing rate < threshold
  3. RECOVER:  In-domain SFT on the train domain
  4. ATTACK:   Adversarial SFT on an OOD domain

Supported train domains: biology, chemistry, math, cyber
The remaining domains are automatically used for OOD eval.

Usage:
  # Full pipeline, biology domain, gemma3 (default):
  python script_scoping_pipeline_stemqa.py --train-domain biology --attack-domain chemistry --stage all

  # Same but with gemma2-9b:
  python script_scoping_pipeline_stemqa.py --train-domain biology --attack-domain chemistry --stage all --gemma2

  # Just recovery training (assumes firing rates already computed):
  python script_scoping_pipeline_stemqa.py --train-domain biology --stage recover

  # Just adversarial training (assumes recovery checkpoint exists):
  python script_scoping_pipeline_stemqa.py --train-domain biology --stage attack \
      --attack-domain chemistry --checkpoint outputs_scoping/biology/recover/checkpoint-XXXX
"""

from __future__ import annotations

import gc
import io
import json
from pathlib import Path
import time

import click
import pandas as pd
import torch
import wandb
from itertools import islice
from datasets import Dataset, load_dataset
from safetensors.torch import load_file, save_file
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from trl import SFTConfig
import tqdm
import sys
import os
sys.path.append(os.path.abspath(".."))
from functools import partial
from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae
from sae_scoping.trainers.sae_enhanced.rank import rank_neurons
from sae_scoping.trainers.sae_enhanced.train import train_sae_enhanced_model
from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
from sae_scoping.utils.hooks.sae import SAEWrapper
from sae_scoping.xxx_evaluation.scoping_eval import OneClickLLMJudgeScopingEval
from sae_scoping.xxx_evaluation.trainer_callbacks import LLMJudgeScopingTrainerCallback

# ── Model configs ─────────────────────────────────────────────────────────────
GEMMA3_CONFIG = dict(
    model_name="google/gemma-3-12b-it",
    sae_release="gemma-scope-2-12b-it-res",
    sae_id="layer_41_width_16k_l0_medium",
    hookpoint="model.language_model.layers.41",
    cache_tag="layer_41--width_16k--l0_medium",
)
GEMMA2_CONFIG = dict(
    model_name="google/gemma-2-9b-it",
    sae_release="gemma-scope-9b-pt-res",
    sae_id="layer_35/width_131k/average_l0_94",
    hookpoint="model.layers.35",
    cache_tag="layer_35--width_131k--l0_94",
)
FIRING_RATE_THRESHOLD = 1e-4  # 0.0001

ALL_DOMAINS = ["biology", "chemistry", "math", "cyber"]

# StemQA domains share the same HF dataset; cyber is MCQ from WMDP.
STEMQA_DOMAINS = {"biology", "chemistry", "math"}


# ── Dataset loaders ───────────────────────────────────────────────────────────

def _stream_qa_dataset(
    dataset_name: str,
    config: str,
    split: str,
    n_samples: int,
    tokenizer: PreTrainedTokenizerBase,
    question_col: str = "question",
    answer_col: str = "answer",
    seed: int = 1,
    stream_flag: bool = True,
) -> Dataset:
    """Stream n_samples from a HF dataset, apply chat template, return Dataset with 'text' column."""
    print(f"Streaming {dataset_name} ({config}) for {n_samples} samples... with streaming={stream_flag}")
    stream = load_dataset(dataset_name, config, split=split, streaming=stream_flag)
    if stream_flag:
        stream = stream.shuffle(seed=seed, buffer_size=50)
    rows = []
    print(f"Processing stream and applying chat template... ", n_samples)
    for i, example in tqdm.tqdm(enumerate(stream), total=n_samples):
        if i >= n_samples:
            break
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": str(example[question_col])},
                {"role": "assistant", "content": str(example[answer_col])},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        rows.append({"text": text, "question": str(example[question_col])})
    return Dataset.from_list(rows)


def _load_wmdp_cyber_raw(n_samples: int, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """Load up to n_samples from WMDP-cyber (MCQ), return Dataset with 'text' and 'question' columns."""
    ds = load_dataset("cais/wmdp", "wmdp-cyber", split="test", streaming=False)
    rows = []
    for ex in islice(ds, n_samples):
        question = ex["question"]
        choices = ex["choices"]
        answer_idx = ex["answer"]
        labels = ["A", "B", "C", "D"]
        choices_str = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices))
        text = f"Question: {question}\n{choices_str}"
        chat = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": f"{labels[answer_idx]}. {choices[answer_idx]}"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        rows.append({"text": chat, "question": text})
    return Dataset.from_list(rows)


def load_domain_train_eval(
    domain: str,
    tokenizer: PreTrainedTokenizerBase,
    eval_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Load a domain dataset and split into non-overlapping 80/20 train/eval subsets."""
    if domain in STEMQA_DOMAINS:
        full = _stream_qa_dataset(
            "4gate/StemQAMixture", domain, "train", 50_000, tokenizer, stream_flag=False
        )
    elif domain == "cyber":
        # WMDP-cyber only has a test split (~1987 samples)
        full = _load_wmdp_cyber_raw(n_samples=1_987, tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown domain {domain!r}. Choose from: {ALL_DOMAINS}")

    full = full.shuffle(seed=seed)
    n_eval = int(len(full) * eval_fraction)
    eval_ds = full.select(range(n_eval))
    train_ds = full.select(range(n_eval, len(full)))
    print(f"  {domain} split: {len(train_ds)} train, {len(eval_ds)} eval (no overlap, {eval_fraction:.0%} eval)")
    return train_ds, eval_ds



# ── Pipeline stages ───────────────────────────────────────────────────────────

def stage_rank(
    train_dataset: Dataset,
    n_samples: int,
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
    model: AutoModelForCausalLM,
    device: torch.device,
    cache_dir: Path,
    sae_release: str,
    sae_id: str,
    hookpoint: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 1: Compute firing rates on the train domain split."""
    cache_path = cache_dir / "firing_rates.safetensors"
    if cache_path.exists():
        print(f"Loading cached firing rates from {cache_path}")
        data = load_file(str(cache_path))
        return data["ranking"], data["distribution"]

    n_samples = min(n_samples, len(train_dataset))
    print(f"Computing firing rates on {n_samples} train samples...")
    dataset = train_dataset.select(range(n_samples))

    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device='cpu')
    sae = sae.to(device)

    ranking, distribution = rank_neurons(
        dataset=dataset,
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        T=0,
        hookpoint=hookpoint,
        batch_size=batch_size,
        token_selection="attention_mask",
        return_distribution=True,
    )

    ranking = ranking.detach().cpu()
    distribution = distribution.detach().cpu()

    cache_dir.mkdir(parents=True, exist_ok=True)
    save_file({"ranking": ranking, "distribution": distribution}, str(cache_path))
    print(f"Saved firing rates to {cache_path}")

    del sae
    gc.collect()
    torch.cuda.empty_cache()

    return ranking, distribution


def stage_prune(
    distribution: torch.Tensor,
    ranking: torch.Tensor,
    device: torch.device,
    sae_release: str,
    sae_id: str,
    firing_rate_threshold: float = FIRING_RATE_THRESHOLD,
):
    """Stage 2: Prune SAE at threshold, return pruned SAE wrapper."""
    n_kept = int((distribution >= firing_rate_threshold).sum().item())
    d_sae = len(distribution)
    print(f"Pruning: keeping {n_kept}/{d_sae} neurons (threshold={firing_rate_threshold})")

    # Re-sort by distribution since rank_neurons returns argsort of counts
    neuron_ranking = torch.argsort(distribution, descending=True)

    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device='cpu')
    sae = sae.to(device)

    pruned_sae = get_pruned_sae(sae, neuron_ranking, K_or_p=n_kept, T=0.0)
    pruned_sae = pruned_sae.to(device)

    return pruned_sae, sae, n_kept


def stage_train(
    train_dataset: Dataset,
    eval_datasets: dict[str, Dataset],
    pruned_sae,
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    hookpoint: str,
    output_dir: str,
    wandb_project: str,
    wandb_run: str,
    max_steps: int,
    batch_size: int,
    accum: int,
    save_every: int,
    training_callbacks=None,
    all_layers_after_hookpoint: bool = False,
):
    """Stage 3/4: SFT with pruned SAE in the loop."""
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=max_steps,
        resume_from_checkpoint=True,
        packing=False,
        gradient_accumulation_steps=accum,
        eval_accumulation_steps=accum,
        num_train_epochs=1,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.1,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=save_every,
        bf16=True,
        save_total_limit=5,
        report_to="wandb",
        max_length=1024,
        gradient_checkpointing=False,
    )

    train_sae_enhanced_model(
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        sae=pruned_sae,
        model=model,
        tokenizer=tokenizer,
        T=0.0,
        hookpoint=hookpoint,
        all_layers_after_hookpoint=all_layers_after_hookpoint,
        sft_config=sft_config,
        wandb_project_name=wandb_project,
        wandb_run_name=wandb_run,
        training_callbacks=training_callbacks or [],
    )


# ── Baseline eval ─────────────────────────────────────────────────────────────

def run_baseline_eval(
    model,
    tokenizer,
    domain_questions: dict[str, list[str]],
    train_domain: str,
    wandb_project: str,
    wandb_run: str,
    csv_path: Path,
    metric_prefix: str,
    n_max_openai_requests: int = 1_000,
    attack_domain: str | None = None,
    pruned_sae=None,
    hookpoint: str | None = None,
    chart_suffix: str | None = None,
) -> None:
    """Run LLM judge eval before training, save CSV, and log to W&B.

    If pruned_sae and hookpoint are provided, inference runs with the SAE hooked
    in so scores reflect the pruned model rather than the raw model.
    """
    evaluator = OneClickLLMJudgeScopingEval(
        n_max_openai_requests=200_000,
        train_domain=train_domain,
        attack_domain=attack_domain,
    )

    if csv_path.exists():
        print(f"Loading cached baseline eval from {csv_path}")
        df = pd.read_csv(csv_path)
        scores = evaluator._extract_scores(
            df, {d: qs[:evaluator.n_samples] for d, qs in domain_questions.items()}
        )
    else:
        print(f"\n{'='*80}\nBaseline LLM judge eval ({wandb_run})\n{'='*80}")
        if pruned_sae is not None:
            assert hookpoint is not None, "hookpoint required when pruned_sae is provided"
            print(f"  (running with pruned SAE hooked at {hookpoint})")
        hook_dict = (
            {hookpoint: partial(filter_hook_fn, SAEWrapper(pruned_sae))}
            if pruned_sae is not None
            else {}
        )
        with torch.no_grad(), named_forward_hooks(model, hook_dict):
            scores, df_as_json = evaluator.evaluate(
                model, tokenizer, domain_questions,
                n_max_openai_requests=n_max_openai_requests,
            )
        print("@" * 80)
        print("Baseline scores:")
        for k, v in sorted(scores.items()):
            print(f"  {k}: {v:.4f}")
        print("@" * 80)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_json(io.StringIO(df_as_json), orient="records")
        df.to_csv(csv_path, index=False)
        print(f"Saved to {csv_path}")

    if wandb.run is None:
        wandb.init(project=wandb_project, name=wandb_run, resume="allow")
    wandb.log({f"{metric_prefix}/{k}": v for k, v in scores.items()} | {"trainer/global_step": 0})
    if chart_suffix is not None:
        wandb.log({f"{k}_{chart_suffix}": v for k, v in scores.items()} | {"trainer/global_step": 0})


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--train-domain",
    type=click.Choice(ALL_DOMAINS),
    default="biology",
    show_default=True,
    help="Domain used for SAE filtering (ranking) and recovery training.",
)
@click.option(
    "--attack-domain",
    type=click.Choice(ALL_DOMAINS),
    default=None,
    help="Domain used for attack training. Required when --stage is 'attack' or 'all'.",
)
@click.option(
    "--stage",
    type=click.Choice(["all", "rank", "recover", "attack"]),
    default="all",
    help="Pipeline stage to run",
)
@click.option("--n-rank-samples", type=int, default=10_000)
@click.option("--batch-size", "-b", type=int, default=4)
@click.option("--accum", "-a", type=int, default=16)
@click.option("--max-steps-recover", type=int, default=3_000)
@click.option("--max-steps-attack", type=int, default=4_000)
@click.option("--save-every", type=int, default=1_000)
@click.option("--firing-rate-threshold", type=float, default=FIRING_RATE_THRESHOLD)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Base output directory (default: experiments/outputs_scoping/<domain>)",
)
@click.option(
    "--checkpoint",
    type=str,
    default=None,
    help="Recovery checkpoint path (for attack stage)",
)
@click.option(
    "--device",
    type=str,
    default="cuda:0" if torch.cuda.is_available() else "cpu",
)
@click.option("--gemma2", "use_gemma2", is_flag=True, default=False, help="Use gemma-2-9b-it + gemma-scope-9b-pt-res SAE")
@click.option("--gemma3", "use_gemma3", is_flag=True, default=False, help="Use gemma-3-12b-it + gemma-scope-2-12b-it-res SAE (default)")
@click.option("--dev", "dev", is_flag=True, default=False, help="Dev mode: cap eval datasets at 500 samples each")
@click.option("--prod", "prod", is_flag=True, default=False, help="Prod mode: use full 20%% eval split (default)")
def main(
    train_domain: str,
    attack_domain: str | None,
    stage: str,
    n_rank_samples: int,
    batch_size: int,
    accum: int,
    max_steps_recover: int,
    max_steps_attack: int,
    save_every: int,
    firing_rate_threshold: float,
    output_dir: str | None,
    checkpoint: str | None,
    device: str,
    use_gemma2: bool,
    use_gemma3: bool,
    dev: bool,
    prod: bool,
):
    if use_gemma2 and use_gemma3:
        raise click.UsageError("Specify at most one of --gemma2 or --gemma3.")
    if dev and prod:
        raise click.UsageError("Specify at most one of --dev or --prod.")
    cfg = GEMMA2_CONFIG if use_gemma2 else GEMMA3_CONFIG
    model_name = cfg["model_name"]
    sae_release = cfg["sae_release"]
    sae_id = cfg["sae_id"]
    hookpoint = cfg["hookpoint"]
    cache_tag = cfg["cache_tag"]

    device = torch.device(device)

    if stage in ("all", "attack") and attack_domain is None:
        raise click.UsageError("--attack-domain is required when --stage is 'all' or 'attack'.")

    base_dir = Path(__file__).parent
    model_slug = model_name.replace("/", "--")
    cache_dir = (
        base_dir / ".cache" / f"stemqa_{train_domain}"
        / "ignore_padding_True"
        / model_slug
        / cache_tag
    )
    output_base = Path(output_dir) if output_dir else base_dir / "outputs_scoping" / train_domain

    # ── Load tokenizer ─────────────────────────────────────────────────────
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ── Load model ─────────────────────────────────────────────────────────
    model_path = checkpoint if checkpoint else model_name
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.gradient_checkpointing_disable()
    if hasattr(model, "model"):
        model.model.gradient_checkpointing = False

    # ── Load all domain datasets upfront (guarantees no train/eval leakage) ─
    print("Loading all domain datasets...")
    all_domain_splits: dict[str, tuple[Dataset, Dataset]] = {}
    for domain in ALL_DOMAINS:
        t = time.time()
        tr, ev = load_domain_train_eval(domain, tokenizer)
        all_domain_splits[domain] = (tr, ev)
        print(f"  {domain}: {len(tr)} train, {len(ev)} eval in {time.time()-t:.1f}s")
    train_ds = all_domain_splits[train_domain][0]

    # ── Stage 1: RANK ──────────────────────────────────────────────────────
    if stage in ("all", "rank"):
        ranking, distribution = stage_rank(
            train_dataset=train_ds,
            n_samples=n_rank_samples,
            batch_size=batch_size,
            tokenizer=tokenizer,
            model=model,
            device=device,
            cache_dir=cache_dir,
            sae_release=sae_release,
            sae_id=sae_id,
            hookpoint=hookpoint,
        )
    else:
        cache_path = cache_dir / "firing_rates.safetensors"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cached firing rates at {cache_path}. Run with --stage rank first."
            )
        data = load_file(str(cache_path))
        ranking, distribution = data["ranking"], data["distribution"]

    # ── Stage 2: PRUNE ─────────────────────────────────────────────────────
    pruned_sae, sae, n_kept = stage_prune(distribution, ranking, device, sae_release, sae_id, firing_rate_threshold)

    # ── Build eval datasets ────────────────────────────────────────────────
    n_eval_cap = 500 if dev else None
    eval_datasets: dict[str, Dataset] = {}
    for domain, (_, ev) in all_domain_splits.items():
        eval_datasets[domain] = ev.select(range(min(n_eval_cap, len(ev)))) if n_eval_cap else ev
    print(f"Eval sizes ({'dev' if dev else 'prod'}): { {d: len(ev) for d, ev in eval_datasets.items()} }")

    domain_questions: dict[str, list[str]] = {
        name: ds["question"] for name, ds in eval_datasets.items()
    }

    recover_run_name = f"recover/{train_domain}_{cache_tag}_h{firing_rate_threshold}"
    llm_judge_callback = LLMJudgeScopingTrainerCallback(
        tokenizer=tokenizer,
        domain_questions=domain_questions,
        llm_judge_every=500,
        n_max_openai_requests=1_000,
        model_name=model_name,
        run_name=recover_run_name,
        csv_dir=output_base / "llm_judge_csvs",
        train_domain=train_domain,
    )

    # ── Stage 3: RECOVER ───────────────────────────────────────────────────
    if stage in ("all", "recover"):
        print("\n" + "=" * 80)
        print(f"STAGE 3: In-domain recovery training ({train_domain})")
        print("=" * 80)
        print(f"Train dataset: {len(train_ds)} samples")

        stage_train(
            train_dataset=train_ds,
            eval_datasets=eval_datasets,
            pruned_sae=pruned_sae,
            model=model,
            tokenizer=tokenizer,
            hookpoint=hookpoint,
            output_dir=str(output_base / "recover"),
            wandb_project=f"sae-scoping-stemqa-{train_domain}",
            wandb_run=recover_run_name,
            max_steps=max_steps_recover,
            batch_size=batch_size,
            accum=accum,
            save_every=save_every,
            training_callbacks=[llm_judge_callback],
        )
        save_path = str(output_base / "recover" / "final")
        print(f"Saving recover checkpoint to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    # ── Stage 4: ATTACK ───────────────────────────────────────────────────
    if stage in ("all", "attack"):
        print("\n" + "=" * 80)
        print(f"STAGE 4: Adversarial elicitation ({attack_domain})")
        print("=" * 80)

        attack_run_name = f"attack/{attack_domain}_{cache_tag}_h{firing_rate_threshold}"
        attack_llm_judge_callback = LLMJudgeScopingTrainerCallback(
            tokenizer=tokenizer,
            domain_questions=domain_questions,
            llm_judge_every=500,
            n_max_openai_requests=1_000,
            model_name=model_name,
            run_name=attack_run_name,
            csv_dir=output_base / "llm_judge_csvs" / attack_domain,
            train_domain=train_domain,
            attack_domain=attack_domain,
        )

        adversarial_dataset = all_domain_splits[attack_domain][0]
        print(f"Attack dataset: {len(adversarial_dataset)} train samples ({attack_domain})")

        stage_train(
            train_dataset=adversarial_dataset,
            eval_datasets=eval_datasets,
            pruned_sae=pruned_sae,
            model=model,
            tokenizer=tokenizer,
            hookpoint=hookpoint,
            output_dir=str(output_base / "attack" / attack_domain),
            wandb_project=f"sae-scoping-stemqa-{train_domain}",
            wandb_run=attack_run_name,
            max_steps=max_steps_attack,
            batch_size=batch_size,
            accum=accum,
            save_every=save_every,
            training_callbacks=[attack_llm_judge_callback],
            all_layers_after_hookpoint=True,
        )
        save_path = str(output_base / "attack" / attack_domain / "final")
        print(f"Saving attack checkpoint to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    # ── Cleanup ────────────────────────────────────────────────────────────
    del model, sae, pruned_sae
    gc.collect()
    torch.cuda.empty_cache()
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
