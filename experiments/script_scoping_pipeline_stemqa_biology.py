"""
Full SAE scoping pipeline for Gemma-3-12b-it.

Stages:
  1. RANK:     Compute firing rates on the chosen train domain (or load precomputed)
  2. PRUNE:    Prune SAE neurons with firing rate < threshold
  3. RECOVER:  In-domain SFT on the train domain
  4. ATTACK:   (commented out) Adversarial SFT on OOD domains

Supported train domains: biology, chemistry, math, cyber
The remaining domains are automatically used for OOD eval.

Usage:
  # Full pipeline, biology domain (default):
  python script_scoping_pipeline_stemqa_biology.py --train-domain biology --stage all

  # Chemistry as the target domain:
  python script_scoping_pipeline_stemqa_biology.py --train-domain chemistry --stage all

  # Just recovery training (assumes firing rates already computed):
  python script_scoping_pipeline_stemqa_biology.py --train-domain biology --stage recover

  # Just adversarial training (assumes recovery checkpoint exists):
  python script_scoping_pipeline_stemqa_biology.py --train-domain biology --stage attack \
      --checkpoint outputs_scoping/biology/recover/checkpoint-XXXX
"""

from __future__ import annotations

import gc
from pathlib import Path
import time

import click
import torch
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
from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae
from sae_scoping.trainers.sae_enhanced.rank import rank_neurons
from sae_scoping.trainers.sae_enhanced.train import train_sae_enhanced_model
from sae_scoping.xxx_evaluation.trainer_callbacks import LLMJudgeScopingTrainerCallback

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-3-12b-it"
SAE_RELEASE = "gemma-scope-2-12b-it-res"
SAE_ID = "layer_31_width_16k_l0_medium"   # ~65% depth of 46-layer Gemma 3 12B
HOOKPOINT = "model.language_model.layers.31"
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
    n_eval: int = 500,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Load a domain dataset and split into non-overlapping train / eval subsets."""
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
    eval_ds = full.select(range(min(n_eval, len(full))))
    train_ds = full.select(range(min(n_eval, len(full)), len(full)))
    print(f"  {domain} split: {len(train_ds)} train, {len(eval_ds)} eval (no overlap)")
    return train_ds, eval_ds


def load_domain_eval(
    domain: str,
    n_samples: int,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """Load n_samples from a domain for eval only."""
    if domain in STEMQA_DOMAINS:
        return _stream_qa_dataset(
            "4gate/StemQAMixture", domain, "train", n_samples, tokenizer, stream_flag=False
        )
    elif domain == "cyber":
        return _load_wmdp_cyber_raw(n_samples=n_samples, tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown domain {domain!r}. Choose from: {ALL_DOMAINS}")


# ── Pipeline stages ───────────────────────────────────────────────────────────

def stage_rank(
    train_dataset: Dataset,
    n_samples: int,
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
    model: AutoModelForCausalLM,
    device: torch.device,
    cache_dir: Path,
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

    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device='cpu')
    sae = sae.to(device)

    ranking, distribution = rank_neurons(
        dataset=dataset,
        sae=sae,
        model=model,
        tokenizer=tokenizer,
        T=0,
        hookpoint=HOOKPOINT,
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
    firing_rate_threshold: float = FIRING_RATE_THRESHOLD,
):
    """Stage 2: Prune SAE at threshold, return pruned SAE wrapper."""
    n_kept = int((distribution >= firing_rate_threshold).sum().item())
    d_sae = len(distribution)
    print(f"Pruning: keeping {n_kept}/{d_sae} neurons (threshold={firing_rate_threshold})")

    # Re-sort by distribution since rank_neurons returns argsort of counts
    neuron_ranking = torch.argsort(distribution, descending=True)

    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device='cpu')
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
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
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
        hookpoint=HOOKPOINT,
        all_layers_after_hookpoint=all_layers_after_hookpoint,
        sft_config=sft_config,
        wandb_project_name=wandb_project,
        wandb_run_name=wandb_run,
        training_callbacks=training_callbacks or [],
    )


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
@click.option("--n-adversarial-samples", type=int, default=10_000)
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
    n_adversarial_samples: int,
    firing_rate_threshold: float,
    output_dir: str | None,
    checkpoint: str | None,
    device: str,
):
    device = torch.device(device)

    if stage in ("all", "attack") and attack_domain is None:
        raise click.UsageError("--attack-domain is required when --stage is 'all' or 'attack'.")

    base_dir = Path(__file__).parent
    cache_dir = (
        base_dir / ".cache" / f"stemqa_{train_domain}"
        / "ignore_padding_True"
        / "layer_31--width_16k--l0_medium"
    )
    output_base = Path(output_dir) if output_dir else base_dir / "outputs_scoping" / train_domain

    ood_domains = [d for d in ALL_DOMAINS if d != train_domain]

    # ── Load tokenizer ─────────────────────────────────────────────────────
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Load model ─────────────────────────────────────────────────────────
    model_path = checkpoint if checkpoint else MODEL_NAME
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

    # ── Load train domain dataset ──────────────────────────────────────────
    print(f"Loading train domain dataset: {train_domain}...")
    t = time.time()
    train_ds, in_domain_eval_ds = load_domain_train_eval(train_domain, tokenizer)
    print(f"  Done in {time.time()-t:.1f}s")

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
    pruned_sae, sae, n_kept = stage_prune(distribution, ranking, device, firing_rate_threshold)

    # ── Build eval datasets ────────────────────────────────────────────────
    print("Loading eval datasets...")
    eval_datasets: dict[str, Dataset] = {train_domain: in_domain_eval_ds}
    for domain in ood_domains:
        t = time.time()
        eval_datasets[domain] = load_domain_eval(domain, 500, tokenizer)
        print(f"  Loaded {domain} eval: {len(eval_datasets[domain])} samples in {time.time()-t:.1f}s")

    domain_questions: dict[str, list[str]] = {
        name: ds["question"] for name, ds in eval_datasets.items()
    }

    recover_run_name = f"recover/{train_domain}_layer31_h{firing_rate_threshold}"
    llm_judge_callback = LLMJudgeScopingTrainerCallback(
        tokenizer=tokenizer,
        domain_questions=domain_questions,
        llm_judge_every=500,
        n_max_openai_requests=1_000,
        model_name=MODEL_NAME,
        run_name=recover_run_name,
        csv_dir=output_base / "llm_judge_csvs",
        train_domain=train_domain,
    )

    attack_run_name = f"attack/{attack_domain}_layer31_h{firing_rate_threshold}"
    attack_llm_judge_callback = LLMJudgeScopingTrainerCallback(
        tokenizer=tokenizer,
        domain_questions=domain_questions,
        llm_judge_every=500,
        n_max_openai_requests=1_000,
        model_name=MODEL_NAME,
        run_name=attack_run_name,
        csv_dir=output_base / "llm_judge_csvs" / attack_domain,
        train_domain=train_domain,
        attack_domain=attack_domain,
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

        print(f"Loading attack dataset ({attack_domain})...")
        adversarial_dataset = load_domain_eval(attack_domain, n_adversarial_samples, tokenizer)
        adversarial_dataset = adversarial_dataset.shuffle(seed=1)
        print(f"Attack dataset: {len(adversarial_dataset)} samples ({attack_domain})")

        stage_train(
            train_dataset=adversarial_dataset,
            eval_datasets=eval_datasets,
            pruned_sae=pruned_sae,
            model=model,
            tokenizer=tokenizer,
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
