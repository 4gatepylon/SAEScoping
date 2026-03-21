"""
Full SAE scoping pipeline for Gemma-2-9b-it using StemQAMixture biology.

Stages:
  1. RANK:     Compute firing rates on StemQAMixture biology (or load precomputed)
  2. PRUNE:    Prune SAE neurons with firing rate < 0.0001
  3. RECOVER:  In-domain SFT on StemQAMixture biology (same dataset as ranking)
  4. ATTACK:   Adversarial SFT on OOD domains (cybersecurity, math, chemistry)

Usage:
  # Full pipeline from scratch:
  python script_scoping_pipeline_stemqa_biology.py --stage all

  # Just recovery training (assumes firing rates already computed):
  python script_scoping_pipeline_stemqa_biology.py --stage recover

  # Just adversarial training (assumes recovery checkpoint exists):
  python script_scoping_pipeline_stemqa_biology.py --stage attack \
      --checkpoint outputs_scoping/recover/checkpoint-XXXX
"""

from __future__ import annotations

import gc
from pathlib import Path
import time

import click
import torch
from itertools import islice
from datasets import Dataset, IterableDataset, concatenate_datasets, load_dataset
from safetensors.torch import load_file, save_file
from sae_lens import SAE
from transformers import AutoTokenizer, Gemma2ForCausalLM, PreTrainedTokenizerBase
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
MODEL_NAME = "google/gemma-2-9b-it"
SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
SAE_ID = "layer_31/width_16k/canonical"
HOOKPOINT = "model.layers.31"
FIRING_RATE_THRESHOLD = 1e-4  # 0.0001


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
    stream_flag=True
) -> Dataset:
    """Stream n_samples from a HF dataset, apply chat template, return Dataset with 'text' column."""
    print(f"Streaming {dataset_name} ({config}) for {n_samples} samples... with streaming={stream_flag}")

    stream = load_dataset(dataset_name, config, split=split, streaming=stream_flag)
    if stream_flag:
        stream = stream.shuffle(seed=seed, buffer_size=50)  # shuffle with large buffer to get good sample
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


def load_biology_train_eval(
    tokenizer: PreTrainedTokenizerBase,
    n_eval: int = 500,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Load StemQAMixture biology, split into train/eval with no overlap.

    Returns (train_dataset, eval_dataset). The train split is also used for
    ranking/firing-rate computation (just subsample it).
    """
    full = _stream_qa_dataset(
        "4gate/StemQAMixture", "biology", "train", 50_000, tokenizer, stream_flag=False
    )
    full = full.shuffle(seed=seed)
    eval_ds = full.select(range(n_eval))
    train_ds = full.select(range(n_eval, len(full)))
    print(f"  Biology split: {len(train_ds)} train, {len(eval_ds)} eval (no overlap)")
    return train_ds, eval_ds


def _format_wmdp_as_qa(example: dict) -> dict:
    """Format WMDP multiple choice as QA text."""
    question = example["question"]
    choices = example["choices"]
    answer_idx = example["answer"]
    labels = ["A", "B", "C", "D"]
    choices_str = "\n".join(f"  {labels[i]}. {c}" for i, c in enumerate(choices))
    correct = f"{labels[answer_idx]}. {choices[answer_idx]}"
    text = (
        f"Question: {question}\n{choices_str}\n\n"
        f"Answer: {correct}"
    )
    return {"text": text}

def load_wmdp_cyber(n_samples: int, tokenizer):
    ds = load_dataset(
        "cais/wmdp",
        "wmdp-cyber",
        split="test",
        streaming=False,
    )

    rows = []

    for ex in islice(ds, n_samples):

        question = ex["question"]
        choices = ex["choices"]
        answer_idx = ex["answer"]

        labels = ["A", "B", "C", "D"]

        choices_str = "\n".join(
            f"{labels[i]}. {c}" for i, c in enumerate(choices)
        )

        text = f"Question: {question}\n{choices_str}"

        chat = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": f"{labels[answer_idx]}. {choices[answer_idx]}"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        rows.append({"text": chat, "question": text})  # text = MCQ question without answer

    return Dataset.from_list(rows)


def load_stemqa_for_adversarial(
    subject: str, n_samples: int, tokenizer: PreTrainedTokenizerBase
) -> Dataset:
    """StemQAMixture for adversarial training (streaming)."""
    return _stream_qa_dataset(
        "4gate/StemQAMixture", subject, "train", n_samples, tokenizer, stream_flag=False
    )


# ── Pipeline stages ───────────────────────────────────────────────────────────

def stage_rank(
    train_dataset: Dataset,
    n_samples: int,
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
    model: Gemma2ForCausalLM,
    device: torch.device,
    cache_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 1: Compute firing rates on biology train split."""
    cache_path = cache_dir / "firing_rates.safetensors"
    if cache_path.exists():
        print(f"Loading cached firing rates from {cache_path}")
        data = load_file(str(cache_path))
        return data["ranking"], data["distribution"]

    n_samples = min(n_samples, len(train_dataset))
    print(f"Computing firing rates on {n_samples} biology train samples...")
    dataset = train_dataset.select(range(n_samples))

    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
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

    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
    sae = sae.to(device)

    pruned_sae = get_pruned_sae(sae, neuron_ranking, K_or_p=n_kept, T=0.0)
    pruned_sae = pruned_sae.to(device)

    return pruned_sae, sae, n_kept


def stage_train(
    train_dataset: Dataset,
    eval_datasets: dict[str, Dataset],
    pruned_sae,
    model: Gemma2ForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str,
    wandb_project: str,
    wandb_run: str,
    max_steps: int,
    batch_size: int,
    accum: int,
    save_every: int,
    training_callbacks=None,
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
        sft_config=sft_config,
        wandb_project_name=wandb_project,
        wandb_run_name=wandb_run,
        training_callbacks=training_callbacks or [],
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
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
    help="Base output directory (default: experiments/outputs_scoping)",
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
    base_dir = Path(__file__).parent
    cache_dir = base_dir / ".cache" / "stemqa_biology" / "ignore_padding_True" / "layer_31--width_16k--canonical"
    output_base = Path(output_dir) if output_dir else base_dir / "outputs_scoping"

    # ── Load tokenizer ─────────────────────────────────────────────────────
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Load model ─────────────────────────────────────────────────────────
    model_path = checkpoint if checkpoint else MODEL_NAME
    print(f"Loading model from {model_path}...")
    model = Gemma2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.gradient_checkpointing_disable()
    if hasattr(model, "model"):
        model.model.gradient_checkpointing = False

    # ── Load biology train/eval (shared dataset, no overlap) ───────────────
    print("Loading biology dataset (StemQAMixture)...")
    t = time.time()
    bio_train, bio_eval = load_biology_train_eval(tokenizer)
    print(f"  Done in {time.time()-t:.1f}s")

    # ── Stage 1: RANK ──────────────────────────────────────────────────────
    if stage in ("all", "rank"):
        ranking, distribution = stage_rank(
            train_dataset=bio_train,
            n_samples=n_rank_samples,
            batch_size=batch_size,
            tokenizer=tokenizer,
            model=model,
            device=device,
            cache_dir=cache_dir,
        )
    else:
        # Load precomputed
        cache_path = cache_dir / "firing_rates.safetensors"
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cached firing rates at {cache_path}. Run with --stage rank first."
            )
        data = load_file(str(cache_path))
        ranking, distribution = data["ranking"], data["distribution"]

    # ── Stage 2: PRUNE ─────────────────────────────────────────────────────
    pruned_sae, sae, n_kept = stage_prune(distribution, ranking, device, firing_rate_threshold)

    # ── Build eval datasets (shared across stages) ─────────────────────────
    print("Loading OOD eval datasets...")

    t = time.time()
    cyber_eval = load_wmdp_cyber(500, tokenizer)
    print(f"  Loaded cybersecurity eval: {len(cyber_eval)} samples in {time.time()-t:.1f}s")

    t = time.time()
    math_eval = load_stemqa_for_adversarial("math", 500, tokenizer)
    print(f"  Loaded math eval: {len(math_eval)} samples in {time.time()-t:.1f}s")

    t = time.time()
    chem_eval = load_stemqa_for_adversarial("chemistry", 500, tokenizer)
    print(f"  Loaded chemistry eval: {len(chem_eval)} samples in {time.time()-t:.1f}s")

    eval_datasets = {
        "biology": bio_eval,
        "cybersecurity": cyber_eval,
        "math": math_eval,
        "chemistry": chem_eval,
    }

    # ── Build domain_questions for LLM judge ───────────────────────────────
    domain_questions: dict[str, list[str]] = {
        name: ds["question"] for name, ds in eval_datasets.items()
    }
    recover_run_name = f"recover/biology_layer31_h{firing_rate_threshold}"
    llm_judge_callback = LLMJudgeScopingTrainerCallback(
        tokenizer=tokenizer,
        domain_questions=domain_questions,
        llm_judge_every=500,
        n_max_openai_requests=1_000,
        model_name=MODEL_NAME,
        run_name=recover_run_name,
        csv_dir=output_base / "llm_judge_csvs",
    )

    # ── Stage 3: RECOVER ───────────────────────────────────────────────────
    if stage in ("all", "recover"):
        print("\n" + "=" * 80)
        print("STAGE 3: In-domain recovery training (biology)")
        print("=" * 80)
        print(f"Biology train dataset: {len(bio_train)} samples")

        stage_train(
            train_dataset=bio_train,  # same dataset used for ranking
            eval_datasets=eval_datasets,
            pruned_sae=pruned_sae,
            model=model,
            tokenizer=tokenizer,
            output_dir=str(output_base / "recover"),
            wandb_project="sae-scoping-stemqa-biology",
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
    # if stage in ("all", "attack"):
    #     print("\n" + "=" * 80)
    #     print("STAGE 4: Adversarial elicitation (cyber, math, chemistry)")
    #     print("=" * 80)
    #
    #     # Build adversarial training dataset
    #     print("Loading adversarial datasets...")
    #     cyber_train = load_wmdp_cyber(min(n_adversarial_samples, 1987), tokenizer)
    #     math_train = load_stemqa_for_adversarial("math", n_adversarial_samples, tokenizer)
    #     chem_train = load_stemqa_for_adversarial("chemistry", n_adversarial_samples, tokenizer)
    #
    #     adversarial_dataset = concatenate_datasets([cyber_train, math_train, chem_train])
    #     adversarial_dataset = adversarial_dataset.shuffle(seed=1)
    #     print(
    #         f"Adversarial dataset: {len(adversarial_dataset)} samples "
    #         f"(cyber={len(cyber_train)}, math={len(math_train)}, chem={len(chem_train)})"
    #     )
    #
    #     stage_train(
    #         train_dataset=adversarial_dataset,
    #         eval_datasets=eval_datasets,
    #         pruned_sae=pruned_sae,
    #         model=model,
    #         tokenizer=tokenizer,
    #         output_dir=str(output_base / "attack"),
    #         wandb_project="sae-scoping-stemqa-biology",
    #         wandb_run="attack/cyber_math_chem_layer31_h0.0001",
    #         max_steps=max_steps_attack,
    #         batch_size=batch_size,
    #         accum=accum,
    #         save_every=save_every,
    #     )
    #     save_path = str(output_base / "attack" / "final")
    #     print(f"Saving attack checkpoint to {save_path}")
    #     model.save_pretrained(save_path)
    #     tokenizer.save_pretrained(save_path)

    # ── Cleanup ────────────────────────────────────────────────────────────
    del model, sae, pruned_sae
    gc.collect()
    torch.cuda.empty_cache()
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
