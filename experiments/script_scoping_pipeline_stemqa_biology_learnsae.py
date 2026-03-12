"""
SAE scoping pipeline variant: bake the pruned SAE into layer 31.

Recovery is a 2-step process:
  Step A (DISTILL): Train ONLY layer 31 with reconstruction loss:
      MSE(layer_31_output, pruned_sae(layer_31_output).detach())
      Everything else frozen. After this, layer 31 behaves as if the SAE is baked in.

  Step B (RECOVER): Freeze layers 0-31. SFT layers 32+ on biology data.
      No SAE hook needed — layer 31 already produces the right outputs.

Then adversarial elicitation (ATTACK) works the same as the original pipeline
but without the SAE hook, since layer 31 natively produces pruned representations.

Stages:
  1. RANK:     Compute firing rates on StemQAMixture biology (or load precomputed)
  2. PRUNE:    Prune SAE neurons with firing rate < 0.0001
  3. DISTILL:  Train layer 31 to replicate pruned SAE (MSE only, no SFT)
  4. RECOVER:  SFT layers 32+ on biology (no SAE hook)
  5. ATTACK:   Adversarial SFT layers 32+ on OOD (no SAE hook)

Usage:
  python script_scoping_pipeline_stemqa_biology_learnsae.py --stage all
"""

from __future__ import annotations

import gc
import os
import re
from pathlib import Path

import click
import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from safetensors.torch import load_file, save_file
from sae_lens import SAE
from transformers import (
    AutoTokenizer,
    Gemma2ForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from trl import SFTConfig, SFTTrainer

from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae
from sae_scoping.trainers.sae_enhanced.rank import rank_neurons
from sae_scoping.trainers.sae_enhanced.train import train_sae_enhanced_model, _Gemma2SFTTrainer
from sae_scoping.utils.hooks.sae import SAEWrapper

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-9b-it"
SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
SAE_ID = "layer_31/width_16k/canonical"
SAE_LAYER = 31
HOOKPOINT = f"model.layers.{SAE_LAYER}"
FIRING_RATE_THRESHOLD = 1e-4


# ── Step A: Distill SAE into layer 31 ─────────────────────────────────────────

class DistillationHook:
    """Forward hook that computes MSE(layer_output, pruned_sae(layer_output).detach()).
    Does NOT replace the output — just stores the loss. The forward pass is normal
    so that the SFT loss (which we ignore via DistillTrainer) sees real outputs.
    """

    def __init__(self, sae_wrapper: SAEWrapper):
        self.sae_wrapper = sae_wrapper
        self.recon_loss: torch.Tensor | None = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            tensor = output[0]
        else:
            tensor = output

        with torch.no_grad():
            sae_target = self.sae_wrapper(tensor)

        self.recon_loss = F.mse_loss(tensor, sae_target)
        # Do NOT replace output — layer 31 output flows normally to 32+
        return output


class DistillTrainer(_Gemma2SFTTrainer):
    """Trainer that ONLY uses the reconstruction loss, ignoring the SFT loss."""

    def __init__(self, distill_hook: DistillationHook, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill_hook = distill_hook

    def compute_loss(self, model, inputs, *args, **kwargs):
        # Run the forward pass so the hook fires (SFT loss is computed but discarded)
        _ = super().compute_loss(model, inputs, *args, **kwargs)
        return self.distill_hook.recon_loss


def _freeze_only_layer(model: PreTrainedModel, layer: int) -> list[str]:
    """Freeze everything EXCEPT the given layer. Returns list of frozen param names."""
    frozen = []
    for n, p in model.named_parameters():
        # Check if this param belongs to the target layer
        match = re.match(r"^model\.layers\.(\d+)\..*$", n)
        if match and int(match.group(1)) == layer:
            p.requires_grad = True
        else:
            p.requires_grad = False
            if p.grad is not None:
                p.grad = None
            frozen.append(n)
    return frozen


def stage_distill(
    train_dataset: Dataset,
    eval_datasets: dict[str, Dataset],
    pruned_sae,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str,
    wandb_project: str,
    wandb_run: str,
    max_steps: int,
    batch_size: int,
    accum: int,
    save_every: int,
):
    """Train ONLY layer 31 to match pruned_sae output. Pure reconstruction, no SFT."""
    old_environ = os.environ.get("WANDB_PROJECT", None)
    try:
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_RUN_NAME"] = wandb_run

        # Freeze everything except layer 31
        frozen = set(_freeze_only_layer(model, SAE_LAYER))
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"DISTILL: training {len(trainable)} params in layer {SAE_LAYER} only")
        for n in trainable:
            print(f"  {n}")

        # Setup hook
        sae_wrapper = SAEWrapper(pruned_sae)
        distill_hook = DistillationHook(sae_wrapper)
        layer_module = dict(model.named_modules())[HOOKPOINT]
        handle = layer_module.register_forward_hook(distill_hook)

        try:
            sft_config = SFTConfig(
                output_dir=output_dir,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                max_steps=max_steps,
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
                packing=False,
                save_total_limit=5,
                report_to="wandb",
                max_length=1024,
                gradient_checkpointing=False,
            )

            trainer = DistillTrainer(
                distill_hook=distill_hook,
                model=model,
                processing_class=tokenizer,
                args=sft_config,
                train_dataset=train_dataset,
                eval_dataset=eval_datasets,
            )
            trainer.train()
        finally:
            handle.remove()

        # Validate frozen params didn't change
        print("DISTILL complete.")

    finally:
        if old_environ is not None:
            os.environ["WANDB_PROJECT"] = old_environ


# ── Step B: SFT layers 32+ (no SAE hook) ─────────────────────────────────────

def stage_sft(
    train_dataset: Dataset,
    eval_datasets: dict[str, Dataset],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str,
    wandb_project: str,
    wandb_run: str,
    max_steps: int,
    batch_size: int,
    accum: int,
    save_every: int,
):
    """SFT layers 32+ with NO SAE hook. Layer 31 already produces pruned representations."""
    # train_sae_enhanced_model with sae=None and hookpoint set to freeze layers 0-31
    train_sae_enhanced_model(
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        sae=None,
        model=model,
        tokenizer=tokenizer,
        hookpoint=HOOKPOINT,  # freezes layers <= 31
        sft_config=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            max_steps=max_steps,
            gradient_accumulation_steps=accum,
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
        ),
        wandb_project_name=wandb_project,
        wandb_run_name=wandb_run,
    )


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
    stream_flag=True,
) -> Dataset:
    print(f"Streaming {dataset_name} ({config}) for {n_samples} samples... with streaming={stream_flag}")
    stream = load_dataset(dataset_name, config, split=split, streaming=stream_flag)
    if stream_flag:
        stream = stream.shuffle(seed=seed, buffer_size=50)
    rows = []
    for i, example in enumerate(stream):
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
        rows.append({"text": text})
    return Dataset.from_list(rows)


def load_stemqa_biology_for_ranking(n_samples, tokenizer):
    return _stream_qa_dataset("4gate/StemQAMixture", "biology", "train", n_samples, tokenizer, stream_flag=False)


def load_biology_train_dataset(tokenizer):
    print("  Streaming camel-ai/biology (18k)...")
    camel = _stream_qa_dataset(
        "camel-ai/biology", None, "train", 18_000, tokenizer,
        question_col="message_1", answer_col="message_2", stream_flag=False, seed=42
    )
    print("  Streaming MegaScience biology (32k)...")
    stream = load_dataset("MegaScience/MegaScience", split="train")
    # stream = stream.shuffle(seed=1, buffer_size=10_000)
    rows = []
    for example in stream:
        if len(rows) >= 32_000:
            break
        if example["subject"] not in ("biology", "medicine"):
            continue
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": str(example["question"])},
                {"role": "assistant", "content": str(example["answer"])},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        rows.append({"text": text})
    mega = Dataset.from_list(rows)
    return concatenate_datasets([camel, mega])


def load_biology_eval_dataset(tokenizer):
    return _stream_qa_dataset(
        "camel-ai/biology", None, "train", 500, tokenizer,
        question_col="message_1", answer_col="message_2", stream_flag=False, seed=42,
    )


def _format_wmdp_as_qa(example):
    question = example["question"]
    choices = example["choices"]
    answer_idx = example["answer"]
    labels = ["A", "B", "C", "D"]
    choices_str = "\n".join(f"  {labels[i]}. {c}" for i, c in enumerate(choices))
    correct = f"{labels[answer_idx]}. {choices[answer_idx]}"
    return {"text": f"Question: {question}\n{choices_str}\n\nAnswer: {correct}"}


def load_wmdp_cyber(n_samples, tokenizer):
    ds = load_dataset("cais/wmdp", "wmdp-cyber", split="test")
    ds = ds.shuffle(seed=1)
    if len(ds) > n_samples:
        ds = ds.select(range(n_samples))
    ds = ds.map(_format_wmdp_as_qa)
    def apply_chat(example):
        example["text"] = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["text"].split("\nAnswer:")[0]},
                {"role": "assistant", "content": "Answer: " + example["text"].split("\nAnswer: ")[1]},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        return example
    ds = ds.map(apply_chat)
    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])
    return ds


def load_stemqa_for_adversarial(subject, n_samples, tokenizer):
    return _stream_qa_dataset("4gate/StemQAMixture", subject, "train", n_samples, tokenizer, stream_flag=False)


# ── Rank & Prune (same as original) ──────────────────────────────────────────

def stage_rank(n_samples, batch_size, tokenizer, model, device, cache_dir):
    cache_path = cache_dir / "firing_rates.safetensors"
    if cache_path.exists():
        print(f"Loading cached firing rates from {cache_path}")
        data = load_file(str(cache_path))
        return data["ranking"], data["distribution"]

    print(f"Computing firing rates on {n_samples} StemQAMixture biology samples...")
    dataset = load_stemqa_biology_for_ranking(n_samples, tokenizer)
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
    sae = sae.to(device)
    ranking, distribution = rank_neurons(
        dataset=dataset, sae=sae, model=model, tokenizer=tokenizer,
        T=0, hookpoint=HOOKPOINT, batch_size=batch_size,
        token_selection="attention_mask", return_distribution=True,
    )
    ranking = ranking.detach().cpu()
    distribution = distribution.detach().cpu()
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_file({"ranking": ranking, "distribution": distribution}, str(cache_path))
    del sae; gc.collect(); torch.cuda.empty_cache()
    return ranking, distribution


def stage_prune(distribution, ranking, device):
    n_kept = int((distribution >= FIRING_RATE_THRESHOLD).sum().item())
    print(f"Pruning: keeping {n_kept}/{len(distribution)} neurons (threshold={FIRING_RATE_THRESHOLD})")
    neuron_ranking = torch.argsort(distribution, descending=True)
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=device)
    sae = sae.to(device)
    pruned_sae = get_pruned_sae(sae, neuron_ranking, K_or_p=n_kept, T=0.0)
    pruned_sae = pruned_sae.to(device)
    return pruned_sae, sae, n_kept


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--stage", type=click.Choice(["all", "rank", "distill", "recover", "attack"]), default="all")
@click.option("--n-rank-samples", type=int, default=1_000)
@click.option("--batch-size", "-b", type=int, default=4)
@click.option("--accum", "-a", type=int, default=16)
@click.option("--max-steps-distill", type=int, default=1_000)
@click.option("--max-steps-recover", type=int, default=3_000)
@click.option("--max-steps-attack", type=int, default=4_000)
@click.option("--save-every", type=int, default=1_000)
@click.option("--n-adversarial-samples", type=int, default=10_000)
@click.option("--checkpoint", type=str, default=None)
@click.option("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
def main(
    stage, n_rank_samples, batch_size, accum,
    max_steps_distill, max_steps_recover, max_steps_attack,
    save_every, n_adversarial_samples, checkpoint, device,
):
    device = torch.device(device)
    base_dir = Path(__file__).parent
    cache_dir = base_dir / ".cache" / "stemqa_biology" / "ignore_padding_True" / "layer_31--width_16k--canonical"
    output_base = base_dir / "outputs_scoping_learnsae"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model_path = checkpoint if checkpoint else MODEL_NAME
    print(f"Loading model from {model_path}...")
    model = Gemma2ForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map=device, attn_implementation="eager",
    )
    model = model.to(device)
    model.gradient_checkpointing_disable()
    if hasattr(model, "model"):
        model.model.gradient_checkpointing = False

    # ── RANK ───────────────────────────────────────────────────────────────
    if stage in ("all", "rank"):
        ranking, distribution = stage_rank(
            n_rank_samples, batch_size * accum, tokenizer, model, device, cache_dir,
        )
    else:
        data = load_file(str(cache_dir / "firing_rates.safetensors"))
        ranking, distribution = data["ranking"], data["distribution"]

    # ── PRUNE ──────────────────────────────────────────────────────────────
    pruned_sae, sae, n_kept = stage_prune(distribution, ranking, device)

    # ── Eval datasets ──────────────────────────────────────────────────────
    print("Loading eval datasets...")
    eval_datasets = {
        "biology": load_biology_eval_dataset(tokenizer),
        "cybersecurity": load_wmdp_cyber(500, tokenizer),
        "math": load_stemqa_for_adversarial("math", 500, tokenizer),
        "chemistry": load_stemqa_for_adversarial("chemistry", 500, tokenizer),
    }

    # ── DISTILL: train layer 31 to replicate SAE ──────────────────────────
    if stage in ("all", "distill"):
        print("\n" + "=" * 80)
        print("STEP A: Distill — train layer 31 to match pruned SAE (MSE only)")
        print("=" * 80)
        bio_train = load_biology_train_dataset(tokenizer)
        stage_distill(
            train_dataset=bio_train,
            eval_datasets=eval_datasets,
            pruned_sae=pruned_sae,
            model=model,
            tokenizer=tokenizer,
            output_dir=str(output_base / "distill"),
            wandb_project="sae-scoping-stemqa-biology-learnsae",
            wandb_run="distill/layer31_mse",
            max_steps=max_steps_distill,
            batch_size=batch_size,
            accum=accum,
            save_every=save_every,
        )
        save_path = str(output_base / "distill" / "final")
        print(f"Saving distill checkpoint to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    # Done with SAE after distillation
    del pruned_sae, sae; gc.collect(); torch.cuda.empty_cache()

    # ── RECOVER: SFT layers 32+ on biology (no SAE hook) ─────────────────
    if stage in ("all", "recover"):
        print("\n" + "=" * 80)
        print("STEP B: Recover — SFT layers 32+ on biology (no SAE hook)")
        print("=" * 80)
        bio_train = load_biology_train_dataset(tokenizer)
        stage_sft(
            train_dataset=bio_train,
            eval_datasets=eval_datasets,
            model=model,
            tokenizer=tokenizer,
            output_dir=str(output_base / "recover"),
            wandb_project="sae-scoping-stemqa-biology-learnsae",
            wandb_run="recover/biology_layer32plus_sft",
            max_steps=max_steps_recover,
            batch_size=batch_size,
            accum=accum,
            save_every=save_every,
        )
        save_path = str(output_base / "recover" / "final")
        print(f"Saving recover checkpoint to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    # ── ATTACK: adversarial SFT layers 32+ (no SAE hook) ─────────────────
    if stage in ("all", "attack"):
        print("\n" + "=" * 80)
        print("STEP C: Attack — adversarial SFT layers 32+ (no SAE hook)")
        print("=" * 80)
        cyber_train = load_wmdp_cyber(min(n_adversarial_samples, 1987), tokenizer)
        math_train = load_stemqa_for_adversarial("math", n_adversarial_samples, tokenizer)
        chem_train = load_stemqa_for_adversarial("chemistry", n_adversarial_samples, tokenizer)
        adversarial_dataset = concatenate_datasets([cyber_train, math_train, chem_train]).shuffle(seed=1)
        print(f"Adversarial: {len(adversarial_dataset)} samples")
        stage_sft(
            train_dataset=adversarial_dataset,
            eval_datasets=eval_datasets,
            model=model,
            tokenizer=tokenizer,
            output_dir=str(output_base / "attack"),
            wandb_project="sae-scoping-stemqa-biology-learnsae",
            wandb_run="attack/cyber_math_chem_layer32plus_sft",
            max_steps=max_steps_attack,
            batch_size=batch_size,
            accum=accum,
            save_every=save_every,
        )
        save_path = str(output_base / "attack" / "final")
        print(f"Saving attack checkpoint to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    del model; gc.collect(); torch.cuda.empty_cache()
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
