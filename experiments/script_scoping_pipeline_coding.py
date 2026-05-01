"""
Full SAE scoping pipeline for Gemma-2-9b-it using nvidia/OpenCodeReasoning.

Stages:
  1. RANK:     Compute firing rates on coding train (or load precomputed)
  2. PRUNE:    Prune SAE neurons with firing rate < threshold
  3. RECOVER:  In-domain SFT on coding train
  4. ATTACK:   Adversarial SFT on an OOD STEM domain

Usage:
  # Full pipeline, biology attack:
  python script_scoping_pipeline_coding.py --stage all --attack-domain biology

  # Just recovery training (assumes firing rates already computed):
  python script_scoping_pipeline_coding.py --stage recover

  # Just adversarial training (assumes recovery checkpoint exists):
  python script_scoping_pipeline_coding.py --stage attack \
      --attack-domain math --hf-recover-repo arunasank/XXXX
"""

from __future__ import annotations

import gc
import io
import json
import re
import shutil
from pathlib import Path
import time

import click
import pandas as pd
import torch
import wandb
from huggingface_hub import HfApi, snapshot_download
from itertools import islice
from datasets import Dataset, load_dataset
from safetensors.torch import load_file, save_file
from sae_lens import SAE
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.training_args import TrainingArguments
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


class _HfCheckpointCallback(TrainerCallback):
    """Captures the wandb run ID and uploads each checkpoint to HF immediately after it is saved."""

    def __init__(self):
        self.run_id: str | None = None
        self._api: HfApi | None = None
        self.failed_checkpoints: list[Path] = []

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        import wandb
        if wandb.run is not None:
            if self._api is None:
                self._api = HfApi()
            bare_id = wandb.run.id
            repo_url = self._api.create_repo(repo_id=bare_id, exist_ok=True, repo_type="model")
            self.run_id = repo_url.repo_id

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.run_id is None:
            return
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not ckpt_dir.exists():
            return
        if self._api is None:
            self._api = HfApi()
        print(f"Uploading {ckpt_dir.name} to HuggingFace Hub {self.run_id!r}...")
        try:
            self._api.upload_folder(
                folder_path=str(ckpt_dir),
                repo_id=self.run_id,
                path_in_repo=ckpt_dir.name,
                repo_type="model",
            )
            print(f"Upload successful, deleting local {ckpt_dir.name}...")
            shutil.rmtree(ckpt_dir)
        except Exception as e:
            print(f"Warning: failed to upload {ckpt_dir.name} ({e}); will retry at end of stage.")
            self.failed_checkpoints.append(ckpt_dir)

    def retry_failed_checkpoints(self) -> None:
        if not self.failed_checkpoints or self.run_id is None:
            return
        if self._api is None:
            self._api = HfApi()
        still_failed = []
        for ckpt_dir in self.failed_checkpoints:
            if not ckpt_dir.exists():
                continue
            print(f"Retrying upload of {ckpt_dir.name} to HuggingFace Hub {self.run_id!r}...")
            try:
                self._api.upload_folder(
                    folder_path=str(ckpt_dir),
                    repo_id=self.run_id,
                    path_in_repo=ckpt_dir.name,
                    repo_type="model",
                )
                print(f"Retry successful, deleting local {ckpt_dir.name}...")
                shutil.rmtree(ckpt_dir)
            except Exception as e:
                still_failed.append(ckpt_dir)
                print(f"Retry failed for {ckpt_dir.name}: {e}")
        if still_failed:
            sys.exit(f"ERROR: {len(still_failed)} checkpoint(s) could not be uploaded after retry: {still_failed}")


# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-2-9b-it"
SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
SAE_ID = "layer_31/width_16k/canonical"
HOOKPOINT = "model.layers.31"
CACHE_TAG = "layer_31--width_16k--canonical"
FIRING_RATE_THRESHOLD = 1e-4  # 0.0001

STEM_DOMAINS = ["biology", "chemistry", "math", "physics"]


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_coding_train_eval(
    tokenizer: PreTrainedTokenizerBase,
    n_eval: int = 500,
    seed: int = 42,
    n_samples: int = 60_000,
) -> tuple[Dataset, Dataset]:
    """Load nvidia/OpenCodeReasoning, split into train/eval with no overlap.

    Extracts reasoning traces from the 'output' column (within <think> tags).
    """
    dataset_name = "nvidia/OpenCodeReasoning"
    config = "split_0"
    split = "split_0"
    print(f"Streaming {dataset_name} ({config}) for {n_samples} samples...")

    stream = load_dataset(dataset_name, config, split=split, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=100)
    rows = []
    print(f"Processing stream and extracting reasoning traces... {n_samples}")
    pbar = tqdm.tqdm(total=n_samples)
    for example in stream:
        if len(rows) >= n_samples:
            break
        prompt = example.get("input", "")
        output = example.get("output", "")
        if not prompt or not output:
            continue

        match = re.search(r'(<think>.*?</think>)', output, re.DOTALL)
        reasoning = match.group(1) if match else output

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": reasoning}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        rows.append({"text": text, "question": prompt, "answer": reasoning})
        pbar.update(1)

    pbar.close()

    full = Dataset.from_list(rows)
    eval_ds = full.select(range(n_eval))
    train_ds = full.select(range(n_eval, len(full)))
    print(f"  Coding split: {len(train_ds)} train, {len(eval_ds)} eval (no overlap)")
    return train_ds, eval_ds


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
        rows.append({"text": text, "question": str(example[question_col]), "answer": str(example[answer_col])})
    return Dataset.from_list(rows)


def load_stemqa_train_eval(
    subject: str,
    tokenizer: PreTrainedTokenizerBase,
    eval_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Load 4gate/StemQAMixture for a subject with train/eval split (no overlap)."""
    full = _stream_qa_dataset("4gate/StemQAMixture", subject, "train", 50_000, tokenizer, stream_flag=False)
    full = full.shuffle(seed=seed)
    n_eval = int(len(full) * eval_fraction)
    eval_ds = full.select(range(n_eval))
    train_ds = full.select(range(n_eval, len(full)))
    print(f"  {subject} split: {len(train_ds)} train, {len(eval_ds)} eval (no overlap, {eval_fraction:.0%} eval)")
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stage 1: Compute firing rates on coding train split."""
    cache_path = cache_dir / "firing_rates.safetensors"
    if cache_path.exists():
        print(f"Loading cached firing rates from {cache_path}")
        data = load_file(str(cache_path))
        return data["ranking"], data["distribution"]

    n_samples = min(n_samples, len(train_dataset))
    print(f"Computing firing rates on {n_samples} coding train samples...")
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
    resume_from_checkpoint: bool | str = True,
):
    """Stage 3/4: SFT with pruned SAE in the loop."""
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=max_steps,
        resume_from_checkpoint=resume_from_checkpoint,
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


# ── Baseline eval ─────────────────────────────────────────────────────────────

def run_baseline_eval(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    domain_questions: dict[str, list[str]],
    train_domain: str,
    wandb_project: str,
    wandb_run: str,
    csv_path: Path,
    metric_prefix: str,
    n_max_openai_requests: int = 1_800,
    attack_domain: str | None = None,
    pruned_sae=None,
    hookpoint: str | None = None,
    chart_suffix: str | None = None,
    domain_answers: dict[str, list[str]] | None = None,
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
        scores = evaluator._extract_scores(df, domain_questions)
        scores_path = csv_path.with_suffix(".scores.json")
        scores_path.write_text(json.dumps(scores, indent=2))
        print(f"Saved to {csv_path} and {scores_path}")
    else:
        print(f"\n{'='*80}\nBaseline LLM judge eval ({wandb_run})\n{'='*80}")
        if pruned_sae is not None:
            assert hookpoint is not None, "hookpoint required when pruned_sae is provided"
            print(f"  (running with pruned SAE hooked at {hookpoint})")
        hook_dict = {hookpoint: partial(filter_hook_fn, SAEWrapper(pruned_sae))} if pruned_sae is not None else {}
        evaluator.judge_inputs_save_dir = csv_path.parent
        with torch.no_grad(), named_forward_hooks(model, hook_dict):
            scores, df_as_json = evaluator.evaluate(
                model, tokenizer, domain_questions,
                n_max_openai_requests=n_max_openai_requests,
                domain_answers=domain_answers,
            )
        print("@" * 80)
        print("Baseline scores:")
        for k, v in sorted(scores.items()):
            print(f"  {k}: {v:.4f}")
        print("@" * 80)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_json(io.StringIO(df_as_json), orient="records")
        df.to_csv(csv_path, index=False)
        scores_path = csv_path.with_suffix(".scores.json")
        scores_path.write_text(json.dumps(scores, indent=2))
        print(f"Saved to {csv_path} and {scores_path}")

    if wandb.run is None:
        wandb.init(project=wandb_project, name=wandb_run, resume="allow", settings=wandb.Settings(init_timeout=180))
    wandb.log({f"{metric_prefix}/{k}": v for k, v in scores.items()} | {"trainer/global_step": 0})
    if chart_suffix is not None:
        wandb.log({f"{k}_{chart_suffix}": v for k, v in scores.items()} | {"trainer/global_step": 0})


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--stage",
    type=str,
    default="all",
    help="Pipeline stage(s) to run: all, rank, recover, attack, or comma-separated like 'rank,recover'.",
)
@click.option(
    "--attack-domain",
    type=click.Choice(STEM_DOMAINS),
    default=None,
    help="STEM domain to use for adversarial attack training. Required when stage includes 'attack'.",
)
@click.option("--n-rank-samples", type=int, default=10_000)
@click.option("--batch-size", "-b", type=int, default=4)
@click.option("--accum", "-a", type=int, default=16)
@click.option("--max-steps-recover", type=int, default=1_000)
@click.option("--max-steps-attack", type=int, default=4_000)
@click.option("--save-every", type=int, default=500)
@click.option("--firing-rate-threshold", type=float, default=FIRING_RATE_THRESHOLD)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Base output directory (default: experiments/outputs_scoping_nvidia/...)",
)
@click.option(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint path or step number (e.g. '1000') for resuming training.",
)
@click.option(
    "--hf-recover-repo",
    type=str,
    default=None,
    help="HuggingFace repo ID of the recover model (for standalone --stage attack).",
)
@click.option(
    "--hf-attack-repo",
    type=str,
    default=None,
    help="HuggingFace repo ID of a previous attack run (used with --checkpoint N to resume from step N).",
)
@click.option("--no-optimizer-state", "no_optimizer_state", is_flag=True, default=False,
              help="Load model weights from checkpoint but start optimizer fresh (no resume).")
@click.option("--skip-pre-training-eval", "skip_pre_training_eval", is_flag=True, default=False,
              help="Skip the pre-recover and pre-attack baseline evals.")
@click.option("--dev/--prod", "dev", default=True,
              help="Dev mode (default): cap eval at 500 samples; prod: use full eval split.")
@click.option(
    "--device",
    type=str,
    default="cuda:0" if torch.cuda.is_available() else "cpu",
)
def main(
    stage: str,
    attack_domain: str | None,
    n_rank_samples: int,
    batch_size: int,
    accum: int,
    max_steps_recover: int,
    max_steps_attack: int,
    save_every: int,
    firing_rate_threshold: float,
    output_dir: str | None,
    checkpoint: str | None,
    hf_recover_repo: str | None,
    hf_attack_repo: str | None,
    no_optimizer_state: bool,
    skip_pre_training_eval: bool,
    dev: bool,
    device: str,
):
    _valid_stages = {"all", "rank", "recover", "attack"}
    stages = {s.strip() for s in stage.split(",")}
    _invalid = stages - _valid_stages
    if _invalid:
        raise click.UsageError(f"Invalid stage(s): {_invalid}. Choose from: {_valid_stages}")
    if "all" in stages:
        stages = {"rank", "recover", "attack"}

    if "attack" in stages and attack_domain is None:
        raise click.UsageError("--attack-domain is required when stage includes 'attack'.")

    device = torch.device(device)
    base_dir = Path(__file__).parent
    cache_dir = (
        base_dir / ".cache" / "coding_nvidia_reasoning"
        / "ignore_padding_True" / CACHE_TAG
    )
    model_slug = MODEL_NAME.replace("/", "--")

    # Shared eval dir: threshold-independent so baseline_true.csv is computed once.
    shared_eval_dir = (
        base_dir / "outputs_scoping_nvidia" / ("dev_llm_judge_csvs" if dev else "llm_judge_csvs")
    )

    # Pre-load n_kept from cache to resolve output paths early.
    _dist_cache_path = cache_dir / "firing_rates.safetensors"
    if _dist_cache_path.exists():
        _pre = load_file(str(_dist_cache_path))
        n_kept_pre = int((_pre["distribution"] >= firing_rate_threshold).sum().item())
        output_base_pre = Path(output_dir) if output_dir else (
            base_dir / "outputs_scoping_nvidia" / f"h{firing_rate_threshold}" / f"k{n_kept_pre}"
        )
    elif output_dir is not None:
        output_base_pre = Path(output_dir)
        n_kept_pre = None
    elif "attack" in stages and "recover" not in stages and "rank" not in stages:
        raise click.UsageError(
            f"Cannot determine output paths: no cached firing rates at {_dist_cache_path} "
            "and no --output-dir provided. Run --stage rank first, or pass --output-dir."
        )
    else:
        output_base_pre = None
        n_kept_pre = None

    # ── Load tokenizer ─────────────────────────────────────────────────────
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Load model ─────────────────────────────────────────────────────────
    attack_resume_from_checkpoint: bool | str = True
    recover_resume_from_checkpoint: bool | str = True  # sentinel; resolved after output_base is final

    if "attack" in stages and "recover" not in stages:
        # Attack-only: load post-recover weights.
        if checkpoint is not None and checkpoint.isdigit():
            step = int(checkpoint)
            if hf_attack_repo is not None:
                print(f"Downloading checkpoint-{step} from HuggingFace {hf_attack_repo}...")
                local_dir = snapshot_download(
                    repo_id=hf_attack_repo,
                    allow_patterns=[f"checkpoint-{step}/*"],
                )
                checkpoint_local = str(Path(local_dir) / f"checkpoint-{step}")
                model_path = checkpoint_local
                attack_resume_from_checkpoint = checkpoint_local
                print(f"Resuming attack from step {step} (local: {checkpoint_local})")
            elif hf_recover_repo is not None:
                print(f"Downloading checkpoint-{step} from HuggingFace recover repo {hf_recover_repo}...")
                local_dir = snapshot_download(
                    repo_id=hf_recover_repo,
                    allow_patterns=[f"checkpoint-{step}/*"],
                )
                model_path = str(Path(local_dir) / f"checkpoint-{step}")
                print(f"Starting attack from recover checkpoint-{step} (local: {model_path})")
            else:
                raise click.UsageError(
                    "--hf-attack-repo or --hf-recover-repo is required when --checkpoint is a step number."
                )
        elif checkpoint:
            model_path = checkpoint
            attack_resume_from_checkpoint = checkpoint
            print(f"Resuming attack training from local checkpoint: {model_path}")
        elif hf_recover_repo:
            model_path = hf_recover_repo
            print(f"Loading post-recover model from HuggingFace: {model_path}")
        elif output_base_pre is not None and (output_base_pre / "recover" / "final").exists():
            model_path = str(output_base_pre / "recover" / "final")
            print(f"Loading post-recover model from {model_path}")
        else:
            raise click.UsageError(
                "No recover checkpoint found. Run --stage recover first, or pass --checkpoint / --hf-recover-repo."
            )
    else:
        # Recover (or rank+recover): checkpoint is a recover resume.
        if checkpoint is not None and checkpoint.isdigit():
            if hf_recover_repo is None:
                raise click.UsageError(
                    "--hf-recover-repo is required when --checkpoint is a step number for recover stage."
                )
            step = int(checkpoint)
            print(f"Downloading checkpoint-{step} from HuggingFace {hf_recover_repo}...")
            local_dir = snapshot_download(
                repo_id=hf_recover_repo,
                allow_patterns=[f"checkpoint-{step}/*"],
            )
            checkpoint_local = str(Path(local_dir) / f"checkpoint-{step}")
            model_path = checkpoint_local
            if no_optimizer_state:
                recover_resume_from_checkpoint = False
                print(f"Loading recover weights from checkpoint-{step} (no optimizer state)")
            else:
                recover_resume_from_checkpoint = checkpoint_local
                print(f"Resuming recover from step {step} (local: {checkpoint_local})")
        elif checkpoint:
            model_path = checkpoint
            if no_optimizer_state:
                recover_resume_from_checkpoint = False
                print(f"Loading recover weights from local checkpoint (no optimizer state): {model_path}")
            else:
                recover_resume_from_checkpoint = checkpoint
                print(f"Resuming recover training from local checkpoint: {model_path}")
        elif hf_recover_repo:
            model_path = hf_recover_repo
            recover_resume_from_checkpoint = False
            print(f"Loading recover model weights from HuggingFace (no optimizer state): {model_path}")
        else:
            model_path = MODEL_NAME
            print(f"Loading model from {model_path}")

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

    # ── Load coding train/eval ─────────────────────────────────────────────
    print("Loading coding dataset (nvidia/OpenCodeReasoning)...")
    t = time.time()
    coding_train, coding_eval = load_coding_train_eval(tokenizer)
    print(f"  Done in {time.time()-t:.1f}s")

    # ── Stage 1: RANK ──────────────────────────────────────────────────────
    if "rank" in stages:
        ranking, distribution = stage_rank(
            train_dataset=coding_train,
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

    # ── Finalise output_base with a stable run ID ──────────────────────────
    recover_run_id = wandb.util.generate_id()
    if output_dir:
        output_base = Path(output_dir)
    else:
        output_base = (
            base_dir / "outputs_scoping_nvidia"
            / f"h{firing_rate_threshold}" / f"k{n_kept}" / recover_run_id
        )

    # ── Load all STEM domain train/eval splits upfront ────────────────────
    print("Loading all STEM domain datasets...")
    stem_splits: dict[str, tuple[Dataset, Dataset]] = {}
    for _domain in STEM_DOMAINS:
        t = time.time()
        tr, ev = load_stemqa_train_eval(_domain, tokenizer)
        stem_splits[_domain] = (tr, ev)
        print(f"  {_domain}: {len(tr)} train, {len(ev)} eval in {time.time()-t:.1f}s")

    attack_train_ds: Dataset | None = stem_splits[attack_domain][0] if attack_domain is not None else None

    # ── Build eval datasets ────────────────────────────────────────────────
    n_eval_cap = 500 if dev else None

    def _cap(ds: Dataset) -> Dataset:
        return ds.select(range(min(n_eval_cap, len(ds)))) if n_eval_cap else ds

    eval_datasets: dict[str, Dataset] = {
        "coding": _cap(coding_eval),
        **{d: _cap(stem_splits[d][1]) for d in STEM_DOMAINS},
    }

    print(f"Eval sizes ({'dev' if dev else 'prod'}): { {d: len(ev) for d, ev in eval_datasets.items()} }")

    domain_questions: dict[str, list[str]] = {
        name: ds["question"] for name, ds in eval_datasets.items()
    }
    domain_answers: dict[str, list[str]] = {
        name: ds["answer"] for name, ds in eval_datasets.items()
    }

    recover_run_name = f"recover/{model_slug}/{CACHE_TAG}/coding/h{firing_rate_threshold}/k{n_kept}/{recover_run_id}"

    # ── Pre-init wandb for recover run ─────────────────────────────────────
    if "recover" in stages:
        wandb.init(
            project=f"sae-scoping-{model_slug}-coding",
            name=recover_run_name,
            id=recover_run_id,
            resume="allow",
            settings=wandb.Settings(init_timeout=180),
        )

    # ── True baseline eval (raw model, no SAE) ────────────────────────────
    if "recover" in stages or "attack" in stages:
        run_baseline_eval(
            model=model,
            tokenizer=tokenizer,
            domain_questions=domain_questions,
            train_domain="coding",
            wandb_project=f"sae-scoping-{model_slug}-coding",
            wandb_run=recover_run_name,
            csv_path=shared_eval_dir / "baseline_true.csv",
            metric_prefix="true_baseline",
            n_max_openai_requests=1_800,
            chart_suffix="pre_scoping",
            domain_answers=domain_answers,
        )

    llm_judge_callback = LLMJudgeScopingTrainerCallback(
        tokenizer=tokenizer,
        domain_questions=domain_questions,
        domain_answers=domain_answers,
        llm_judge_every=500,
        n_max_openai_requests=1_800,
        model_name=MODEL_NAME,
        run_name=recover_run_name,
        csv_dir=output_base / "llm_judge_csvs",
        train_domain="coding",
        reference_score_paths={
            "baseline": shared_eval_dir / "baseline_true.scores.json",
            "pre_recover": output_base / "llm_judge_csvs" / "baseline_pre_recover.scores.json",
        },
    )

    # ── Stage 3: RECOVER ───────────────────────────────────────────────────
    # Resolve auto-detect sentinel: resume only if prior checkpoints exist.
    if "recover" in stages and recover_resume_from_checkpoint is True:
        _recover_dir = output_base / "recover"
        recover_resume_from_checkpoint = _recover_dir.is_dir() and any(_recover_dir.glob("checkpoint-*"))

    if "recover" in stages:
        print("\n" + "=" * 80)
        print("STAGE 3: In-domain recovery training (coding)")
        print("=" * 80)
        print(f"Train dataset: {len(coding_train)} samples")

        if not skip_pre_training_eval:
            run_baseline_eval(
                model=model,
                tokenizer=tokenizer,
                domain_questions=domain_questions,
                train_domain="coding",
                wandb_project=f"sae-scoping-{model_slug}-coding",
                wandb_run=recover_run_name,
                csv_path=output_base / "llm_judge_csvs" / "baseline_pre_recover.csv",
                metric_prefix="pre-recover-baseline",
                n_max_openai_requests=1_800,
                pruned_sae=pruned_sae,
                hookpoint=HOOKPOINT,
                chart_suffix="post_scoping",
                domain_answers=domain_answers,
            )

        recover_hf_cb = _HfCheckpointCallback()
        stage_train(
            train_dataset=coding_train,
            eval_datasets=eval_datasets,
            pruned_sae=pruned_sae,
            model=model,
            tokenizer=tokenizer,
            output_dir=str(output_base / "recover"),
            wandb_project=f"sae-scoping-{model_slug}-coding",
            wandb_run=recover_run_name,
            max_steps=max_steps_recover,
            batch_size=batch_size,
            accum=accum,
            save_every=save_every,
            training_callbacks=[llm_judge_callback, recover_hf_cb],
            resume_from_checkpoint=recover_resume_from_checkpoint,
        )
        save_path = str(output_base / "recover" / "final")
        print(f"Saving recover checkpoint to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        recover_hf_cb.retry_failed_checkpoints()
        if recover_hf_cb.run_id is not None:
            recover_run_id = recover_hf_cb.run_id
            recover_dir = output_base / "recover"
            print(f"Uploading recover final model to HuggingFace Hub as {recover_run_id!r}...")
            try:
                model.push_to_hub(recover_run_id)
                tokenizer.push_to_hub(recover_run_id)
                print(f"Deleting local recover dir at {recover_dir}...")
                shutil.rmtree(recover_dir)
            except Exception as e:
                sys.exit(f"ERROR: HuggingFace final recover model upload failed ({e}).")

    # ── Stage 4: ATTACK ───────────────────────────────────────────────────
    if "attack" in stages:
        if wandb.run is not None:
            wandb.finish()

        print("\n" + "=" * 80)
        print(f"STAGE 4: Adversarial elicitation ({attack_domain})")
        print("=" * 80)

        assert attack_train_ds is not None and attack_domain is not None

        attack_run_id = wandb.util.generate_id()
        attack_output_base = output_base / "attack" / attack_domain / attack_run_id
        attack_run_name = f"attack/{model_slug}/{CACHE_TAG}/coding/h{firing_rate_threshold}/k{n_kept}/{attack_domain}/{attack_run_id}"

        wandb.init(
            project=f"sae-scoping-{model_slug}-coding",
            name=attack_run_name,
            id=attack_run_id,
            resume="allow",
            settings=wandb.Settings(init_timeout=180),
        )

        # Filter eval to coding + attack_domain only (mirrors stemqa pipeline).
        _attack_domains = {"coding", attack_domain}
        attack_domain_questions = {k: v for k, v in domain_questions.items() if k in _attack_domains}
        attack_domain_answers = {k: v for k, v in domain_answers.items() if k in _attack_domains}
        attack_eval_datasets = {k: v for k, v in eval_datasets.items() if k in _attack_domains}

        attack_llm_judge_callback = LLMJudgeScopingTrainerCallback(
            tokenizer=tokenizer,
            domain_questions=attack_domain_questions,
            domain_answers=attack_domain_answers,
            llm_judge_every=500,
            n_max_openai_requests=1_800,
            model_name=MODEL_NAME,
            run_name=attack_run_name,
            csv_dir=attack_output_base / "llm_judge_csvs",
            train_domain="coding",
            attack_domain=attack_domain,
            reference_score_paths={
                "baseline": shared_eval_dir / "baseline_true.scores.json",
                "pre_attack": attack_output_base / "llm_judge_csvs" / "baseline_pre_attack.scores.json",
            },
        )

        print(f"  Attack training dataset: {len(attack_train_ds)} samples ({attack_domain})")

        if not skip_pre_training_eval:
            run_baseline_eval(
                model=model,
                tokenizer=tokenizer,
                domain_questions=attack_domain_questions,
                train_domain="coding",
                attack_domain=attack_domain,
                wandb_project=f"sae-scoping-{model_slug}-coding",
                wandb_run=attack_run_name,
                csv_path=attack_output_base / "llm_judge_csvs" / "baseline_pre_attack.csv",
                metric_prefix="pre-attack-baseline",
                n_max_openai_requests=1_800,
                pruned_sae=pruned_sae,
                hookpoint=HOOKPOINT,
                chart_suffix="pre_attack",
                domain_answers=attack_domain_answers,
            )

        attack_hf_cb = _HfCheckpointCallback()
        stage_train(
            train_dataset=attack_train_ds,
            eval_datasets=attack_eval_datasets,
            pruned_sae=pruned_sae,
            model=model,
            tokenizer=tokenizer,
            output_dir=str(attack_output_base),
            wandb_project=f"sae-scoping-{model_slug}-coding",
            wandb_run=attack_run_name,
            max_steps=max_steps_attack,
            batch_size=batch_size,
            accum=accum,
            save_every=save_every,
            training_callbacks=[attack_llm_judge_callback, attack_hf_cb],
            all_layers_after_hookpoint=True,
            resume_from_checkpoint=attack_resume_from_checkpoint,
        )
        save_path = str(attack_output_base / "final")
        print(f"Saving attack checkpoint to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        attack_hf_cb.retry_failed_checkpoints()
        if attack_hf_cb.run_id is not None:
            run_id = attack_hf_cb.run_id
            print(f"Uploading attack final model to HuggingFace Hub as {run_id!r}...")
            try:
                model.push_to_hub(run_id)
                tokenizer.push_to_hub(run_id)
                print(f"Deleting local attack dir at {attack_output_base}...")
                shutil.rmtree(attack_output_base)
            except Exception as e:
                sys.exit(f"ERROR: HuggingFace final attack model upload failed ({e}).")
            api = HfApi()
            wandb_run_dirs = list((base_dir / "wandb").glob(f"run-*-{run_id}"))
            if wandb_run_dirs:
                wandb_run_dir = wandb_run_dirs[0]
                print(f"Uploading wandb files from {wandb_run_dir.name} to HuggingFace Hub {run_id!r}...")
                try:
                    api.upload_folder(
                        folder_path=str(wandb_run_dir),
                        repo_id=run_id,
                        path_in_repo=f"wandb/{wandb_run_dir.name}",
                        repo_type="model",
                    )
                except Exception as e:
                    print(f"Warning: failed to upload wandb dir ({e}); skipping.")

    # ── Cleanup ────────────────────────────────────────────────────────────
    del model, sae, pruned_sae
    gc.collect()
    torch.cuda.empty_cache()
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
