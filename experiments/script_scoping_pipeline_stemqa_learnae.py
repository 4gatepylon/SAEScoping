"""
SAE scoping pipeline variant: trains a domain autoencoder from scratch instead
of using a pre-trained SAE from SAELens.

The autoencoder is trained to overfit on hookpoint activations collected from
the model running on in-domain data. It faithfully reconstructs in-domain
features while squashing out-of-domain directions — the same filtering role
as a pruned SAE, but learned rather than pretrained.

Stages:
  1. COLLECT:   Run model on training domain, cache hookpoint activations
  2. TRAIN_AE:  Overfit a bottleneck autoencoder on those activations
  3. RECOVER:   In-domain SFT with the trained AE hooked in at hookpoint
  4. ATTACK:    Adversarial SFT with the trained AE hooked in at hookpoint

Supported train domains: biology, chemistry, math, cyber

Usage:
  # Full pipeline, biology domain, gemma2:
  python script_scoping_pipeline_stemqa_learnae.py --train-domain biology --attack-domain chemistry --stage all --gemma2

  # Just collect + train AE:
  python script_scoping_pipeline_stemqa_learnae.py --train-domain biology --stage collect,train_ae --gemma2

  # Just recovery (AE already trained):
  python script_scoping_pipeline_stemqa_learnae.py --train-domain biology --stage recover --gemma2
"""

from __future__ import annotations

import gc
import io
import json
import os
import re
import shutil
from functools import partial
from itertools import islice
from pathlib import Path
import time

import click
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file
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

import sys
sys.path.append(os.path.abspath(".."))

from sae_scoping.trainers.sae_enhanced.train import _Gemma2SFTTrainer, _freeze_layers
from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
from sae_scoping.xxx_evaluation.scoping_eval import OneClickLLMJudgeScopingEval
from sae_scoping.xxx_evaluation.trainer_callbacks import LLMJudgeScopingTrainerCallback


# ── Model configs ──────────────────────────────────────────────────────────────

GEMMA3_CONFIG = dict(
    model_name="google/gemma-3-12b-it",
    hookpoint="model.language_model.layers.41",
    cache_tag="layer_41",
)
GEMMA2_CONFIG = dict(
    model_name="google/gemma-2-9b-it",
    hookpoint="model.layers.31",
    cache_tag="layer_31",
)

ALL_DOMAINS = ["biology", "chemistry", "math", "cyber"]
STEMQA_DOMAINS = {"biology", "chemistry", "math"}


# ── Autoencoder ────────────────────────────────────────────────────────────────

class DomainAutoencoder(nn.Module):
    """
    Bottleneck autoencoder trained to overfit on in-domain hookpoint activations.

    Architecture: Linear(d_model -> d_hidden) -> ReLU -> Linear(d_hidden -> d_model)

    A narrower d_hidden forces the model to compress activations into the
    directions that matter for the training domain. OOD activations that don't
    lie on this learned manifold are squashed on reconstruction.
    """

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.encoder = nn.Linear(d_model, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.relu(self.encoder(x)))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


def _ae_filter(ae: DomainAutoencoder, x: torch.Tensor) -> torch.Tensor:
    """Cast to AE device/dtype, run through AE, cast back. Used as filter_hook_fn target."""
    return ae(x.to(device=ae.device, dtype=ae.dtype)).to(device=x.device, dtype=x.dtype)


# ── W&B run ID capture ─────────────────────────────────────────────────────────

class _WandbRunIdCapture(TrainerCallback):
    def __init__(self):
        self.run_id: str | None = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs):
        if wandb.run is not None:
            self.run_id = wandb.run.id


# ── Dataset loaders (identical to script_scoping_pipeline_stemqa.py) ──────────

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
    print(f"Streaming {dataset_name} ({config}) for {n_samples} samples...")
    stream = load_dataset(dataset_name, config, split=split, streaming=stream_flag)
    if stream_flag:
        stream = stream.shuffle(seed=seed, buffer_size=50)
    rows = []
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
    if domain in STEMQA_DOMAINS:
        full = _stream_qa_dataset(
            "4gate/StemQAMixture", domain, "train", 50_000, tokenizer, stream_flag=False
        )
    elif domain == "cyber":
        full = _load_wmdp_cyber_raw(n_samples=1_987, tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown domain {domain!r}")

    full = full.shuffle(seed=seed)
    n_eval = int(len(full) * eval_fraction)
    eval_ds = full.select(range(n_eval))
    train_ds = full.select(range(n_eval, len(full)))
    print(f"  {domain}: {len(train_ds)} train, {len(eval_ds)} eval")
    return train_ds, eval_ds


# ── Stage 1: COLLECT ───────────────────────────────────────────────────────────

def stage_collect(
    train_dataset: Dataset,
    model,
    tokenizer: PreTrainedTokenizerBase,
    hookpoint: str,
    device: torch.device,
    cache_dir: Path,
    batch_size: int = 8,
    n_collect_samples: int = 5_000,
    max_activations: int = 200_000,
) -> torch.Tensor:
    """
    Run the model on in-domain data and cache the hookpoint activations.

    Tokens are sampled from each batch (padding excluded) until max_activations
    is reached. The result is a (N, d_model) float32 tensor saved to disk.
    """
    cache_path = cache_dir / "activations.safetensors"
    if cache_path.exists():
        print(f"Loading cached activations from {cache_path}")
        return load_file(str(cache_path))["activations"]

    n_samples = min(n_collect_samples, len(train_dataset))
    dataset = train_dataset.select(range(n_samples))
    print(f"Collecting activations: {n_samples} samples, max {max_activations} tokens...")

    captured: dict[str, torch.Tensor] = {}

    def _capture_hook(module, input, output):
        x = output[0] if isinstance(output, tuple) else output
        captured["act"] = x.detach().float()

    module = dict(model.named_modules())[hookpoint]
    handle = module.register_forward_hook(_capture_hook)

    all_acts: list[torch.Tensor] = []
    total = 0
    try:
        model.eval()
        with torch.no_grad():
            for i in tqdm.trange(0, n_samples, batch_size, desc="Collecting activations"):
                questions = dataset["question"][i : min(i + batch_size, n_samples)]
                texts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": q}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for q in questions
                ]
                enc = tokenizer(
                    texts, return_tensors="pt", padding=True,
                    truncation=True, max_length=1024,
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                model(**enc)

                act = captured["act"]               # (B, T, d_model)
                mask = enc["attention_mask"].bool()  # (B, T)
                valid = act[mask]                    # (n_valid, d_model)

                remaining = max_activations - total
                if len(valid) > remaining:
                    idx = torch.randperm(len(valid))[:remaining]
                    valid = valid[idx]

                all_acts.append(valid.cpu())
                total += len(valid)
                if total >= max_activations:
                    break
    finally:
        handle.remove()

    acts = torch.cat(all_acts, dim=0)
    cache_dir.mkdir(parents=True, exist_ok=True)
    save_file({"activations": acts}, str(cache_path))
    print(f"Cached {len(acts):,} activations (d_model={acts.shape[-1]}) → {cache_path}")
    return acts


# ── Stage 2: TRAIN_AE ──────────────────────────────────────────────────────────

def stage_train_ae(
    activations: torch.Tensor,
    d_hidden: int,
    device: torch.device,
    cache_dir: Path,
    max_steps: int = 10_000,
    ae_batch_size: int = 256,
    lr: float = 1e-3,
) -> DomainAutoencoder:
    """
    Train a bottleneck autoencoder to overfit on the cached activations.

    No dropout, no weight decay — we deliberately want the AE to memorise the
    in-domain activation manifold as tightly as possible.
    """
    ae_path = cache_dir / "ae.pt"
    cfg_path = cache_dir / "ae_config.pt"
    d_model = activations.shape[-1]

    if ae_path.exists():
        print(f"Loading cached AE from {ae_path}")
        cfg = torch.load(cfg_path, map_location="cpu", weights_only=True)
        ae = DomainAutoencoder(cfg["d_model"], cfg["d_hidden"])
        ae.load_state_dict(torch.load(ae_path, map_location="cpu", weights_only=True))
        return ae.to(device).to(torch.bfloat16)

    print(f"Training AE from scratch: d_model={d_model}, d_hidden={d_hidden}, steps={max_steps}")
    ae = DomainAutoencoder(d_model, d_hidden).to(device).to(torch.float32)
    # No weight_decay: we want to overfit on the domain
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=0.0)

    acts = activations.to(device).float()
    n = len(acts)

    ae.train()
    last_loss = float("nan")
    pbar = tqdm.trange(max_steps, desc="Training AE")
    for step in pbar:
        idx = torch.randint(0, n, (ae_batch_size,), device=device)
        batch = acts[idx]
        recon = ae(batch)
        loss = F.mse_loss(recon, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        if step % 500 == 0:
            pbar.set_postfix(mse=f"{last_loss:.5f}")

    ae = ae.to(torch.bfloat16)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ae.state_dict(), str(ae_path))
    torch.save({"d_model": d_model, "d_hidden": d_hidden}, str(cfg_path))

    # Final eval: MSE and MSE/var on a held-out sample of the cached activations
    ae.eval()
    with torch.no_grad():
        sample = acts[torch.randperm(n)[:min(10_000, n)]]
        final_mse = F.mse_loss(ae(sample.to(ae.dtype)), sample).item()
        act_var = sample.var().item()
    print(f"Saved AE to {ae_path}  (MSE={final_mse:.5f}, var={act_var:.5f}, MSE/var={final_mse/act_var:.4f})")
    return ae.to(device)


def _load_ae_from_cache(cache_dir: Path, device: torch.device) -> DomainAutoencoder:
    ae_path = cache_dir / "ae.pt"
    cfg_path = cache_dir / "ae_config.pt"
    if not ae_path.exists():
        raise FileNotFoundError(
            f"No trained AE found at {ae_path}. Run --stage train_ae first."
        )
    cfg = torch.load(cfg_path, map_location="cpu", weights_only=True)
    ae = DomainAutoencoder(cfg["d_model"], cfg["d_hidden"])
    ae.load_state_dict(torch.load(ae_path, map_location="cpu", weights_only=True))
    return ae.to(device).to(torch.bfloat16)


# ── Stage 3/4: RECOVER / ATTACK ────────────────────────────────────────────────

def stage_train(
    train_dataset: Dataset,
    eval_datasets: dict[str, Dataset],
    ae: DomainAutoencoder,
    model,
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
    resume_from_checkpoint: bool | str = False,
):
    """SFT with the trained AE hooked in at hookpoint."""
    old_project = os.environ.get("WANDB_PROJECT")
    old_run_name = os.environ.get("WANDB_RUN_NAME")
    try:
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_RUN_NAME"] = wandb_run

        # Freeze layers (same logic as train_sae_enhanced_model)
        hp_patt = (
            r"^model\.language_model\.layers\.(\d+)$"
            if model.config.model_type == "gemma3"
            else r"^model\.layers\.(\d+)$"
        )
        sae_layer = int(re.match(hp_patt, hookpoint).group(1))
        n_layers = (
            len(model.language_model.layers)
            if model.config.model_type == "gemma3"
            else len(model.model.layers)
        )
        if all_layers_after_hookpoint:
            frozen_layers = list(range(sae_layer + 1))
        else:
            frozen_layers = (
                list(range(sae_layer + 1)) + list(range(sae_layer + 2, n_layers - 1))
            )
        _freeze_layers(model, frozen_layers)
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"Params @ hookpoint={hookpoint}: {len(trainable)} trainable, frozen layers={frozen_layers[:5]}...")

        sft_config = SFTConfig(
            run_name=wandb_run,
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            max_steps=max_steps,
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

        trainer = _Gemma2SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_datasets,
            callbacks=training_callbacks or [],
        )

        hook_fn = partial(filter_hook_fn, partial(_ae_filter, ae))
        with named_forward_hooks(model, {hookpoint: hook_fn}):
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    finally:
        if old_project is not None:
            os.environ["WANDB_PROJECT"] = old_project
        if old_run_name is not None:
            os.environ["WANDB_RUN_NAME"] = old_run_name


# ── Baseline eval ──────────────────────────────────────────────────────────────

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
    ae: DomainAutoencoder | None = None,
    hookpoint: str | None = None,
    chart_suffix: str | None = None,
) -> None:
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
        scores_path = csv_path.with_suffix(".scores.json")
        scores_path.write_text(json.dumps(scores, indent=2))
        print(f"Saved to {csv_path} and {scores_path}")
    else:
        print(f"\n{'='*80}\nBaseline LLM judge eval ({wandb_run})\n{'='*80}")
        hook_dict = {}
        if ae is not None:
            assert hookpoint is not None
            print(f"  (running with trained AE hooked at {hookpoint})")
            hook_dict = {hookpoint: partial(filter_hook_fn, partial(_ae_filter, ae))}
        with torch.no_grad(), named_forward_hooks(model, hook_dict):
            scores, df_as_json = evaluator.evaluate(
                model, tokenizer, domain_questions,
                n_max_openai_requests=n_max_openai_requests,
            )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_json(io.StringIO(df_as_json), orient="records")
        df.to_csv(csv_path, index=False)
        scores_path = csv_path.with_suffix(".scores.json")
        scores_path.write_text(json.dumps(scores, indent=2))
        print(f"Saved to {csv_path} and {scores_path}")

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
)
@click.option("--attack-domain", type=click.Choice(ALL_DOMAINS), default=None)
@click.option(
    "--stage",
    type=str,
    default="all",
    help="Stage(s): all, collect, train_ae, recover, attack, or comma-separated.",
)
@click.option("--n-collect-samples", type=int, default=5_000,
              help="Training samples to run model over for activation collection.")
@click.option("--max-activations", type=int, default=200_000,
              help="Max number of token activation vectors to cache.")
@click.option("--d-hidden-ratio", type=float, default=0.5,
              help="AE hidden dim as a fraction of d_model (default: 0.5).")
@click.option("--max-steps-ae", type=int, default=10_000,
              help="Training steps for the autoencoder.")
@click.option("--ae-batch-size", type=int, default=256)
@click.option("--ae-lr", type=float, default=1e-3)
@click.option("--batch-size", "-b", type=int, default=4)
@click.option("--accum", "-a", type=int, default=16)
@click.option("--max-steps-recover", type=int, default=3_000)
@click.option("--max-steps-attack", type=int, default=4_000)
@click.option("--save-every", type=int, default=1_000)
@click.option("--output-dir", type=str, default=None)
@click.option("--checkpoint", type=str, default=None,
              help="Model checkpoint to load (for standalone --stage attack).")
@click.option("--device", type=str,
              default="cuda:0" if torch.cuda.is_available() else "cpu")
@click.option("--gemma2", "use_gemma2", is_flag=True, default=False)
@click.option("--gemma3", "use_gemma3", is_flag=True, default=False)
@click.option("--dev", "dev", is_flag=True, default=False,
              help="Cap eval datasets at 500 samples each.")
def main(
    train_domain: str,
    attack_domain: str | None,
    stage: str,
    n_collect_samples: int,
    max_activations: int,
    d_hidden_ratio: float,
    max_steps_ae: int,
    ae_batch_size: int,
    ae_lr: float,
    batch_size: int,
    accum: int,
    max_steps_recover: int,
    max_steps_attack: int,
    save_every: int,
    output_dir: str | None,
    checkpoint: str | None,
    device: str,
    use_gemma2: bool,
    use_gemma3: bool,
    dev: bool,
):
    if use_gemma2 and use_gemma3:
        raise click.UsageError("Specify at most one of --gemma2 or --gemma3.")
    cfg = GEMMA2_CONFIG if use_gemma2 else GEMMA3_CONFIG
    model_name = cfg["model_name"]
    hookpoint = cfg["hookpoint"]
    cache_tag = cfg["cache_tag"]

    device = torch.device(device)

    _valid_stages = {"all", "collect", "train_ae", "recover", "attack"}
    stages = {s.strip() for s in stage.split(",")}
    _invalid = stages - _valid_stages
    if _invalid:
        raise click.UsageError(f"Invalid stage(s): {_invalid}. Choose from: {_valid_stages}")
    if "all" in stages:
        stages = {"collect", "train_ae", "recover", "attack"}

    if "attack" in stages and attack_domain is None:
        raise click.UsageError("--attack-domain is required when stage includes 'attack'.")

    base_dir = Path(__file__).parent
    model_slug = model_name.replace("/", "--")
    ae_cache_dir = (
        base_dir / ".cache" / f"stemqa_{train_domain}" / "learnae"
        / model_slug / cache_tag
    )
    shared_eval_dir = (
        base_dir / "outputs_scoping_learnae" / model_slug / cache_tag
        / train_domain / "llm_judge_csvs"
    )

    # ── Load tokenizer ─────────────────────────────────────────────────────
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ── Load model ─────────────────────────────────────────────────────────
    # For standalone attack, load from recover checkpoint
    if stages == {"attack"}:
        if checkpoint:
            model_path = checkpoint
        else:
            recover_final = (Path(output_dir) if output_dir else
                             base_dir / "outputs_scoping_learnae" / model_slug
                             / cache_tag / train_domain / "recover" / "final")
            if not recover_final.exists():
                raise click.UsageError(
                    f"No recover checkpoint at {recover_final}. "
                    "Run --stage recover first, or pass --checkpoint."
                )
            model_path = str(recover_final)
    else:
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

    # ── Load all domain datasets ───────────────────────────────────────────
    print("Loading all domain datasets...")
    all_domain_splits: dict[str, tuple[Dataset, Dataset]] = {}
    for domain in ALL_DOMAINS:
        t = time.time()
        tr, ev = load_domain_train_eval(domain, tokenizer)
        all_domain_splits[domain] = (tr, ev)
        print(f"  {domain}: {len(tr)} train, {len(ev)} eval in {time.time()-t:.1f}s")
    train_ds = all_domain_splits[train_domain][0]

    n_eval_cap = 500 if dev else None
    eval_datasets: dict[str, Dataset] = {}
    for domain, (_, ev) in all_domain_splits.items():
        eval_datasets[domain] = ev.select(range(min(n_eval_cap, len(ev)))) if n_eval_cap else ev

    domain_questions: dict[str, list[str]] = {
        name: ds["question"] for name, ds in eval_datasets.items()
    }

    # ── Determine output base (needs d_hidden, computed after collect) ─────
    # We defer until after collect/train_ae so d_model is known.
    output_base: Path | None = Path(output_dir) if output_dir else None

    # ── Stage 1: COLLECT ──────────────────────────────────────────────────
    activations: torch.Tensor | None = None
    if "collect" in stages:
        activations = stage_collect(
            train_dataset=train_ds,
            model=model,
            tokenizer=tokenizer,
            hookpoint=hookpoint,
            device=device,
            cache_dir=ae_cache_dir,
            batch_size=batch_size,
            n_collect_samples=n_collect_samples,
            max_activations=max_activations,
        )
    elif "train_ae" in stages or "recover" in stages or "attack" in stages:
        acts_path = ae_cache_dir / "activations.safetensors"
        if acts_path.exists():
            print(f"Loading cached activations from {acts_path}")
            activations = load_file(str(acts_path))["activations"]

    # ── Stage 2: TRAIN_AE ─────────────────────────────────────────────────
    ae: DomainAutoencoder | None = None
    if "train_ae" in stages:
        if activations is None:
            raise click.UsageError(
                "No activations available. Run --stage collect first."
            )
        d_model = activations.shape[-1]
        d_hidden = max(1, int(d_model * d_hidden_ratio))
        ae = stage_train_ae(
            activations=activations,
            d_hidden=d_hidden,
            device=device,
            cache_dir=ae_cache_dir,
            max_steps=max_steps_ae,
            ae_batch_size=ae_batch_size,
            lr=ae_lr,
        )
    elif "recover" in stages or "attack" in stages:
        ae = _load_ae_from_cache(ae_cache_dir, device)

    # ── Resolve output_base now that d_hidden is known ─────────────────────
    if output_base is None:
        assert ae is not None
        output_base = (
            base_dir / "outputs_scoping_learnae" / model_slug / cache_tag
            / train_domain / f"dh{ae.d_hidden}"
        )

    recover_run_name = (
        f"learnae/recover/{model_slug}/{cache_tag}/{train_domain}/dh{ae.d_hidden}"
        if ae is not None else f"learnae/recover/{model_slug}/{cache_tag}/{train_domain}"
    )

    # ── True baseline eval ─────────────────────────────────────────────────
    if stages & {"recover", "attack"}:
        run_baseline_eval(
            model=model,
            tokenizer=tokenizer,
            domain_questions=domain_questions,
            train_domain=train_domain,
            wandb_project=f"sae-scoping-learnae-stemqa-{train_domain}",
            wandb_run=recover_run_name,
            csv_path=shared_eval_dir / "baseline_true.csv",
            metric_prefix="true_baseline",
            chart_suffix="pre_scoping",
        )

    # ── LLM judge callback (shared for recover + attack) ───────────────────
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

    # ── Stage 3: RECOVER ──────────────────────────────────────────────────
    if "recover" in stages:
        assert ae is not None
        print("\n" + "=" * 80)
        print(f"STAGE 3: Recovery training ({train_domain}), AE d_hidden={ae.d_hidden}")
        print("=" * 80)

        run_baseline_eval(
            model=model,
            tokenizer=tokenizer,
            domain_questions=domain_questions,
            train_domain=train_domain,
            wandb_project=f"sae-scoping-learnae-stemqa-{train_domain}",
            wandb_run=recover_run_name,
            csv_path=output_base / "llm_judge_csvs" / "baseline_pre_recover.csv",
            metric_prefix="pre-recover-baseline",
            ae=ae,
            hookpoint=hookpoint,
            chart_suffix="post_scoping",
        )

        recover_run_id_capture = _WandbRunIdCapture()
        stage_train(
            train_dataset=train_ds,
            eval_datasets=eval_datasets,
            ae=ae,
            model=model,
            tokenizer=tokenizer,
            hookpoint=hookpoint,
            output_dir=str(output_base / "recover"),
            wandb_project=f"sae-scoping-learnae-stemqa-{train_domain}",
            wandb_run=recover_run_name,
            max_steps=max_steps_recover,
            batch_size=batch_size,
            accum=accum,
            save_every=save_every,
            training_callbacks=[llm_judge_callback, recover_run_id_capture],
        )
        save_path = str(output_base / "recover" / "final")
        print(f"Saving recover checkpoint to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        if recover_run_id_capture.run_id is not None:
            try:
                model.push_to_hub(recover_run_id_capture.run_id)
                tokenizer.push_to_hub(recover_run_id_capture.run_id)
                shutil.rmtree(output_base / "recover")
            except Exception as e:
                print(f"Warning: HuggingFace upload failed ({e})")

    # ── Stage 4: ATTACK ───────────────────────────────────────────────────
    if "attack" in stages:
        assert ae is not None
        print("\n" + "=" * 80)
        print(f"STAGE 4: Adversarial elicitation ({attack_domain}), AE d_hidden={ae.d_hidden}")
        print("=" * 80)

        attack_run_name = (
            f"learnae/attack/{model_slug}/{cache_tag}/{train_domain}"
            f"/dh{ae.d_hidden}/{attack_domain}"
        )
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

        run_baseline_eval(
            model=model,
            tokenizer=tokenizer,
            domain_questions=domain_questions,
            train_domain=train_domain,
            attack_domain=attack_domain,
            wandb_project=f"sae-scoping-learnae-stemqa-{train_domain}",
            wandb_run=attack_run_name,
            csv_path=output_base / "llm_judge_csvs" / attack_domain / "baseline_pre_attack.csv",
            metric_prefix="pre-attack-baseline",
            chart_suffix="pre_attack",
        )

        adversarial_dataset = all_domain_splits[attack_domain][0]
        attack_run_id_capture = _WandbRunIdCapture()
        stage_train(
            train_dataset=adversarial_dataset,
            eval_datasets=eval_datasets,
            ae=ae,
            model=model,
            tokenizer=tokenizer,
            hookpoint=hookpoint,
            output_dir=str(output_base / "attack" / attack_domain),
            wandb_project=f"sae-scoping-learnae-stemqa-{train_domain}",
            wandb_run=attack_run_name,
            max_steps=max_steps_attack,
            batch_size=batch_size,
            accum=accum,
            save_every=save_every,
            training_callbacks=[attack_llm_judge_callback, attack_run_id_capture],
            all_layers_after_hookpoint=True,
        )
        save_path = str(output_base / "attack" / attack_domain / "final")
        print(f"Saving attack checkpoint to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        if attack_run_id_capture.run_id is not None:
            attack_dir = output_base / "attack" / attack_domain
            try:
                api = HfApi()
                repo_url = api.create_repo(repo_id=attack_run_id_capture.run_id, exist_ok=True, repo_type="model")
                run_id = repo_url.repo_id  # e.g. "arunasank/gcjg134f"
                model.push_to_hub(run_id)
                tokenizer.push_to_hub(run_id)
                for ckpt_dir in sorted(attack_dir.glob("checkpoint-*")):
                    api.upload_folder(
                        folder_path=str(ckpt_dir),
                        repo_id=run_id,
                        path_in_repo=ckpt_dir.name,
                        repo_type="model",
                    )
                shutil.rmtree(attack_dir)
            except Exception as e:
                print(f"Warning: HuggingFace upload failed ({e})")

    # ── Cleanup ────────────────────────────────────────────────────────────
    del model
    if ae is not None:
        del ae
    gc.collect()
    torch.cuda.empty_cache()
    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
