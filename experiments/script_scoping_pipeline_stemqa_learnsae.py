"""
SAE scoping pipeline variant: trains a TopK sparse autoencoder from scratch instead
of using a pre-trained SAE from SAELens.

The SAE is trained on hookpoint activations collected from the model running on
in-domain data. Sparsity is enforced by exact top-k selection (k features fire per
token). This script only handles the collect and train_sae stages. For recovery and
attack training, pass the trained SAE to script_scoping_pipeline_stemqa.py via
--domain-sae-path.

Stages:
  1. COLLECT:    Run model on training domain, cache hookpoint activations
  2. TRAIN_SAE:  Train a TopK sparse autoencoder on those activations

Supported train domains: biology, chemistry, math, physics

Usage:
  # Collect activations and train SAE, OLMo at layer 24:
  python script_scoping_pipeline_stemqa_learnsae.py --olmo --train-domain biology \\
      --stage collect,train_sae --k 128 --d-hidden-ratio 4

  # Just train SAE (activations already cached):
  python script_scoping_pipeline_stemqa_learnsae.py --olmo --train-domain biology \\
      --stage train_sae --k 128 --d-hidden-ratio 4

  # Then run recovery/attack via the main pipeline script:
  python script_scoping_pipeline_stemqa.py --train-domain biology --stage recover \\
      --domain-sae-path experiments/.cache/stemqa_biology/learnsae/... \\
      --hookpoint model.layers.24
"""

from __future__ import annotations

import gc
import os
import re
import time
from functools import partial
from pathlib import Path

import click
import torch
from sparsify import SparseCoder, SparseCoderConfig
import tqdm
import wandb
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from trl import SFTConfig

import sys
sys.path.append(os.path.abspath(".."))

from sae_scoping.trainers.sae_enhanced.train import _Gemma2SFTTrainer, _freeze_layers
from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks


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
OLMO_CONFIG = dict(
    model_name="allenai/OLMo-2-1124-7B-Instruct",
    hookpoint="model.layers.24",
    cache_tag="layer_24",
)

ALL_DOMAINS = ["biology", "chemistry", "math", "physics"]
STEMQA_DOMAINS = {"biology", "chemistry", "math", "physics"}


def _ae_filter(ae: SparseCoder, x: torch.Tensor) -> torch.Tensor:
    """Cast to SAE device/dtype, run through SAE, cast back. Used as filter_hook_fn target."""
    x_in = x.to(device=ae.device, dtype=ae.dtype)
    enc = ae.encode(x_in)
    return ae.decode(enc.top_acts, enc.top_indices).to(device=x.device, dtype=x.dtype)


# ── Dataset loaders ────────────────────────────────────────────────────────────

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
    cache_path = cache_dir / f"activations_n{max_activations}.safetensors"
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

    try:
        module = dict(model.named_modules())[hookpoint]
    except KeyError:
        raise ValueError(f"Hookpoint {hookpoint} not found in model modules {list(model.named_modules())}.")

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


# ── Stage 2: TRAIN_SAE ─────────────────────────────────────────────────────────

def stage_train_sae(
    activations: torch.Tensor,
    d_hidden: int,
    k: int,
    device: torch.device,
    cache_dir: Path,
    max_steps: int = 10_000,
    ae_batch_size: int = 256,
    lr: float = 1e-3,
    auxk_alpha: float = 1 / 32,
) -> SparseCoder:
    """
    Train a TopK sparse autoencoder on cached activations using sparsify.SparseCoder.

    Sparsity is enforced by exact top-k selection: exactly k features fire per token.
    Decoder columns are kept unit-norm after each gradient step.
    """
    sae_dir = cache_dir / "sae"
    d_model = activations.shape[-1]

    if (sae_dir / "sae.safetensors").exists():
        print(f"Loading cached SAE from {sae_dir}")
        return SparseCoder.load_from_disk(sae_dir, device=str(device))

    print(
        f"Training TopK SAE: d_model={d_model}, d_hidden={d_hidden}, k={k}, steps={max_steps}"
    )
    cfg = SparseCoderConfig(num_latents=d_hidden, k=k)
    sae = SparseCoder(d_model, cfg, device=str(device), dtype=torch.float32)

    # Initialize b_dec to the mean of the activations (same as sparsify Trainer does
    # on the first forward pass) to help center the pre-activations at init.
    with torch.no_grad():
        sae.b_dec.data = activations.mean(0).to(device=str(device), dtype=torch.float32)

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    acts = activations.to(device).float()
    n = len(acts)

    sae.train()
    pbar = tqdm.trange(max_steps, desc="Training SAE")
    for step in pbar:
        idx = torch.randint(0, n, (ae_batch_size,), device=device)
        batch = acts[idx]

        sae.set_decoder_norm_to_unit_norm()
        out = sae(batch)
        loss = out.fvu + auxk_alpha * out.auxk_loss

        optimizer.zero_grad()
        loss.backward()
        sae.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()

        if step % 500 == 0:
            pbar.set_postfix(fvu=f"{out.fvu.item():.4f}", k=k)

    sae.eval()
    with torch.no_grad():
        sample = acts[torch.randperm(n)[:min(10_000, n)]]
        out_eval = sae(sample)
        final_fvu = out_eval.fvu.item()
        act_var = sample.var().item()

    sae_dir.mkdir(parents=True, exist_ok=True)
    sae.save_to_disk(sae_dir)
    print(
        f"Saved SAE to {sae_dir}  "
        f"(FVU={final_fvu:.5f}, var={act_var:.5f}, MSE≈{final_fvu * act_var:.5f}, k={k})"
    )
    return sae


def eval_sae_selectivity(
    ae: SparseCoder,
    model,
    tokenizer: PreTrainedTokenizerBase,
    hookpoint: str,
    device: torch.device,
    domain_datasets: dict[str, Dataset],
    n_tokens: int = 10_000,
    batch_size: int = 8,
) -> None:
    """Collect activations per domain and print FVU, MSE, L0 for each."""
    ae.eval()
    print("\n" + "=" * 60)
    print("SAE selectivity eval (per domain)")
    print(f"{'Domain':<12} {'FVU':>10} {'var':>10} {'MSE≈':>10} {'L0':>10}")
    print("-" * 60)
    captured: dict[str, torch.Tensor] = {}

    def _hook(module, input, output):
        x = output[0] if isinstance(output, tuple) else output
        captured["act"] = x.detach().float()

    module = dict(model.named_modules())[hookpoint]
    handle = module.register_forward_hook(_hook)

    try:
        for domain, ds in sorted(domain_datasets.items()):
            all_acts: list[torch.Tensor] = []
            total = 0
            model.eval()
            with torch.no_grad():
                for i in range(0, len(ds), batch_size):
                    questions = ds["question"][i: i + batch_size]
                    texts = [
                        tokenizer.apply_chat_template(
                            [{"role": "user", "content": q}],
                            tokenize=False, add_generation_prompt=True,
                        )
                        for q in questions
                    ]
                    enc = tokenizer(texts, return_tensors="pt", padding=True,
                                    truncation=True, max_length=512)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    model(**enc)
                    act = captured["act"]
                    mask = enc["attention_mask"].bool()
                    valid = act[mask].cpu()
                    remaining = n_tokens - total
                    if len(valid) > remaining:
                        valid = valid[torch.randperm(len(valid))[:remaining]]
                    all_acts.append(valid)
                    total += len(valid)
                    if total >= n_tokens:
                        break

            acts = torch.cat(all_acts, dim=0).to(device)
            with torch.no_grad():
                acts_f = acts.to(ae.dtype)
                out = ae(acts_f)
                fvu = out.fvu.item()
                var = acts_f.var().item()
                l0 = ae.cfg.k  # TopK SAE always fires exactly k features
            print(f"{domain:<12} {fvu:>10.4f} {var:>10.4f} {fvu*var:>10.4f} {l0:>10}")
    finally:
        handle.remove()
    print("=" * 60 + "\n")


def export_sae_to_hub(ae: SparseCoder, repo_id: str) -> None:
    """Upload a SparseCoder to HuggingFace in native sparsify format."""
    import tempfile
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    with tempfile.TemporaryDirectory() as tmp:
        sae_path = Path(tmp) / "sae"
        ae.save_to_disk(sae_path)
        api.upload_folder(folder_path=str(sae_path), repo_id=repo_id, repo_type="model")
    print(f"Uploaded SparseCoder SAE to HuggingFace: {repo_id}")


def _load_sae_from_cache(cache_dir: Path, device: torch.device) -> SparseCoder:
    sae_dir = cache_dir / "sae"
    if not (sae_dir / "sae.safetensors").exists():
        raise FileNotFoundError(
            f"No trained SAE found at {sae_dir}. Run --stage train_sae first."
        )
    return SparseCoder.load_from_disk(sae_dir, device=str(device))


# ── Stage 3/4: RECOVER / ATTACK (called from script_scoping_pipeline_stemqa.py) ─

def stage_train(
    train_dataset: Dataset,
    eval_datasets: dict[str, Dataset],
    ae: SparseCoder,
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
    """SFT with the trained SAE hooked in at hookpoint. Called by script_scoping_pipeline_stemqa.py."""
    old_project = os.environ.get("WANDB_PROJECT")
    old_run_name = os.environ.get("WANDB_RUN_NAME")
    try:
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_RUN_NAME"] = wandb_run

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


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--train-domain",
    type=click.Choice(ALL_DOMAINS),
    default="biology",
    show_default=True,
)
@click.option(
    "--stage",
    type=str,
    default="all",
    help="Stage(s): all, collect, train_sae, or comma-separated.",
)
@click.option("--n-collect-samples", type=int, default=5_000,
              help="Training samples to run model over for activation collection.")
@click.option("--max-activations", type=int, default=200_000,
              help="Max number of token activation vectors to cache.")
@click.option("--d-hidden-ratio", type=float, default=8.0,
              help="SAE hidden dim as a fraction of d_model (default: 8.0, overcomplete).")
@click.option("--k", type=int, default=64,
              help="Number of active features per token (TopK sparsity).")
@click.option("--max-steps-sae", type=int, default=10_000,
              help="Training steps for the sparse autoencoder.")
@click.option("--ae-batch-size", type=int, default=256)
@click.option("--ae-lr", type=float, default=1e-3)
@click.option("--auxk-alpha", type=float, default=1/32,
              help="Weight for auxk dead-latent loss (0 disables it; default 1/32).")
@click.option("--batch-size", "-b", type=int, default=4,
              help="Batch size for activation collection.")
@click.option("--device", type=str,
              default="cuda:0" if torch.cuda.is_available() else "cpu")
@click.option("--gemma2", "use_gemma2", is_flag=True, default=False)
@click.option("--gemma3", "use_gemma3", is_flag=True, default=False)
@click.option("--olmo", "use_olmo", is_flag=True, default=False)
@click.option("--dev", "dev", is_flag=True, default=False,
              help="Cap eval datasets at 500 samples each.")
@click.option("--hookpoint", "hookpoint_override", type=str, default=None,
              help="Override the default hookpoint, e.g. model.layers.24")
@click.option("--hf-user", type=str, default=None,
              help="HuggingFace username; if set, uploads the trained SAE as <user>/sae-<layer>-<train_domain>")
def main(
    train_domain: str,
    stage: str,
    n_collect_samples: int,
    max_activations: int,
    d_hidden_ratio: float,
    k: int,
    max_steps_sae: int,
    ae_batch_size: int,
    ae_lr: float,
    auxk_alpha: float,
    batch_size: int,
    device: str,
    use_gemma2: bool,
    use_gemma3: bool,
    use_olmo: bool,
    dev: bool,
    hookpoint_override: str | None,
    hf_user: str | None,
):
    if sum([use_gemma2, use_gemma3, use_olmo]) > 1:
        raise click.UsageError("Specify at most one of --gemma2, --gemma3, --olmo.")
    cfg = GEMMA2_CONFIG if use_gemma2 else OLMO_CONFIG if use_olmo else GEMMA3_CONFIG
    model_name = cfg["model_name"]
    hookpoint = hookpoint_override if hookpoint_override else cfg["hookpoint"]
    _layer_match = re.search(r"layers\.(\d+)$", hookpoint)
    cache_tag = f"layer_{_layer_match.group(1)}" if _layer_match else cfg["cache_tag"]

    device = torch.device(device)

    _valid_stages = {"all", "collect", "train_sae"}
    stages = {s.strip() for s in stage.split(",")}
    _invalid = stages - _valid_stages
    if _invalid:
        raise click.UsageError(f"Invalid stage(s): {_invalid}. Choose from: {_valid_stages}")
    if "all" in stages:
        stages = {"collect", "train_sae"}

    base_dir = Path(__file__).parent
    model_slug = model_name.replace("/", "--")
    sae_cache_dir = (
        base_dir / ".cache" / f"stemqa_{train_domain}" / "learnsae"
        / model_slug / cache_tag / f"n{max_activations}"
    )

    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    model = model.to(device)
    model.gradient_checkpointing_disable()
    if hasattr(model, "model"):
        model.model.gradient_checkpointing = False

    # Load all domain datasets — needed for eval_sae_selectivity after train_sae.
    print("Loading all domain datasets...")
    all_domain_splits: dict[str, tuple[Dataset, Dataset]] = {}
    for domain in ALL_DOMAINS:
        t = time.time()
        tr, ev = load_domain_train_eval(domain, tokenizer)
        all_domain_splits[domain] = (tr, ev)
        print(f"  {domain}: {len(tr)} train, {len(ev)} eval in {time.time()-t:.1f}s")
    train_ds = all_domain_splits[train_domain][0]

    n_eval_cap = 500 if dev else None

    # ── Stage 1: COLLECT ──────────────────────────────────────────────────
    activations: torch.Tensor | None = None
    if "collect" in stages:
        activations = stage_collect(
            train_dataset=train_ds,
            model=model,
            tokenizer=tokenizer,
            hookpoint=hookpoint,
            device=device,
            cache_dir=sae_cache_dir,
            batch_size=batch_size,
            n_collect_samples=n_collect_samples,
            max_activations=max_activations,
        )
    elif "train_sae" in stages:
        acts_path = sae_cache_dir / f"activations_n{max_activations}.safetensors"
        if not acts_path.exists():
            raise click.UsageError(
                f"No cached activations at {acts_path}. Run --stage collect first."
            )
        print(f"Loading cached activations from {acts_path}")
        activations = load_file(str(acts_path))["activations"]

    # ── Stage 2: TRAIN_SAE ────────────────────────────────────────────────
    if "train_sae" in stages:
        if activations is None:
            raise click.UsageError("No activations available. Run --stage collect first.")
        d_model = activations.shape[-1]
        d_hidden = max(1, int(d_model * d_hidden_ratio))
        ae = stage_train_sae(
            activations=activations,
            d_hidden=d_hidden,
            k=k,
            device=device,
            cache_dir=sae_cache_dir,
            max_steps=max_steps_sae,
            ae_batch_size=ae_batch_size,
            lr=ae_lr,
            auxk_alpha=auxk_alpha,
        )
        eval_domain_datasets = {
            d: (
                all_domain_splits[d][1].select(range(min(n_eval_cap, len(all_domain_splits[d][1]))))
                if n_eval_cap else all_domain_splits[d][1]
            )
            for d in ALL_DOMAINS
        }
        eval_sae_selectivity(
            ae=ae, model=model, tokenizer=tokenizer, hookpoint=hookpoint,
            device=device,
            domain_datasets=eval_domain_datasets,
        )
        if hf_user is not None:
            _layer = re.search(r"layers\.(\d+)$", hookpoint)
            _layer_str = f"layer{_layer.group(1)}" if _layer else cache_tag
            repo_id = f"{hf_user}/sae-{_layer_str}-{train_domain}-dim{d_hidden}-k{k}"
            export_sae_to_hub(ae, repo_id=repo_id)

        print(f"\nSAE cache dir: {sae_cache_dir}")
        print(f"To run recovery/attack training, pass to the main pipeline script:")
        print(f"  --domain-sae-path {sae_cache_dir} --hookpoint {hookpoint}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
