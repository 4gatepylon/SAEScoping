"""
Evaluate per-domain selectivity of a pretrained GemmaScope SAE (optionally pruned).

For each domain, collects hookpoint activations and measures:
  - MSE:      mean squared reconstruction error
  - var:      variance of input activations
  - MSE/var:  normalised reconstruction error (lower = better reconstruction)
  - L0:       average number of active SAE features per token

Usage:
  # Unpruned SAE, gemma2, train domain biology:
  python script_eval_sae_selectivity.py --gemma2 --train-domain biology

  # Pruned SAE (firing-rate threshold applied):
  python script_eval_sae_selectivity.py --gemma2 --train-domain biology --prune --firing-rate-threshold 1e-4

  # Gemma3-later (layer 41):
  python script_eval_sae_selectivity.py --gemma3-later --train-domain biology --prune
"""
from __future__ import annotations

import os
import sys
import gc
from pathlib import Path
from functools import partial

import click
import torch
import torch.nn.functional as F
import tqdm
from datasets import Dataset, load_dataset
from safetensors.torch import load_file
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

sys.path.append(os.path.abspath(".."))
from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae

ALL_DOMAINS = ["biology", "chemistry", "math", "physics"]

GEMMA3_CONFIG = dict(
    model_name="google/gemma-3-12b-it",
    sae_release="gemma-scope-2-12b-it-res",
    sae_id="layer_31_width_16k_l0_medium",
    hookpoint="model.language_model.layers.31",
    cache_tag="layer_31--width_16k--canonical",
)
GEMMA2_CONFIG = dict(
    model_name="google/gemma-2-9b-it",
    sae_release="gemma-scope-9b-it-res-canonical",
    sae_id="layer_31/width_16k/canonical",
    hookpoint="model.layers.31",
    cache_tag="layer_31--width_16k--canonical",
)
GEMMA3_LATER_CONFIG = dict(
    model_name="google/gemma-3-12b-it",
    sae_release="gemma-scope-2-12b-it-res",
    sae_id="layer_41_width_16k_l0_medium",
    hookpoint="model.language_model.layers.41",
    cache_tag="layer_41--width_16k--canonical",
)


def _collect_activations(
    model,
    tokenizer: PreTrainedTokenizerBase,
    hookpoint: str,
    device: torch.device,
    dataset: Dataset,
    n_tokens: int = 10_000,
    batch_size: int = 8,
) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def _hook(module, input, output):
        x = output[0] if isinstance(output, tuple) else output
        captured["act"] = x.detach().float()

    module = dict(model.named_modules())[hookpoint]
    handle = module.register_forward_hook(_hook)
    all_acts: list[torch.Tensor] = []
    total = 0
    model.eval()
    try:
        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                questions = dataset["question"][i: i + batch_size]
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
    finally:
        handle.remove()
    return torch.cat(all_acts, dim=0)


def _eval_domain(sae, acts: torch.Tensor, device: torch.device) -> dict:
    with torch.no_grad():
        if hasattr(sae, "sae"):  # SAELensEncDecCallbackWrapper (pruned)
            underlying = sae.sae
            underlying.eval()
            acts_f = acts.to(device=underlying.device, dtype=underlying.dtype)
            h_full = underlying.encode(acts_f)
            h = sae.callback(h_full, sae.ctx)  # apply pruning mask
            recon = underlying.decode(h)
        else:  # plain sae_lens.SAE (unpruned)
            sae.eval()
            acts_f = acts.to(device=sae.device, dtype=sae.dtype)
            h = sae.encode(acts_f)
            recon = sae.decode(h)
        mse = F.mse_loss(recon, acts_f).item()
        var = acts_f.var().item()
        l0 = (h > 0).float().sum(dim=-1).mean().item()
    return {"mse": mse, "var": var, "mse_var": mse / var if var > 0 else float("nan"), "l0": l0}


@click.command()
@click.option("--gemma2", "use_gemma2", is_flag=True, default=False)
@click.option("--gemma3", "use_gemma3", is_flag=True, default=False)
@click.option("--gemma3-later", "later_gemma3", is_flag=True, default=False)
@click.option("--train-domain", type=click.Choice(ALL_DOMAINS), default="biology", show_default=True,
              help="Domain used to compute firing rates for pruning.")
@click.option("--prune", "do_prune", is_flag=True, default=False,
              help="Prune SAE neurons below firing-rate threshold before eval.")
@click.option("--firing-rate-threshold", type=float, default=1e-4, show_default=True)
@click.option("--n-rank-samples", type=int, default=10_000, show_default=True,
              help="Number of train samples used when computing firing rates (must match cached value).")
@click.option("--n-eval-tokens", type=int, default=10_000, show_default=True,
              help="Number of tokens to collect per domain for eval.")
@click.option("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
def main(
    use_gemma2: bool,
    use_gemma3: bool,
    later_gemma3: bool,
    train_domain: str,
    do_prune: bool,
    firing_rate_threshold: float,
    n_rank_samples: int,
    n_eval_tokens: int,
    device: str,
):
    if sum([use_gemma2, use_gemma3, later_gemma3]) > 1:
        raise click.UsageError("Specify at most one of --gemma2, --gemma3, --gemma3-later.")
    cfg = GEMMA2_CONFIG if use_gemma2 else (GEMMA3_LATER_CONFIG if later_gemma3 else GEMMA3_CONFIG)
    model_name = cfg["model_name"]
    sae_release = cfg["sae_release"]
    sae_id = cfg["sae_id"]
    hookpoint = cfg["hookpoint"]
    cache_tag = cfg["cache_tag"]
    device = torch.device(device)

    base_dir = Path(__file__).parent
    model_slug = model_name.replace("/", "--")

    # ── Load tokenizer + model ────────────────────────────────────────────────
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="eager"
    )
    model = model.to(device)
    model.gradient_checkpointing_disable()

    # ── Load SAE ──────────────────────────────────────────────────────────────
    print(f"Loading SAE: {sae_release} / {sae_id}...")
    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=str(device))
    if isinstance(sae, tuple):
        sae = sae[0]

    if do_prune:
        cache_dir = (
            base_dir / ".cache" / f"stemqa_{train_domain}"
            / "ignore_padding_True" / model_slug / cache_tag / f"n{n_rank_samples}"
        )
        dist_path = cache_dir / "firing_rates.safetensors"
        if not dist_path.exists():
            raise click.UsageError(
                f"No cached firing rates at {dist_path}. "
                "Run --stage rank first with script_scoping_pipeline_stemqa.py."
            )
        data = load_file(str(dist_path))
        distribution = data["distribution"]
        ranking = data["ranking"]
        n_kept = int((distribution >= firing_rate_threshold).sum().item())
        print(f"Pruning SAE: keeping {n_kept} / {len(distribution)} neurons "
              f"(threshold={firing_rate_threshold})")
        sae = get_pruned_sae(sae, ranking, K_or_p=n_kept, T=0.0)

    sae = sae.to(device)

    # ── Load eval datasets ────────────────────────────────────────────────────
    print("Loading eval datasets...")
    domain_datasets: dict[str, Dataset] = {}
    for domain in ALL_DOMAINS:
        ds = load_dataset("4gate/StemQAMixture", domain, split="train", streaming=False)
        ds = ds.shuffle(seed=42)
        n_eval = int(len(ds) * 0.2)
        domain_datasets[domain] = ds.select(range(n_eval))
        print(f"  {domain}: {n_eval} eval samples")

    # ── Per-domain eval ───────────────────────────────────────────────────────
    print(f"\nCollecting {n_eval_tokens} tokens per domain and evaluating SAE reconstruction...\n")
    print(f"{'Domain':<12} {'MSE':>10} {'var':>10} {'MSE/var':>10} {'L0':>10}")
    print("-" * 60)
    for domain in ALL_DOMAINS:
        acts = _collect_activations(
            model=model, tokenizer=tokenizer, hookpoint=hookpoint,
            device=device, dataset=domain_datasets[domain],
            n_tokens=n_eval_tokens,
        )
        metrics = _eval_domain(sae, acts, device)
        print(f"{domain:<12} {metrics['mse']:>10.4f} {metrics['var']:>10.4f} "
              f"{metrics['mse_var']:>10.4f} {metrics['l0']:>10.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
