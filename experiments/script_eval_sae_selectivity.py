"""
Evaluate per-domain selectivity of a pretrained GemmaScope SAE (optionally pruned).
Supports sweeping over layers and firing-rate thresholds to find the best combination.

Per-domain metrics:
  - MSE/var:  normalised reconstruction error (lower = better)
  - L0:       average active features per token

Inter-domain selectivity metrics (pairwise averages, lower = more selective):
  - mean_jaccard:  Jaccard similarity of above-threshold active-feature sets
  - mean_cosine:   cosine similarity of mean firing-rate vectors

Aggregate:
  - specificity:  mean(max_domain_rate / mean_domain_rate) across active features
                  (higher = features tend to be domain-exclusive)

Usage:
  # Single layer, gemma2, train domain biology (original behaviour):
  python script_eval_sae_selectivity.py --gemma2 --train-domain biology

  # Pruned SAE, single layer:
  python script_eval_sae_selectivity.py --gemma2 --train-domain biology --prune

  # Sweep layers only (unpruned):
  python script_eval_sae_selectivity.py --gemma3 --layers 20,25,31,35,41

  # Sweep layers + thresholds (2-D grid):
  python script_eval_sae_selectivity.py --gemma3 --train-domain biology \\
      --layers 20,25,31,35,41 --thresholds 1e-3,1e-4,1e-5 \\
      --output-csv results/layer_threshold_sweep.csv

  # Save to CSV:
  python script_eval_sae_selectivity.py --gemma3 --layers 20,25,31,35,41 --output-csv results/layer_sweep.csv
"""
from __future__ import annotations

import csv
import gc
import os
import sys
from pathlib import Path

import click
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from safetensors.torch import load_file, save_file
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

sys.path.append(os.path.abspath(".."))
from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae

ALL_DOMAINS = ["biology", "chemistry", "math", "physics"]

# sae_id_template, hookpoint_template, cache_tag_template use {layer} and {l0} as placeholders.
# gemma3 uses gemma-scope-2-12b-it-res-all (resid_post at every layer); IDs use l0_big / l0_small.
# gemma2 uses gemma-scope-9b-it-res-canonical (canonical layers only); IDs use /width_16k/canonical.
GEMMA3_CONFIG = dict(
    model_name="google/gemma-3-12b-it",
    sae_release="gemma-scope-2-12b-it-res-all",
    sae_id_template="layer_{layer}_width_16k_l0_{l0}",
    hookpoint_template="model.language_model.layers.{layer}",
    cache_tag_template="layer_{layer}--width_16k--l0_{l0}",
    default_layers=[31],
    default_l0="small",
)
GEMMA2_CONFIG = dict(
    model_name="google/gemma-2-9b-it",
    sae_release="gemma-scope-9b-it-res-canonical",
    sae_id_template="layer_{layer}/width_16k/canonical",
    hookpoint_template="model.layers.{layer}",
    cache_tag_template="layer_{layer}--width_16k--canonical",
    default_layers=[31],
    default_l0=None,  # gemma2 canonical IDs don't use l0 suffix
)
GEMMA3_LATER_CONFIG = dict(
    model_name="google/gemma-3-12b-it",
    sae_release="gemma-scope-2-12b-it-res-all",
    sae_id_template="layer_{layer}_width_16k_l0_{l0}",
    hookpoint_template="model.language_model.layers.{layer}",
    cache_tag_template="layer_{layer}--width_16k--l0_{l0}",
    default_layers=[41],
    default_l0="small",
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
                questions = dataset["question"][i : i + batch_size]
                texts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": q}],
                        tokenize=False,
                        add_generation_prompt=True,
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


def _eval_domain(
    sae, acts: torch.Tensor, device: torch.device
) -> tuple[dict, torch.Tensor]:
    """Returns (per-domain metrics dict, per-feature firing rates of shape (d_sae,)).

    Firing rate = fraction of tokens where the feature is active (h > 0), consistent
    with the convention used by rank_neurons and the pruning threshold.
    """
    with torch.no_grad():
        if hasattr(sae, "sae"):  # SAELensEncDecCallbackWrapper (pruned)
            underlying = sae.sae
            underlying.eval()
            acts_f = acts.to(device=underlying.device, dtype=underlying.dtype)
            h_full = underlying.encode(acts_f)
            h = sae.callback(h_full, sae.ctx)
            recon = underlying.decode(h)
        else:  # plain sae_lens.SAE
            sae.eval()
            acts_f = acts.to(device=sae.device, dtype=sae.dtype)
            h = sae.encode(acts_f)
            recon = sae.decode(h)
        mse = F.mse_loss(recon, acts_f).item()
        var = acts_f.var().item()
        l0 = (h > 0).float().sum(dim=-1).mean().item()
        firing_rates = (h > 0).float().mean(dim=0).cpu()
    return (
        {"mse": mse, "var": var, "mse_var": mse / var if var > 0 else float("nan"), "l0": l0},
        firing_rates,
    )


def _compute_selectivity(
    feature_rates: dict[str, torch.Tensor],
    train_domain: str,
    threshold: float = 1e-4,
) -> dict:
    """Train-domain vs each other domain selectivity metrics.

    For each other domain, computes:
      jaccard_<train>_<other>:     Jaccard similarity of above-threshold active-feature sets
      cosine_<train>_<other>:      cosine similarity of firing-rate vectors
      specificity_<train>_<other>: mean(max(r_train, r_other) / mean(r_train, r_other))
                                   per active feature — higher means features more exclusive
    """
    result = {}
    r_train = feature_rates[train_domain]
    a_train = set((r_train > threshold).nonzero(as_tuple=True)[0].tolist())
    other_domains = [d for d in ALL_DOMAINS if d != train_domain]

    for other in other_domains:
        r_other = feature_rates[other]
        key = f"{train_domain[:3]}_{other[:3]}"

        a_other = set((r_other > threshold).nonzero(as_tuple=True)[0].tolist())
        jaccard = len(a_train & a_other) / len(a_train | a_other) if (a_train | a_other) else float("nan")
        result[f"jaccard_{key}"] = jaccard

        cosine = F.cosine_similarity(r_train.unsqueeze(0), r_other.unsqueeze(0)).item()
        result[f"cosine_{key}"] = cosine

        rate_pair = torch.stack([r_train, r_other], dim=0)
        mean_rates = rate_pair.mean(dim=0)
        max_rates = rate_pair.max(dim=0).values
        active = mean_rates > 0
        result[f"specificity_{key}"] = (
            (max_rates[active] / mean_rates[active]).mean().item() if active.any() else float("nan")
        )

    return result


def _compute_or_load_firing_rates(
    sae,
    hookpoint: str,
    model,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    device: torch.device,
    n_rank_samples: int,
    n_rank_tokens: int,
    cache_dir: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (or load cached) per-feature firing rates on the train domain at this layer.

    The cache format is compatible with the pipeline's stage_rank output so pre-computed
    caches are reused automatically.
    Returns (ranking, distribution) where distribution[i] is the fraction of tokens
    where feature i fired.
    """
    cache_path = cache_dir / "firing_rates.safetensors"
    if cache_path.exists():
        print(f"  Loading cached firing rates from {cache_path}")
        data = load_file(str(cache_path))
        return data["ranking"], data["distribution"]

    n = min(n_rank_samples, len(train_dataset))
    rank_ds = train_dataset.select(range(n))
    print(f"  Computing firing rates: up to {n_rank_tokens} tokens from {n} train samples at {hookpoint}...")

    acts = _collect_activations(model, tokenizer, hookpoint, device, rank_ds, n_tokens=n_rank_tokens)

    encode_batch = 2048
    fired_sum: torch.Tensor | None = None
    with torch.no_grad():
        for i in range(0, len(acts), encode_batch):
            batch = acts[i : i + encode_batch].to(device=sae.device, dtype=sae.dtype)
            h = sae.encode(batch)
            fired = (h > 0).float().sum(dim=0).cpu()
            fired_sum = fired if fired_sum is None else fired_sum + fired
    distribution = fired_sum / len(acts)
    ranking = torch.argsort(distribution, descending=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    save_file({"ranking": ranking, "distribution": distribution}, str(cache_path))
    print(f"  Saved to {cache_path}")

    return ranking, distribution


@click.command()
@click.option("--gemma2", "use_gemma2", is_flag=True, default=False)
@click.option("--gemma3", "use_gemma3", is_flag=True, default=False)
@click.option("--gemma3-later", "later_gemma3", is_flag=True, default=False)
@click.option("--train-domain", type=click.Choice(ALL_DOMAINS), default="biology", show_default=True,
              help="Domain used to compute firing rates for pruning.")
@click.option("--prune", "do_prune", is_flag=True, default=False,
              help="Prune with --firing-rate-threshold. Superseded by --thresholds when both are set.")
@click.option("--firing-rate-threshold", type=float, default=1e-4, show_default=True,
              help="Single threshold used when --prune is set (and --thresholds is not).")
@click.option("--thresholds", "thresholds_str", type=str, default=None,
              help="Comma-separated firing-rate thresholds to sweep, e.g. '1e-3,1e-4,1e-5'. "
                   "Enables pruning automatically; overrides --firing-rate-threshold.")
@click.option("--n-rank-samples", type=int, default=10_000, show_default=True,
              help="Train samples used to compute firing rates (must match pipeline cache if reusing).")
@click.option("--n-rank-tokens", type=int, default=50_000, show_default=True,
              help="Max tokens collected from train domain for firing rate computation.")
@click.option("--n-eval-tokens", type=int, default=10_000, show_default=True,
              help="Tokens to collect per domain for selectivity eval.")
@click.option("--layers", type=str, default=None,
              help="Comma-separated layer numbers to sweep (default: model canonical layer).")
@click.option("--l0", "l0_level", type=click.Choice(["small", "big"]), default=None,
              help="SAE sparsity level for gemma3 (small=sparser, big=denser). Default: small.")
@click.option("--output-csv", "output_csv", type=str, default=None,
              help="Path to save results CSV.")
@click.option("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
def main(
    use_gemma2: bool,
    use_gemma3: bool,
    later_gemma3: bool,
    train_domain: str,
    do_prune: bool,
    firing_rate_threshold: float,
    thresholds_str: str | None,
    n_rank_samples: int,
    n_rank_tokens: int,
    n_eval_tokens: int,
    layers: str | None,
    l0_level: str | None,
    output_csv: str | None,
    device: str,
):
    if sum([use_gemma2, use_gemma3, later_gemma3]) > 1:
        raise click.UsageError("Specify at most one of --gemma2, --gemma3, --gemma3-later.")
    cfg = GEMMA2_CONFIG if use_gemma2 else (GEMMA3_LATER_CONFIG if later_gemma3 else GEMMA3_CONFIG)
    model_name = cfg["model_name"]
    device = torch.device(device)
    base_dir = Path(__file__).parent
    model_slug = model_name.replace("/", "--")

    # Resolve l0 level (only applies to gemma3; gemma2 canonical IDs have no l0 suffix).
    l0 = l0_level or cfg["default_l0"]

    layer_list = (
        [int(x.strip()) for x in layers.split(",")]
        if layers is not None
        else cfg["default_layers"]
    )

    # Determine which thresholds to sweep (empty list = no pruning).
    if thresholds_str:
        threshold_list = [float(t.strip()) for t in thresholds_str.split(",")]
    elif do_prune:
        threshold_list = [firing_rate_threshold]
    else:
        threshold_list = []

    # Default Jaccard/cosine threshold for unpruned eval.
    selectivity_threshold = firing_rate_threshold

    # Load tokenizer + model
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="eager"
    )
    model = model.to(device)
    model.gradient_checkpointing_disable()

    # Load eval datasets: first 20% of each domain (consistent with pipeline).
    print("Loading eval datasets...")
    domain_datasets: dict[str, Dataset] = {}
    for domain in ALL_DOMAINS:
        ds = load_dataset("4gate/StemQAMixture", domain, split="train", streaming=False)
        ds = ds.shuffle(seed=42)
        n_eval = int(len(ds) * 0.2)
        domain_datasets[domain] = ds.select(range(n_eval))
        print(f"  {domain}: {n_eval} eval samples")

    # Load train domain train split (remaining 80%) for firing rate computation.
    train_ds: Dataset | None = None
    if threshold_list:
        print(f"Loading {train_domain} train split for firing rate computation...")
        full = load_dataset("4gate/StemQAMixture", train_domain, split="train", streaming=False)
        full = full.shuffle(seed=42)
        n_eval = int(len(full) * 0.2)
        train_ds = full.select(range(n_eval, len(full)))
        print(f"  {train_domain} train: {len(train_ds)} samples")

    # ── Main sweep ─────────────────────────────────────────────────────────────
    all_results: list[dict] = []

    for layer in layer_list:
        fmt = dict(layer=layer, l0=l0)
        hookpoint = cfg["hookpoint_template"].format(**fmt)
        sae_id = cfg["sae_id_template"].format(**fmt)
        cache_tag = cfg["cache_tag_template"].format(**fmt)
        print(f"\n{'='*70}")
        print(f"Layer {layer}  |  SAE: {sae_id}")
        print("=" * 70)

        # Load SAE (on device; needed for both firing rate computation and eval).
        sae = SAE.from_pretrained(release=cfg["sae_release"], sae_id=sae_id, device="cpu")
        if isinstance(sae, tuple):
            sae = sae[0]
        sae = sae.to(device)

        # Compute or load firing rates for this layer (only when pruning).
        ranking: torch.Tensor | None = None
        distribution: torch.Tensor | None = None
        if threshold_list:
            cache_dir = (
                base_dir / ".cache" / f"stemqa_{train_domain}"
                / "ignore_padding_True" / model_slug / cache_tag / f"n{n_rank_samples}"
            )
            ranking, distribution = _compute_or_load_firing_rates(
                sae, hookpoint, model, tokenizer, train_ds, device, n_rank_samples, n_rank_tokens, cache_dir
            )

        # Collect eval activations once for all domains at this layer.
        print(f"Collecting {n_eval_tokens} eval tokens per domain at layer {layer}...")
        domain_acts: dict[str, torch.Tensor] = {}
        for domain, ds in domain_datasets.items():
            print(f"  {domain}...", end=" ", flush=True)
            domain_acts[domain] = _collect_activations(
                model, tokenizer, hookpoint, device, ds, n_eval_tokens
            )
            print("done")

        # Evaluate for each threshold (or unpruned if no threshold list).
        thresholds_to_eval: list[float | None] = threshold_list if threshold_list else [None]
        for threshold in thresholds_to_eval:
            if threshold is not None:
                n_kept = int((distribution >= threshold).sum().item())
                eval_sae = get_pruned_sae(sae, ranking, K_or_p=n_kept, T=0.0).to(device)
                threshold_label: str = str(threshold)
                sel_threshold = threshold
            else:
                n_kept = None
                eval_sae = sae
                threshold_label = "unpruned"
                sel_threshold = selectivity_threshold

            label = f"layer={layer}, threshold={threshold_label}"
            print(f"  [{label}]  n_kept={n_kept}", end="  ", flush=True)

            feature_rates: dict[str, torch.Tensor] = {}
            row: dict = {"layer": layer, "threshold": threshold_label, "n_kept": n_kept}
            for domain, acts in domain_acts.items():
                metrics, rates = _eval_domain(eval_sae, acts, device)
                for k, v in metrics.items():
                    row[f"{domain}_{k}"] = v
                feature_rates[domain] = rates

            sel = _compute_selectivity(feature_rates, train_domain=train_domain, threshold=sel_threshold)
            row.update(sel)
            all_results.append(row)
            other_domains = [d for d in ALL_DOMAINS if d != train_domain]
            sel_summary = "  ".join(
                f"{other[:3]}: j={sel[f'jaccard_{train_domain[:3]}_{other[:3]}']:.3f}"
                f" c={sel[f'cosine_{train_domain[:3]}_{other[:3]}']:.3f}"
                f" s={sel[f'specificity_{train_domain[:3]}_{other[:3]}']:.2f}"
                for other in other_domains
            )
            print(sel_summary)

            if threshold is not None:
                del eval_sae
                gc.collect()

        del sae, domain_acts
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary table ──────────────────────────────────────────────────────────
    has_thresholds = threshold_list or thresholds_str
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)

    # Per-domain MSE/var
    col = 14
    header = f"{'layer':<7} {'threshold':<12} {'n_kept':<8}"
    for d in ALL_DOMAINS:
        header += f"  {(d[:4]+' MSE/v'):>{col}}  {(d[:4]+' L0'):>8}"
    print("\n" + header)
    print("-" * len(header))
    for row in all_results:
        line = f"{row['layer']:<7} {str(row['threshold']):<12} {str(row.get('n_kept') or '-'):<8}"
        for d in ALL_DOMAINS:
            line += f"  {row.get(f'{d}_mse_var', float('nan')):>{col}.4f}  {row.get(f'{d}_l0', float('nan')):>8.1f}"
        print(line)

    # Selectivity (train domain vs each other domain)
    other_domains = [d for d in ALL_DOMAINS if d != train_domain]
    sel_header = f"{'layer':<7} {'threshold':<12} {'n_kept':<8}"
    for other in other_domains:
        key = f"{train_domain[:3]}_{other[:3]}"
        sel_header += f"  {'jac_'+key:>12}  {'cos_'+key:>12}  {'spc_'+key:>12}"
    print("\n" + sel_header)
    print("-" * len(sel_header))
    for row in all_results:
        line = f"{row['layer']:<7} {str(row['threshold']):<12} {str(row.get('n_kept') or '-'):<8}"
        for other in other_domains:
            key = f"{train_domain[:3]}_{other[:3]}"
            line += (
                f"  {row.get(f'jaccard_{key}', float('nan')):>12.4f}"
                f"  {row.get(f'cosine_{key}', float('nan')):>12.4f}"
                f"  {row.get(f'specificity_{key}', float('nan')):>12.4f}"
            )
        print(line)
    print("=" * len(sel_header))
    print("(lower jaccard/cosine = more selective;  higher specificity = more domain-exclusive features)")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    if output_csv and all_results:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
