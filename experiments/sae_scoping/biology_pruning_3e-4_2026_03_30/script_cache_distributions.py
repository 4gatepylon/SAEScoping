"""script_cache_distributions.py

Cache SAE neuron firing-rate distributions for any StemQA subject (chemistry,
physics, math, biology, ...).  Adapted from
experiments/sae_scoping/script_2025_12_08_cache_distributions.py.

Key differences from the original:
  - Supports any StemQA subset via dataset_utils (single unified source).
  - --checkpoint lets you specify a fine-tuned model directory instead of the
    bare google/gemma-2-9b-it base model.
  - --layer restricts which SAE layer(s) to compute (default: 31 only, since
    that is the hookpoint used throughout this experiment).
  - --output-dir controls where distributions are written (default:
    ./distributions_cache next to this script).
  - --n-samples controls how many ranking samples to draw per subset
    (default 2000).

Output layout (same convention as the original .cache folder):
    <output_dir>/ignore_padding_<True|False>/<dataset>/<sae_id>/distribution.safetensors

Usage example (see jobs_2026_04_03/get_dist_chemistry.sh for the full command):
    CUDA_VISIBLE_DEVICES=0 python script_cache_distributions.py \\
        --datasets chemistry \\
        --checkpoint downloaded/model_layers_31_h0.0003/outputs/checkpoint-2000 \\
        --layer 31 --batch-size 4
"""

from __future__ import annotations

import gc
import itertools
import re
from pathlib import Path

import click
import torch
import tqdm
from datasets import Dataset
from jaxtyping import Float, Integer
from safetensors.torch import save_file
from sae_lens import SAE
from transformers import (
    AutoTokenizer,
    Gemma2ForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from sae_scoping.trainers.sae_enhanced.rank import rank_neurons

# dataset_utils lives next to this script in the same experiment folder
from dataset_utils import load_stem_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_MODEL = "google/gemma-2-9b-it"
_SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"

# All available layer/width combos; narrow to just layer 31 by default via
# --layer, since that is the hookpoint used throughout this experiment.
_ALL_SAE_IDS: list[str] = [
    f"layer_{layer}/width_{width}/canonical"
    for layer, width in [
        (9, "16k"),
        (20, "16k"),
        (31, "16k"),
    ]
]

_STEMQA_SUBSETS = ("physics", "chemistry", "math", "biology")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sae_id2hookpoint(sae_id: str) -> str:
    assert re.match(r"^layer_\d+/width_16k/canonical$", sae_id)
    layer_num = int(sae_id.split("/", 1)[0].split("_")[1])
    return f"model.layers.{layer_num}"


def load_stemqa_dataset(
    subset: str, n_samples: int, tokenizer: PreTrainedTokenizerBase
) -> Dataset:
    """Load n_samples from StemQAMixture train split for a given subset."""
    return load_stem_dataset(
        tokenizer,
        subsets=(subset,),
        split="train",
        max_samples_per_subset=n_samples,
        seed=42,
    )


def rank_neurons_shim(
    tokenized: Dataset,
    sae_id: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    ignore_padding: bool,
    device: torch.device,
) -> tuple[Integer[torch.Tensor, "d_sae"], Float[torch.Tensor, "d_sae"]]:
    sae: SAE = SAE.from_pretrained(
        release=_SAE_RELEASE, sae_id=sae_id, device=device,
    )
    sae = sae.to(device)
    hookpoint = sae_id2hookpoint(sae_id)
    with torch.no_grad():
        ranking, distribution = rank_neurons(
            dataset=tokenized,
            sae=sae,
            model=model,
            tokenizer=tokenizer,
            T=0.0,
            hookpoint=hookpoint,
            batch_size=batch_size,
            token_selection="attention_mask" if ignore_padding else "all",
            return_distribution=True,
        )
    ranking = ranking.detach().cpu()
    distribution = distribution.detach().cpu()
    sae = sae.to("cpu")
    del sae
    gc.collect()
    torch.cuda.empty_cache()
    return ranking, distribution


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--datasets", "-d", type=str, default="chemistry,physics",
    help="Comma-separated list of StemQA subsets to process. "
         f"Recognised names: {', '.join(_STEMQA_SUBSETS)}. "
         "Default: chemistry,physics",
)
@click.option(
    "--checkpoint", "-c", type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to a fine-tuned model directory. "
         "Defaults to the base google/gemma-2-9b-it model.",
)
@click.option(
    "--layer", "-l", type=int, default=31,
    help="SAE layer to compute distributions for (9, 20, or 31). Default: 31.",
)
@click.option(
    "--ignore-padding/--no-ignore-padding", "ignore_padding",
    default=True,
    help="Whether to ignore padding tokens when counting firings. Default: True.",
)
@click.option(
    "--n-samples", "-n", type=int, default=2000,
    help="Number of ranking samples to draw per subset (train split). Default: 2000.",
)
@click.option("--batch-size", "-b", type=int, default=4)
@click.option(
    "--output-dir", "-o", type=str, default=None,
    help="Where to write distribution.safetensors files. "
         "Default: ./distributions_cache next to this script.",
)
def main(
    datasets: str,
    checkpoint: Path | None,
    layer: int,
    ignore_padding: bool,
    n_samples: int,
    batch_size: int,
    output_dir: str | None,
) -> None:
    """Compute and cache SAE neuron firing-rate distributions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve SAE IDs
    assert layer in (9, 20, 31), f"--layer must be one of 9, 20, 31; got {layer}"
    sae_ids = [f"layer_{layer}/width_16k/canonical"]

    # Resolve output directory
    script_dir = Path(__file__).parent
    out_root = Path(output_dir) if output_dir else script_dir / "distributions_cache"

    # Resolve model
    model_path = str(checkpoint) if checkpoint else _BASE_MODEL
    print(f"Loading model from: {model_path}")
    model = Gemma2ForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager",
    )
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    # Build (dataset, name) pairs
    requested = [s.strip() for s in datasets.split(",") if s.strip()]
    print(f"Datasets to process: {requested}")
    datasets_and_names: list[tuple[Dataset, str]] = []
    for name in requested:
        if name not in _STEMQA_SUBSETS:
            raise click.BadParameter(
                f"Unknown dataset '{name}'. "
                f"Valid: {', '.join(_STEMQA_SUBSETS)}"
            )
        ds = load_stemqa_dataset(name, n_samples, tokenizer)
        print(f"  Loaded '{name}': {len(ds)} samples")
        datasets_and_names.append((ds, name))

    # Run distributions
    combos = list(itertools.product(datasets_and_names, sae_ids))
    print(f"\nWill compute {len(combos)} combo(s) "
          f"(ignore_padding={ignore_padding})\n")

    for (ds, ds_name), sae_id in tqdm.tqdm(combos, desc="Computing distributions"):
        subfolder = (
            out_root
            / f"ignore_padding_{ignore_padding}"
            / ds_name
            / sae_id.replace("/", "--")
        )
        if subfolder.exists():
            print(f"  Skipping {subfolder} (already exists)")
            continue

        print(f"  Computing: dataset={ds_name}, sae={sae_id}")
        _, distribution = rank_neurons_shim(
            tokenized=ds,
            sae_id=sae_id,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            ignore_padding=ignore_padding,
            device=device,
        )

        subfolder.mkdir(parents=True, exist_ok=True)
        save_file({"distribution": distribution}, subfolder / "distribution.safetensors")
        print(f"  Saved -> {subfolder / 'distribution.safetensors'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
