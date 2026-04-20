"""
Compute SAE neuron firing rates on any HuggingFace dataset and select neurons
above a given firing rate threshold.

Saves firing rates and selected neuron indices to --output-dir.

Examples
--------
# StemQA biology, default Gemma 3 12B-IT SAEs, threshold 1e-4:
python script_firing_rates.py \
    --dataset 4gate/StemQAMixture --dataset-config biology \
    --question-col question --answer-col answer \
    --threshold 1e-4

# GSM8k, single SAE layer, no threshold (save all):
python script_firing_rates.py \
    --dataset openai/gsm8k --dataset-config main \
    --question-col question --answer-col answer \
    --sae-ids layer_23_width_16k_l0_medium \
    --threshold 0
"""

from __future__ import annotations
import gc
from pathlib import Path

import click
import torch
import tqdm
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from safetensors.torch import save_file

from sae_scoping.training.sae_enhanced.firing_rates import rank_neurons
from sae_lens import SAE


# Gemma Scope 2 — 12B IT residual-stream SAEs (25 / 50 / 65 / 85 % depth of 46 layers)
GEMMA3_12B_SAE_IDS: list[str] = [
    "resid_post/layer_12_width_16k_l0_medium",
    "resid_post/layer_24_width_16k_l0_medium",
    "resid_post/layer_31_width_16k_l0_medium",
    "resid_post/layer_41_width_16k_l0_medium",
]
GEMMA3_12B_SAE_RELEASE = "gemma-scope-2-12b-pt"

def sae_id2hookpoint(sae_id: str) -> str:
    # Format: resid_post/layer_{N}_width_{W}k_l0_{size}
    base = sae_id.split("/")[-1]  # strip "resid_post/" prefix if present
    layer_num = int(base.split("_")[1])
    return f"model.layers.{layer_num}"


def load_dataset_as_text(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    n_samples: int,
    tokenizer: PreTrainedTokenizerBase,
    question_col: str,
    answer_col: str,
    shuffle_seed: int,
) -> Dataset:
    """Load a HF question-answer dataset and apply the model's chat template."""
    load_kwargs: dict = dict(split=dataset_split, streaming=False)
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, **load_kwargs)
    else:
        ds = load_dataset(dataset_name, **load_kwargs)

    ds = ds.shuffle(seed=shuffle_seed)
    ds = ds.select(range(min(n_samples, len(ds))))

    def apply_template(example: dict) -> dict:
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": str(example[question_col])},
                {"role": "assistant", "content": str(example[answer_col])},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = ds.map(apply_template, remove_columns=ds.column_names)
    return ds


def select_neurons(
    distribution: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return neuron indices whose firing rate >= threshold, sorted descending."""
    if threshold <= 0.0:
        # Return all neurons sorted by firing rate
        return torch.argsort(distribution, descending=True)
    mask = distribution >= threshold
    indices = mask.nonzero(as_tuple=False).squeeze(1)
    # Sort by firing rate descending
    order = torch.argsort(distribution[indices], descending=True)
    return indices[order]


@click.command()
@click.option("--dataset", "-D", required=True, help="HuggingFace dataset name (e.g. 4gate/StemQAMixture)")
@click.option("--dataset-config", default=None, help="Dataset config/subset (e.g. biology, main)")
@click.option("--dataset-split", default="train", show_default=True)
@click.option("--question-col", default="question", show_default=True, help="Column name for questions")
@click.option("--answer-col", default="answer", show_default=True, help="Column name for answers")
@click.option("--n-samples", "-n", type=int, default=10_000, show_default=True)
@click.option("--batch-size", "-b", type=int, default=7, show_default=True)
@click.option("--threshold", "-t", type=float, default=1e-4, show_default=True,
              help="Firing rate threshold. Neurons with rate >= threshold are selected. "
                   "Set to 0 to return all neurons sorted by rate.")
@click.option("--ignore-padding/--no-ignore-padding", default=True, show_default=True)
@click.option("--shuffle-seed", type=int, default=1, show_default=True)
@click.option("--device", "-d", default="cuda" if torch.cuda.is_available() else "cpu", show_default=True)
@click.option("--model", "-m", default="google/gemma-3-12b-it", show_default=True)
@click.option("--sae-release", default=GEMMA3_12B_SAE_RELEASE, show_default=True)
@click.option(
    "--sae-ids", "-s", default=None,
    help="Comma-separated SAE IDs. Default: layers 8/17/22/28 width_16k l0_medium.",
)
@click.option(
    "--output-dir", "-o", default=None,
    help="Output directory. Default: .cache/<dataset_slug>/",
)
def main(
    dataset: str,
    dataset_config: str | None,
    dataset_split: str,
    question_col: str,
    answer_col: str,
    n_samples: int,
    batch_size: int,
    threshold: float,
    ignore_padding: bool,
    shuffle_seed: int,
    device: str,
    model: str,
    sae_release: str,
    sae_ids: str | None,
    output_dir: str | None,
):
    # Resolve output directory
    if output_dir is None:
        slug = dataset.replace("/", "--")
        if dataset_config:
            slug += f"--{dataset_config}"
        resolved_output = Path(__file__).parent / ".cache" / slug
    else:
        resolved_output = Path(output_dir)

    print(f"Output directory: {resolved_output}")

    # Load tokenizer & dataset
    print(f"Loading tokenizer from {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model)

    print(f"Loading dataset {dataset!r} (config={dataset_config}, split={dataset_split}, n={n_samples})...")
    text_dataset = load_dataset_as_text(
        dataset_name=dataset,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        n_samples=n_samples,
        tokenizer=tokenizer,
        question_col=question_col,
        answer_col=answer_col,
        shuffle_seed=shuffle_seed,
    )
    print(f"Dataset loaded: {len(text_dataset)} samples")
    print(f"Sample: {text_dataset[0]['text'][:200]}...")

    # Load model
    print(f"Loading model {model}...")
    lm = AutoModelForCausalLM.from_pretrained(model, device_map="cpu", torch_dtype=torch.bfloat16)
    lm = lm.to(device)

    # Determine SAE list
    sae_id_list = [s.strip() for s in sae_ids.split(",")] if sae_ids else GEMMA3_12B_SAE_IDS
    token_selection = "attention_mask" if ignore_padding else "all"

    print(f"Running {len(sae_id_list)} SAE(s) | threshold={threshold} | token_selection={token_selection}")

    for sae_id in tqdm.tqdm(sae_id_list, desc="SAEs"):
        subfolder = resolved_output / f"ignore_padding_{ignore_padding}" / sae_id.replace("/", "--")
        output_path = subfolder / "firing_rates.safetensors"

        if output_path.exists():
            print(f"Skipping {subfolder} (already exists)")
            continue

        hookpoint = sae_id2hookpoint(sae_id)
        print(f"\nLoading SAE {sae_id} (hookpoint={hookpoint})...")
        sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
        sae = sae.to(device)

        print("Running rank_neurons...")
        with torch.no_grad():
            ranking, distribution = rank_neurons(
                dataset=text_dataset,
                sae=sae,
                model=lm,
                tokenizer=tokenizer,
                T=0,
                hookpoint=hookpoint,
                batch_size=batch_size,
                token_selection=token_selection,
                return_distribution=True,
            )

        ranking = ranking.detach().cpu()
        distribution = distribution.detach().cpu()

        selected = select_neurons(distribution, threshold)
        n_selected = len(selected)
        print(f"Selected {n_selected}/{len(distribution)} neurons (threshold={threshold})")

        subfolder.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "ranking": ranking,
                "distribution": distribution,
                "selected_neurons": selected,
            },
            output_path,
        )
        print(f"Saved to {output_path}")

        sae = sae.to("cpu")
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    print("Done!")


if __name__ == "__main__":
    main()
