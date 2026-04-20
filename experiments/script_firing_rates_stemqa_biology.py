"""
Compute SAE neuron firing rates on 4gate/StemQAMixture (biology split)
using google/gemma-2-9b-it with gemma-scope SAEs.

Saves firing counts and distributions to experiments/.cache/stemqa_biology/
"""

from __future__ import annotations
import gc
import itertools
from pathlib import Path

import click
import torch
import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, Gemma2ForCausalLM, PreTrainedTokenizerBase
from safetensors.torch import save_file

from sae_scoping.training.sae_enhanced.firing_rates import rank_neurons
from sae_scoping.datasets.text_datasets import get_qa_dataset_dict

import sae_lens
from sae_lens import SAE


GEMMA2_9B_SAE_IDS: list[str] = [
    f"layer_{layer}/width_{width}/canonical"
    for layer, width in [
        (9, "16k"),
        (20, "16k"),
        (31, "16k"),
    ]
]
GEMMA2_9B_SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"


def sae_id2hookpoint(sae_id: str) -> str:
    layer_num = int(sae_id.split("/", 1)[0].split("_")[1])
    return f"model.layers.{layer_num}"


def load_stemqa_biology(
    n_samples_ranking: int,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    """Load 4gate/StemQAMixture biology split formatted with chat template."""
    dataset = get_qa_dataset_dict(
        dataset_name="4gate/StemQAMixture",
        n_samples_ranking=n_samples_ranking,
        n_samples_training=1,
        n_samples_evaluation=1,
        shuffle_seed=1,
        args=["biology"],
        splits=["train"],
        question_column_name="question",
        answer_column_name="answer",
        qa_templatting_function=tokenizer,
        format_question_key=None,
        format_answer_key=None,
        text_column_name="text",
        verbose=True,
    )
    ranking_ds = dataset["ranking"]
    # Keep only "text" column
    ranking_ds = ranking_ds.remove_columns(
        [col for col in ranking_ds.column_names if col != "text"]
    )
    return ranking_ds


@click.command()
@click.option("--n-samples", "-n", type=int, default=10_000)
@click.option("--batch-size", "-b", type=int, default=7)
@click.option("--ignore-padding/--no-ignore-padding", default=True)
@click.option("--device", "-d", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
@click.option(
    "--sae-ids",
    "-s",
    type=str,
    default=None,
    help="Comma-separated SAE IDs to use. Default: all 3 (layer 9, 20, 31)",
)
def main(n_samples: int, batch_size: int, ignore_padding: bool, sae_ids: str, device: str | None):
    output_folder = Path(__file__).parent / ".cache" / "stemqa_biology"

    # 1. Load tokenizer and dataset
    model_name = "google/gemma-2-9b-it"
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading StemQAMixture biology ({n_samples} samples)...")
    dataset = load_stemqa_biology(n_samples, tokenizer)
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Sample: {dataset[0]['text'][:200]}...")

    # 2. Load model
    print(f"Loading model {model_name}...")
    model = Gemma2ForCausalLM.from_pretrained(
        model_name, device_map="cpu", torch_dtype=torch.bfloat16
    )
    model = model.to(device)

    # 3. Determine which SAEs to run
    if sae_ids is not None:
        sae_id_list = [s.strip() for s in sae_ids.split(",")]
    else:
        sae_id_list = GEMMA2_9B_SAE_IDS

    ignore_paddings = [ignore_padding]

    combos = list(itertools.product(ignore_paddings, sae_id_list))
    print(f"Running {len(combos)} combinations...")

    # 4. Run ranking for each SAE
    for ign_pad, sae_id in tqdm.tqdm(combos, desc="Processing SAEs"):
        subfolder = (
            output_folder
            / f"ignore_padding_{ign_pad}"
            / sae_id.replace("/", "--")
        )
        if subfolder.exists():
            print(f"Skipping {subfolder} (already exists)")
            continue

        print(f"\nLoading SAE {sae_id}...")
        sae = SAE.from_pretrained(
            release=GEMMA2_9B_SAE_RELEASE,
            sae_id=sae_id,
            device=device,
        )
        sae = sae.to(device)
        hookpoint = sae_id2hookpoint(sae_id)

        print(f"Running rank_neurons (hookpoint={hookpoint}, ignore_padding={ign_pad})...")
        with torch.no_grad():
            ranking, distribution = rank_neurons(
                dataset=dataset,
                sae=sae,
                model=model,
                tokenizer=tokenizer,
                T=0,
                hookpoint=hookpoint,
                batch_size=batch_size,
                token_selection="attention_mask" if ign_pad else "all",
                return_distribution=True,
            )

        ranking = ranking.detach().cpu()
        distribution = distribution.detach().cpu()

        subfolder.mkdir(parents=True, exist_ok=True)
        save_file(
            {"ranking": ranking, "distribution": distribution},
            subfolder / "firing_rates.safetensors",
        )
        print(f"Saved to {subfolder / 'firing_rates.safetensors'}")

        # Cleanup SAE
        sae = sae.to("cpu")
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    print("Done!")


if __name__ == "__main__":
    main()
