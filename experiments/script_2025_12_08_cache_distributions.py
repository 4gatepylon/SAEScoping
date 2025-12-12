from __future__ import annotations
import click
import re
import itertools
from pathlib import Path
import gc
from jaxtyping import Integer, Float
from beartype import beartype
import torch
import tqdm
from datasets import Dataset, concatenate_datasets
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Gemma2ForCausalLM,
    AutoTokenizer,
)
from safetensors.torch import save_file
from sae_scoping.datasets.text_datasets import (
    get_camel_ai_biology_dataset,
    get_megascience_biology_dataset,
    load_apps,
    load_ultrachat_dataset,
)
from sae_scoping.trainers.sae_enhanced.rank import rank_neurons


# Copied
def sae_id2hookpoint(sae_id: str | None) -> str:
    if sae_id is None:
        return None
    assert re.match(r"^layer_\d+/width_16k/canonical$", sae_id)
    layer_num = int(sae_id.split("/", 1)[0].split("_")[1])
    return f"model.layers.{layer_num}"


"""
This module has one job and one job only: to provide you with both a library and click
CLI to calculate and store all the distributions of triggerings on SAE neurons for
your dataset (or one of the defaults from the datasets in `text_datasets`).

Importantly, this is primarily meant to support chat-templated models and is used for
our 9B Gemma-2 models with gemma-scope SAEs.

NOTE that it DUPLICATES CODE. That should be fixed up in the near future. The reason
is that this is basically part of `thresholded_sae_lens_recovery_training.py` (but
a more beefed up version of the part that just collects the distributions of
activations).

This supports BOTH:
- SAE Lens: 
- Eleuther Sparsify: https://github.com/EleutherAI/sparsify (which is used for TopK
    SAEs, primarily), and any "sae" callable that specifically has: `encode` and
    `decode` methods.

And for callbakcs it ONLY supports:
- Getting counts per neuron (which lets you get distributions of firing rates per
    neuron)
(Joint distribution i.e. with covariances, etc... is not supported yet and may be in the
future).
"""


# TODO please support this shit
# def check_if_enc_dec_compatible(
#     sae: SAE | SparseCoder | Any,
#     d_in: int | None = None,  # In case it matters but not an attribute
#     need_decode: bool = True,  # If not, then you can both use sparsify/sae-lens
# ) -> tuple[bool, str]:
#     if not hasattr(sae, "encode"):
#         return False, "SAE does not have an encode method"
#     if not hasattr(sae, "decode"):
#         return False, "SAE does not have a decode method"
#     if d_in is None:
#         try:
#             d_in = sae.cfg.d_in
#         except AttributeError:
#             try:
#                 d_in = sae.d_in
#             except AttributeError:
#                 d_in = 4096  # ?
#     with torch.no_grad():
#         x = torch.randn(10, d_in, requires_grad=False)
#         x = x.to(sae.dtype).to(sae.device)
#         enc = sae.encode(x)  # technically allow any type since you write callback
#         # TODO(Adriano) have some kind of strict mode
#         # TODO(Adriano), see what we want to do about this... (there are questions here
#         # because sparsify requires top_indices and some other stuff, look below:
#         # https://github.com/EleutherAI/sparsify/blob/2177983136a1447c25115a692a65dac0bd518779/sparsify/sparse_coder.py#L198)
#         if need_decode:
#             dec = sae.decode(enc)
#             if dec.shape != x.shape:
#                 return (
#                     False,
#                     f"Decoded shape {dec.shape} does not match input shape {x.shape}",
#                 )
#         return True, "SAE is (probably) compatible (if the behavior is deterministic)"

# GEMMA2_9B_SAE_IDS: list[str] = [
#     # https://huggingface.co/google/gemma-scope-9b-pt-res/tree/main
#     f"layer_{i}/width_16k/canonical"
#     for i in range(0, 50, 1)
# ]
# https://github.com/decoderesearch/SAELens/blob/bd44804c64ff0d2d920fb0896635f9d9830dafab/sae_lens/pretrained_saes.yaml#L3321
# We want to use IT
GEMMA2_9B_SAE_IDS: list[str] = [
    f"layer_{layer}/width_{width}/canonical"
    for layer, width in [
        (9, "16k"),
        (20, "16k"),
        (31, "16k"),
        # We do not use these because they are big af lol
        # (9, "131k"),
        # (20, "131k"),
        # (31, "131k"),
    ]
]

GEMMA2_9B_SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"


@beartype
def get_biology_chat_dataset_dict(
    n_samples: int = 10_000, tokenizer: PreTrainedTokenizerBase | None = None
) -> Dataset:
    assert n_samples >= 6, "n_samples must be greater than or equal to 3"
    assert n_samples % 2 == 0, "n_samples must be even"
    camel_dd = get_camel_ai_biology_dataset(
        n_samples_ranking=n_samples // 2 - 2,
        n_samples_training=1,
        n_samples_evaluation=1,
        seed=1,
        verbose=True,
        qa_templatting_function=tokenizer,
    )
    assert set(camel_dd.keys()) == {"ranking", "training", "evaluation"}
    assert {"text"} <= set(camel_dd["ranking"].column_names)
    assert {"text"} <= set(camel_dd["training"].column_names)
    assert {"text"} <= set(camel_dd["evaluation"].column_names)

    megascience_dd = get_megascience_biology_dataset(
        n_samples_ranking=n_samples // 2 - 2,
        n_samples_training=1,
        n_samples_evaluation=1,
        seed=1,
        verbose=True,
        qa_templatting_function=tokenizer,
    )
    assert set(megascience_dd.keys()) == {"ranking", "training", "evaluation"}
    assert {"text"} <= set(megascience_dd["ranking"].column_names)
    assert {"text"} <= set(megascience_dd["training"].column_names)
    assert {"text"} <= set(megascience_dd["evaluation"].column_names)

    # Remove all non-text columns for merge
    for dd in [camel_dd, megascience_dd]:
        for k in dd.keys():
            d = dd[k]
            for col in d.column_names:
                if col != "text":
                    d = d.remove_columns([col])
            dd[k] = d

    # Merge ALL this shit
    big_dataset = concatenate_datasets(
        [
            camel_dd["ranking"],
            megascience_dd["ranking"],
            camel_dd["training"],
            megascience_dd["training"],
            camel_dd["evaluation"],
            megascience_dd["evaluation"],
        ]
    )
    assert set(big_dataset.column_names) == {"text"}, (
        f"Column names are: {big_dataset.column_names}"
    )
    assert len(big_dataset) == n_samples, (
        f"Dataset has {len(big_dataset)} samples but {n_samples} were requested"
    )
    return big_dataset


@beartype
def rank_neurons_shim(
    tokenized: Dataset | list[dict[str, torch.Tensor]],
    sae_id: str,
    sae_release: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int | None = None,
    T: float | int = 0.0,
    ignore_padding: bool = True,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[Integer[torch.Tensor, "d_sae"], Float[torch.Tensor, "d_sae"]]:
    sae: SAE = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )
    sae = sae.to(device)
    hookpoint = sae_id2hookpoint(sae_id)
    with torch.no_grad():
        ranking, distribution = rank_neurons(
            dataset=tokenized,
            sae=sae,
            model=model,
            tokenizer=tokenizer,
            T=T,
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


# TODO(Adriano) don't hardcode lol plz
@click.command()
@click.option("--datasets", "-d", type=str, default="biology,apps,ultrachat")
@click.option("--ignore_paddings", "-i", type=str, default="True,False")
@click.option("--batch-size", "-b", type=int, default=7)
def cli(datasets: str, ignore_paddings: str, batch_size: int):
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Load dataset and tokenize it
    print("=" * 100)
    print("Loading dataset and tokenizing it...")
    model_name = "google/gemma-2-9b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    assert tokenizer.pad_token is not None
    dataset = get_biology_chat_dataset_dict(
        n_samples=10_000,
        tokenizer=tokenizer,
    )
    apps_dataset = load_apps(  # eh whatever
        n_samples_ranking=1_000,
        n_samples_training=1,
        n_samples_evaluation=1,
        seed=1,
        verbose=True,
        # tokenizer=tokenizer,
    )["ranking"]
    apps_dataset = apps_dataset.remove_columns(
        [col for col in apps_dataset.column_names if col != "text"]
    )
    ultrachat_dataset = load_ultrachat_dataset(
        n_samples_ranking=1_000,
        n_samples_training=1,
        n_samples_evaluation=1,
        seed=1,
        verbose=True,
        tokenizer=tokenizer,
    )["ranking"]
    ultrachat_dataset = ultrachat_dataset.remove_columns(
        [col for col in ultrachat_dataset.column_names if col != "text"]
    )
    assert set(dataset.column_names) == {"text"}
    n_views = 30
    print("=" * 100)
    for i in range(n_views):
        print(dataset[i]["text"])
        print("=" * 100)

    print("=" * 100)

    # 2. Load model
    model = Gemma2ForCausalLM.from_pretrained(
        "google/gemma-2-9b-it", device_map="cpu", torch_dtype=torch.bfloat16
    )
    model = model.to(device)  # you sohuld have set cuda visible devices

    # 3. For each SAE, run through inference on this
    output_folder = Path(__file__).parent / ".cache"
    datasets_and_names = [
        (dataset, "biology"),
        (apps_dataset, "apps"),
        (ultrachat_dataset, "ultrachat"),
    ]
    datasets = list(set(list(map(str.strip, datasets.split(",")))))
    datasets_and_names = [x for x in datasets_and_names if x[1] in datasets]

    def to_bool(x: str) -> bool:
        return x.lower().strip() == "true"

    ignore_paddings = list(set(list(map(to_bool, ignore_paddings.split(",")))))
    combos = list(
        itertools.product(datasets_and_names, ignore_paddings, GEMMA2_9B_SAE_IDS)
    )
    print("=" * 100)
    print(f"WILL ITERATE FOR {len(combos)} COMBOS")
    print("=" * 100)
    for (dataset, dataset_name), ignore_padding, sae_id in tqdm.tqdm(
        combos, desc="Processing datasets..."
    ):
        subfolder = (
            output_folder
            / f"ignore_padding_{ignore_padding}"
            / dataset_name
            / sae_id.replace("/", "--")
        )
        if subfolder.exists():
            continue
        _, distribution = rank_neurons_shim(
            tokenized=dataset,
            sae_id=sae_id,
            sae_release=GEMMA2_9B_SAE_RELEASE,
            model=model,
            batch_size=batch_size,
            tokenizer=tokenizer,
            T=0,
            ignore_padding=ignore_padding,
        )

        assert not subfolder.exists(), f"Subfolder {subfolder} already exists"
        subfolder.mkdir(parents=True, exist_ok=True)
        # Distribution IMPLIES ranking so we don't need to do anything there.
        save_file(
            {"distribution": distribution},
            subfolder / "distribution.safetensors",
        )


if __name__ == "__main__":
    cli()
