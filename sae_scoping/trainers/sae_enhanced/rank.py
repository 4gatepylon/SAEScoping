from __future__ import annotations
from beartype import beartype
from beartype.typing import Literal
from jaxtyping import Float, Integer, jaxtyped
import torch
import sae_lens
import sparsify
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from functools import partial
from tqdm import tqdm, trange
from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
from sae_scoping.utils.hooks.pt_hooks_stateful import Context
from sae_scoping.utils.hooks.sae import (
    SAELensEncDecCallbackWrapper,
    SAEWrapper,
)


"""
Support functionality to collect activations from an SAE's latent space and calculate
firing counts (among possibly other things in the future). This is used to select which
features to retain during pruning.
"""


@jaxtyped(typechecker=beartype)
def accumulate_firing_counts_callback_fn(
    firing_counts: Integer[torch.Tensor, "d_sae"],
    T: float | int,
    encoding: Float[torch.Tensor, "batch d_sae"],
    ctx: Context | None = None,
) -> None:  # Not meant to passthrough
    attention_mask = None
    if ctx is not None:
        value = ctx.value
        if isinstance(value, dict) and "attention_mask" in value:
            print(f"USING CONTEXT: {value['attention_mask'].shape}")
            attention_mask = value["attention_mask"]
        else:
            raise ValueError(
                f"ctx.value is not a dict or does not contain 'attention_mask'. Got {type(value)}"
            )
    assert attention_mask is None or attention_mask.shape == encoding.shape[:-1], (
        f"attention_mask shape {attention_mask.shape}, encoding shape {encoding.shape}"
    )
    if attention_mask is not None:
        # zero out to not count the padding tokens
        encoding = encoding.detach() * attention_mask.detach().unsqueeze(-1)
    firing_counts += (encoding.detach() > T).sum(dim=0)


@jaxtyped(typechecker=beartype)
def rank_neurons(
    dataset: Dataset | list[dict[str, torch.Tensor]],
    sae: sae_lens.SAE | sparsify.SparseCoder,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    T: float | int = 0.0,
    hookpoint: str = "",
    batch_size: int | None = 128,
    context_length: int = 1024,
    return_distribution: (
        Literal["no", "fraction", "counts", "histograms", "magnitudes"] | bool
    ) = False,
    histograms_n_bins: int = 100,
    token_selection: Literal["all", "attention_mask"] = "all",
) -> tuple[Integer[torch.Tensor, "d_sae"], Float[torch.Tensor, "d_sae"] | None]:
    """
    Return argsort order of neurons by frequency of firing. Next return the distribution
    of firing counts as well.

    Return distribution gives us the following options:
    - "no" => Don't return distribution
    - "fraction" => Return distribution of the firing rates; this loses information about
        the total number of data-points and size of the firing (magnitude of the vector)
    - "counts" => Return per neuron how many times it fired
    - "histograms" => Return histograms of the firing magnitudes
    - "magnitudes" => Return the magnitudes of the firing vectors (n_dataset x d_sae)
        per-token selected
    """
    if (batch_size is not None) != isinstance(dataset, Dataset):
        raise ValueError(f"batch_size none IFF Dataset")
    if isinstance(return_distribution, bool):
        return_distribution = "fraction" if return_distribution else "no"
    if return_distribution not in ["no", "fraction"]:
        raise ValueError(f"Invalid return distribution: {return_distribution}")
    return_distribution = True if return_distribution == "fraction" else False
    if not isinstance(sae, (sae_lens.JumpReLUSAE,)):
        raise ValueError("Only JumpReLUSAE is supported for now")
    if len(hookpoint.strip()) == 0:
        raise ValueError("hookpoint must be provided")
    d_sae = sae.cfg.d_sae
    device = sae.device
    if isinstance(dataset, Dataset):
        assert {"text"} <= set(dataset.column_names)
        assert all(isinstance(text, str) for text in dataset["text"])
    # 1. setup accumulation and hooking
    ctx = None
    if token_selection == "attention_mask":
        ctx = Context(value=None)  # right now, nothing, gets set before forwards
    firing_counts = torch.zeros(d_sae, dtype=torch.long, device=device)
    sw = SAEWrapper(
        SAELensEncDecCallbackWrapper(
            sae,
            partial(accumulate_firing_counts_callback_fn, firing_counts, T),
            passthrough=True,
            ctx=ctx,
        )
    )
    hook_dict = {
        hookpoint: partial(filter_hook_fn, sw)
    }  # TODO change this to accumulate into firing counts
    # 2. Run inference (tokenize just-in-time to save memory; shouldn't matter though)
    with torch.no_grad():
        with named_forward_hooks(model, hook_dict):
            if batch_size is None:
                batch_size = 1  # step by 1 each time
            for i in trange(0, len(dataset), batch_size):
                if isinstance(dataset, Dataset):
                    texts = dataset["text"][i : min(i + batch_size, len(dataset))]
                    assert all(isinstance(text, str) for text in texts)
                    batch = tokenizer(
                        texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=context_length,
                    )
                    batch = {k: v.to(device) for k, v in batch.items()}
                else:
                    batch = dataset[i]
                    batch = {k: v.to(device) for k, v in batch.items()}
                assert (
                    isinstance(batch, dict)
                    and all(isinstance(k, str) for k in batch.keys())
                    and all(isinstance(v, torch.Tensor) for v in batch.values())
                ), (
                    f"type(batch) is {type(batch)}, batch.keys() "
                    + f"is {None if not isinstance(batch.keys(), dict) else batch.keys()}"
                )
                batch = {k: v.to(device) for k, v in batch.items()}  # low mem. so OK
                if ctx is not None:
                    assert "attention_mask" in batch
                    # The ctx reader in the hook expects flat
                    ctx.set_value({"attention_mask": batch["attention_mask"].flatten()})
                model(**batch)
    # 4. Sanity and return
    assert firing_counts.min().item() >= 0
    assert firing_counts.max().item() > 0
    ranks = firing_counts.argsort(dim=0, descending=True)
    distribution = None
    if return_distribution:
        distribution = firing_counts / firing_counts.sum()
    return ranks, distribution
