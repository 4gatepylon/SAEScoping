from __future__ import annotations

import json
import os
import re
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import tqdm
from beartype import beartype
from beartype.typing import Any, Callable, Literal
from datasets import Dataset

# https://docs.kidger.site/jaxtyping/api/array/
from jaxtyping import Float, Integer, jaxtyped
from sae_lens import SAE, JumpReLUSAE
from transformers import (
    Gemma2ForCausalLM,
    LlamaForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer
from utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks

# Our libraries
from utils.hooks.sae import SaeWrapper

"""
The point of this module is to provide a method that, given a dataset and an SAE-lens SAE
will optimize the model on your dataset with the SAE selected so as to retain the top
K neurons that fire on the dataset.

You may provide seperate datasets or use the same one.

The general flow is fairly simply:
```
dataset_ranking, dataset_training, sae, model, tokenizer = arguments provided to this code
for T in T's I want to try (may be dynamic):
    ranks = rank_neurons( # <---- key function here
        dataset_ranking,
        sae,
        model,
        tokenizer,
        T,
        hookpoint,
        batch_size
    )
    for K in K's I want to try (may be dynamic):
        pruned_sae = get_pruned_sae( # <---- key function here
            sae,
            ranks,
            K,
        )
        assert sae has not changed (it's OK to have two copies. mem. not that bad)
        best_accuracy = train_sae_enhanced_model( # <---- key function here
            dataset_training,
            dataset_evaluation,
            pruned_sae,
            model,
            tokenizer,
            hookpoint,
            batch_size,
        )
        possibly some logic in terms of best accuracy
        possible some evaluation logic (or that can be part of trainer)
        del pruned_sae etc... (cleanup)
```

XXX what needs to be done (sorted by priority):
1. Report the overlap
2. Analyze overlap for different models; clean up the script (use dynamic
    batch size, etc...). Report multiple SAEs' overlap tables. Doesn't have to be
    all of them. We mainly want to understand if deeper layers change qualitative
    overlap.
3. Plot distribution of magnitudes for a few exemplars. Use this to justify your choice
    of T.
4. Sweep K on inference and use this to justify your choice of strategy for training
    with K. Describe the hyperparameter optimization stategy. Report a table of sweeping
    K's effect on CE. Overlay it with the actual curve of presence at our chosen T so
    that we can understand the importance I guess.
5. Launch the trainer. It should have the following properties:
    - Should reach any K eventually (highest priority)
    - Should reach any max_steps <= 1 epoch eventually
    - Should reach an T eventually (lowest priority)
6. Write up and report some hypotheses. Next we should do the GCG/prompt optimization
    experiments and ideally some retraining experiments. For the old model I guess we
    might want judges or something? not sure... XXX tbd when we do this we also need
    to decide on RLVR.

XXX what will need to be done after we ^ (not yet planned since we need to get this ^
shit done first)
(- probably, before doing v I will go do some GCG/prefill or training the original
    models)
- Add support for better (judge) callbacks and metrics; at this point we will also want
    to evaluate qualitatively in more detail to decide whether we should move up to
    a larger model (gemma-2 9B); this is very TBD and it will come later (another day)
- Add support for the important todos, clean up code, etc... (basically it should be
    possible to train on only some tokens, apply on only some tokens, etc...)
"""


class Context:
    def __init__(self, value: Any | None = None):
        self.value = value

    def clear_value(self) -> None:
        self.value = None

    def set_value(self, value: Any) -> None:
        self.value = value


class SAELensEncDecCallbackWrapper(nn.Module):
    """
    Simple class whose purpose is to allow you to run arbitrary callbacks on SAE latents.
    The idea is that you may want to try a few different things:
    - Steering in SAE-space
    - Counting statistics on the SAE latents
    - Quantizing/modifying the SAE latents
    - etc...

    It supports:
    - passthrough=True => Don't modify DNN computation; just let the callback operate on
        the SAE latents. This is useful for gathering statistics, calculating steering
        vectors, etc... Note that there are two cases here:
            - Your callback may modify SAE latents so that the output is
                `decode(your_modification(encode(x)))`
            - Your callback may NOT modifiy SAE latents so that the output is
                `decode(encode(x))`
    - passthrough=False => Modify the DNN computation. Here there is only one case:
        - output is input
    """

    @beartype
    def __init__(
        self,
        sae: SAE,
        callback: Callable[[torch.Tensor], torch.Tensor] | nn.Module,
        passthrough: bool = False,
        defensive_passthrough_sanity_check: bool = True,
        ctx: Context | None = None,
    ):
        super().__init__()
        self.sae = sae
        self.callback = callback
        self.passthrough = passthrough
        self.defensive_passthrough_sanity_check = defensive_passthrough_sanity_check
        self.ctx = ctx

    @jaxtyped(typechecker=beartype)
    def forward(
        self, x: Float[torch.Tensor, "batch d_model"]
    ) -> Float[torch.Tensor, "batch d_model"]:
        assert x.ndim >= 1 and x.shape[-1] == self.d_in
        # 1. get encoding and run the callback
        encoding = self.sae.encode(x)
        callback_output = self.callback(encoding, self.ctx)
        # 2. Sanity check that callback is giving proper output, etc... also deal with
        # None callback (default to passthrough/identity; just syntax sugar)
        if (
            callback_output is not None
            and self.passthrough
            and self.defensive_passthrough_sanity_check
        ):
            raise ValueError(
                "callback_output is NOT None, but set self.passthrough=True. "
                + "This means your output will NOT be used! Are you sure you wanted to return a value? "
                + "To disable this raise, pass self.defensive_passthrough_sanity_check=False."
            )
        if callback_output is None:
            callback_output = encoding  # Default to pasthrough
        # 3. Decode and return
        if self.passthrough:
            return x  # Passthrough => Don't actually use SAE for later operations
        decoding = self.sae.decode(callback_output)
        assert decoding.shape == x.shape
        return decoding

    @property
    def device(self) -> torch.device:
        return self.sae.device

    @property
    def dtype(self) -> torch.dtype:
        return self.sae.dtype

    @property
    def d_sae(self) -> int:
        return self.sae.cfg.d_sae

    @property
    def d_in(self) -> int:
        return self.sae.cfg.d_in


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
    sae: SAE,
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
    if not isinstance(sae, JumpReLUSAE):
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
    sw = SaeWrapper(
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
            for i in tqdm.trange(0, len(dataset), batch_size):
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


@beartype
def _is_int(x: int | float) -> bool:
    return float(int(x)) == float(x)


# sanity check lol
assert _is_int(1)
assert _is_int(1.0)
assert not _is_int(1.1)
assert _is_int(0)
assert _is_int(-1)
assert not _is_int(0.1)


class MaskCallbackFn(nn.Module):
    """
    This is meant to be called via `SAELensEncDecCallbackWrapper`.

    I realize there may be some misconceptions about how enc/dec works across hook fns.
    The idea is simple:
    ```
    (hooking via pytorch on hf models gives us some tuples and random shit) ->
    (named_forward_hooks collects these and possibly adds hookpoint metadata) ->
    (partial(filter_hook_fn, sw) takes the tensor OUT of the tuple to pass into
        your callback fn and then puts the output of your callback fn where it was
        going to go in the tuple ->
    (SaeWrapper.forward() takes the tensor and then makes it 2D: batch x d_model for
        your callback; then when your callback returns it re-shapes it to be like the
        original version ->
    (SAELensEncDecCallbackWrapper.forward() runs encode/decode with an SAE on that and
        lets you basically hook INTO the SAE itself via your callback fn ->
    this is the callback fn at the bottom of the stack
    ```
    """

    @property
    def device(self) -> torch.device:
        return self.top_K_mask.device

    @property
    def dtype(self) -> torch.dtype:
        return self.top_K_mask.dtype

    @property
    def d_sae(self) -> int:
        assert self.neuron_indices.ndim == 1
        return self.neuron_indices.shape[0]

    @property
    def K(self) -> int:
        assert self.top_K_mask.ndim == 1
        return self.top_K_mask.sum().item()

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        neuron_indices: Integer[torch.Tensor, "d_sae"],
        K: int,
        T: float | int = 0.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bool,
    ):
        super().__init__()
        self.neuron_indices = neuron_indices
        assert neuron_indices.ndim == 1
        d_sae = neuron_indices.shape[0]
        device = neuron_indices.device if device is None else torch.device(device)
        self.top_K_mask = torch.zeros(d_sae, dtype=dtype, device=device)
        self.top_K_mask[neuron_indices[:K]] = True

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch d_model"],
        ctx: Context | None = None,  # will be passed as positional (for tokens...?)
    ) -> Float[torch.Tensor, "batch d_model"]:
        assert x.ndim >= 1 and x.shape[-1] == self.d_sae
        return x * self.top_K_mask


@beartype
def _str_dict_diff(
    found: dict[str, Any],
    expected: dict[str, Any],
    jsonifiable_fn: Callable[[Any], str] = str,
) -> str:
    assert all(isinstance(k, str) for k in found.keys())
    assert all(isinstance(k, str) for k in expected.keys())
    found2str = {k: jsonifiable_fn(v) for k, v in found.items()}
    expected2str = {k: jsonifiable_fn(v) for k, v in expected.items()}
    found_minus_expected = {k: v for k, v in found.items() if k not in expected}
    expected_minus_found = {k: v for k, v in expected.items() if k not in found}
    not_equal = {
        k: f"Found: {jsonifiable_fn(v)}. Expected: {jsonifiable_fn(expected[k])}"
        for k, v in found.items()
        if v != expected[k]
    }
    return (
        "\n"
        + "=" * 100
        + "\n"
        + f"Found: {json.dumps(found2str, indent=4)}"
        + "\n"
        + "=" * 100
        + "\n"
        + f"Expected: {json.dumps(expected2str, indent=4)}"
        + "\n"
        + "=" * 100
        + "\n"
        + f"Difference (present in both, but not equal): {json.dumps(not_equal, indent=4)}"
        + "\n"
        + "=" * 100
        + "\n"
        + f"Difference: (Found-Expected): {json.dumps(found_minus_expected, indent=4)}"
        + "\n"
        + "=" * 100
        + "\n"
        + f"Difference: (Expected-Found): {json.dumps(expected_minus_found, indent=4)}"
        + "\n"
        + "=" * 100
        + "\n"
    )


@jaxtyped(typechecker=beartype)
def get_pruned_sae(
    sae: SAE,
    neuron_indices: Integer[torch.Tensor, "d_sae"],
    K_or_p: int | float,
    T: float | int = 0.0,
) -> SAELensEncDecCallbackWrapper:
    # Validate that the weights, type, etc... of the SAE are what we expect
    if not isinstance(sae, JumpReLUSAE):
        raise ValueError("Only JumpReLUSAE is supported for now")
    found_parameters_and_shapes = {n: tuple(p.shape) for n, p in sae.named_parameters()}
    # https://github.com/decoderesearch/SAELens/blob/bd44804c64ff0d2d920fb0896635f9d9830dafab/sae_lens/saes/sae.py#L337
    # and
    # https://github.com/decoderesearch/SAELens/blob/bd44804c64ff0d2d920fb0896635f9d9830dafab/sae_lens/saes/jumprelu_sae.py#L123
    d_in, d_sae = sae.cfg.d_in, sae.cfg.d_sae
    expected_parameters_and_shapes = {
        "b_dec": (d_in,),
        "W_dec": (d_sae, d_in),
        "W_enc": (d_in, d_sae),
        "threshold": (d_sae,),
        "b_enc": (d_sae,),
    }
    if found_parameters_and_shapes != expected_parameters_and_shapes:
        raise ValueError(
            _str_dict_diff(found_parameters_and_shapes, expected_parameters_and_shapes)
        )
    # Validate that the forward pass acts exactly as we expect
    if sae.use_error_term:
        raise ValueError("SAE uses error term. Not supported for now")
    # https://github.com/decoderesearch/SAELens/blob/bd44804c64ff0d2d920fb0896635f9d9830dafab/sae_lens/saes/sae.py#L460
    _input = torch.zeros((2, d_in), device=sae.device)
    output = sae(_input)
    if not isinstance(output, torch.Tensor):
        raise ValueError(f"Output is not a torch.Tensor. Got {type(output)}")
    enc = sae.encode(_input)
    if not isinstance(enc, torch.Tensor):
        raise ValueError(f"Enc is not a torch.Tensor. Got {type(enc)}")
    if enc.shape != (2, d_sae):
        raise ValueError(f"Enc shape is {enc.shape}. Expected (2, {d_sae})")
    encdec = sae.decode(enc)
    if not isinstance(encdec, torch.Tensor):
        raise ValueError(f"Enc/Dec is not a torch.Tensor. Got {type(encdec)}")
    if not torch.allclose(output, encdec):
        raise ValueError("Output and enc/dec are not close")
    # Validate no normalization and some other parameters
    expected_config_subset = {
        "apply_b_dec_to_input": False,
        "normalize_activations": "none",
        "reshape_activations": "none",
        "architecture": "jumprelu",
    }
    # https://github.com/decoderesearch/SAELens/blob/bd44804c64ff0d2d920fb0896635f9d9830dafab/sae_lens/saes/sae.py#L448
    # ^ this is called in `encode`
    if hasattr(sae, "apply_b_dec_to_input") and sae.apply_b_dec_to_input:
        raise ValueError("SAE applies b_dec to input. Not supported for now")
    cfg_dict = sae.cfg.to_dict()
    found_config_subset = {
        k: v for k, v in cfg_dict.items() if k in expected_config_subset
    }
    if found_config_subset != expected_config_subset:
        raise ValueError(
            f"SAE config is not as expected. "
            + f"Found: {json.dumps(found_config_subset, indent=4)}. "
            + f"Expected: {json.dumps(expected_config_subset, indent=4)}"
        )
    # TODO(Adriano) fix this
    # expected_config_metadata_subset = {"model_name": "gemma-2-2b"}
    # found_config_metadata_subset = {
    #     k: v
    #     for k, v in cfg_dict["metadata"].items()
    #     if k in expected_config_metadata_subset
    # }
    # if found_config_metadata_subset != expected_config_metadata_subset:
    #     raise ValueError(
    #         f"SAE config metadata is not as expected. "
    #         + f"Found: {found_config_metadata_subset}. "
    #         + f"Expected: {expected_config_metadata_subset}"
    #     )
    if cfg_dict["metadata"]["model_name"] not in {"gemma-2-2b", "gemma-2-9b"}:
        raise ValueError(
            f"SAE model name is not supported. Got {cfg_dict['metadata']['model_name']}"
        )
    # encode: https://github.com/decoderesearch/SAELens/blob/bd44804c64ff0d2d920fb0896635f9d9830dafab/sae_lens/saes/jumprelu_sae.py#L132
    # decode: https://github.com/decoderesearch/SAELens/blob/bd44804c64ff0d2d920fb0896635f9d9830dafab/sae_lens/saes/jumprelu_sae.py#L150
    # No scaling the input
    _input = torch.randn(d_in)
    output = sae.run_time_activation_norm_fn_in(_input)
    assert torch.allclose(output, _input)
    # Nor scaling in the output
    _input = torch.randn(d_in)
    output = sae.run_time_activation_norm_fn_out(_input)
    assert torch.allclose(output, _input)
    # Validate and setup K (select how many neurons)
    if not _is_int(K_or_p) and not 0 <= K_or_p <= 1:
        raise ValueError("K_or_p must be an integer or a float between 0 and 1")
    if _is_int(K_or_p) and K_or_p > d_sae:
        raise ValueError(
            f"K must be less than or equal to d_sae. Got K={K_or_p}, d_sae={d_sae}"
        )
    if not _is_int(K_or_p):
        K_or_p = int(K_or_p * d_sae)
    assert _is_int(K_or_p), f"K_or_p is not an integer. Got {K_or_p}"
    assert 0 <= K_or_p <= d_sae, f"K_or_p is not between 0 and d_sae. Got {K_or_p}"
    if K_or_p == 0:
        raise ValueError("K_or_p cannot be 0")
    K = int(K_or_p)

    callback_fn = MaskCallbackFn(neuron_indices.to(sae.device), K, T, device=sae.device)
    return SAELensEncDecCallbackWrapper(sae, callback_fn, passthrough=False)


@beartype
def _freeze_parameters_before_layer(
    model: PreTrainedModel, sae_layer: int
) -> list[str]:
    parameters_to_freeze = []
    if type(model) not in [
        Gemma2ForCausalLM,
        LlamaForCausalLM,
    ]:
        raise ValueError(f"Model {type(model)} is not supported")
    for n, p in model.named_parameters():
        if not n.startswith("model.layers"):
            if "lm_head" in n:
                p.requires_grad = True
            if type(model) == Gemma2ForCausalLM and n.startswith("model.norm"):
                p.requires_grad = True
            else:
                # Freeze all non-layer parameters (embedding, lm_head, etc.)
                p.requires_grad = False
                if p.grad is not None:
                    p.grad = None
                parameters_to_freeze.append(n)
        else:
            # Extract layer number and freeze if before SAE layer
            patt = r"^model\.layers\.(\d+)\..*$"
            match = re.match(patt, n)
            assert match is not None, (
                f"Parameter name {n} doesn't match expected pattern"
            )
            layer_num = int(match.group(1))
            if layer_num <= sae_layer:
                p.requires_grad = False
                if p.grad is not None:
                    p.grad = None
                parameters_to_freeze.append(n)
    return parameters_to_freeze


@beartype
def train_sae_enhanced_model(
    train_dataset: Dataset,
    eval_dataset: Dataset | dict[str, Dataset],  # to eval on multiple OOD datasets
    sae: SAE | SAELensEncDecCallbackWrapper | None,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    T: float | int = 0.0,
    hookpoint: str | None = "",
    save_output: bool = False,
    return_trained_model: bool = False,
    # TODO(Adriano) add support for better callbacks
    training_callbacks: list[TrainerCallback] = [],
    sft_config: SFTConfig | None = None,  # None => use default (one below)
    **kwargs: dict[str, Any],
) -> PreTrainedModel | None:
    wandb_project_name = kwargs.get(
        "wandb_project_name", os.environ.get("WANDB_PROJECT", None)
    )
    if wandb_project_name is None:
        raise ValueError("WANDB_PROJECT is not set")
    wandb_run_name = kwargs.get(
        "wandb_run_name", os.environ.get("WANDB_RUN_NAME", None)
    )
    old_environ_name = os.environ.get("WANDB_PROJECT", None)
    try:
        # 1. setup SFT arguments
        os.environ["WANDB_PROJECT"] = wandb_project_name
        os.environ["WANDB_RUN_NAME"] = wandb_run_name
        default_sft_config = SFTConfig(
            run_name=wandb_run_name,  # None => use default
            output_dir=kwargs.get("output_dir", "./deleteme_sft_output"),
            per_device_train_batch_size=kwargs.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=kwargs.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.1,
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",
            save_steps=kwargs.get("save_steps", 1_000_000),  # Do not intend to save
            save_strategy="no",  # Do not intend to save
            logging_steps=10,
            fp16=False,
            bf16=True,  # H100/A100 hopefully will work here? Llama2
            remove_unused_columns=False,
            eval_strategy="steps",
            eval_steps=100,  # wanna do this somewhat often, but not tooo much
            save_total_limit=2,
            # load_best_model_at_end=True, # <- can't do this w/out matching save/eval strat
            # metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="wandb",
            max_steps=kwargs.get("max_steps", 1_000),
            max_length=kwargs.get("context_length", 1024),
            # TODO(adriano) don't hardcode
            gradient_checkpointing=False,
        )
        if sft_config is None:
            sft_config = default_sft_config

        # 2. freeze (and sanity check/prinout)
        # NOTE: even if no SAE, then you COULD still pass in a hookpoint to limit which
        # layers are trained
        if sae is not None and hookpoint is None:
            raise ValueError(
                "If SAE is provided, then you must also provide a hookpoint"
            )
        p2f = set()
        if hookpoint is not None:
            hp_patt = r"^model\.layers\.(\d+)$"
            if not re.match(hp_patt, hookpoint):
                raise ValueError(
                    f"Hookpoint {hookpoint} is not a valid layer hookpoint"
                )
            sae_layer = int(re.match(hp_patt, hookpoint).group(1))
            p2f = set(_freeze_parameters_before_layer(model, sae_layer))
        trainable_params_be4 = sorted(
            [n for n, p in model.named_parameters() if p.requires_grad]
        )
        frozen_params_be4 = sorted(
            [n for n, p in model.named_parameters() if not p.requires_grad]
        )
        print("hookpoint: ", hookpoint)
        print(
            f"Trainable params @ hookpoint={hookpoint}: {json.dumps(trainable_params_be4, indent=4)}"
        )
        print(
            f"Frozen params @ hookpoint={hookpoint}: {json.dumps(frozen_params_be4, indent=4)}"
        )
        assert set(frozen_params_be4) == p2f
        assert (set(trainable_params_be4) & p2f) == set()

        # copy a small word; surely the words will change w.h.p. or smth?
        p2s1 = {
            n: p.data.detach().view(-1)[:32].cpu() for n, p in model.named_parameters()
        }

        # 3. Setup and train
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=training_callbacks,
        )
        trainable_params_after = sorted(
            [n for n, p in model.named_parameters() if p.requires_grad]
        )
        frozen_params_after = sorted(
            [n for n, p in model.named_parameters() if not p.requires_grad]
        )
        assert trainable_params_be4 == trainable_params_after
        assert frozen_params_be4 == frozen_params_after
        if sae is not None:
            # This will work no matter which of the above types you use
            sae_wrapper = SaeWrapper(sae)
            hook_dict = {hookpoint: partial(filter_hook_fn, sae_wrapper)}
            with named_forward_hooks(model, hook_dict):
                trainer.train()
        else:
            trainer.train()

        # Sanity
        trainable_params_end = sorted(
            [n for n, p in model.named_parameters() if p.requires_grad]
        )
        frozen_params_end = sorted(
            [n for n, p in model.named_parameters() if not p.requires_grad]
        )
        assert trainable_params_be4 == trainable_params_end
        assert frozen_params_be4 == frozen_params_end

        # SAnity check we learend but not on the forzen ones
        parameters_that_changed = []
        for n, p in model.named_parameters():
            slc = p.data.detach().view(-1)[:32].cpu()
            if not torch.allclose(slc, p2s1[n]):
                parameters_that_changed.append(n)
        # fmt: off
        assert set(parameters_that_changed) == set(trainable_params_end), f"Parameters that changed: {json.dumps(list(parameters_that_changed), indent=4)}\n\nShould be: {json.dumps(list(trainable_params_end), indent=4)}"
        assert len(set(parameters_that_changed) & set(frozen_params_end)) == 0, f"Parameters that changed and are frozen: {json.dumps(list(parameters_that_changed & set(frozen_params_end)), indent=4)}\n\nShould be empty"
        assert len(set(parameters_that_changed) & p2f) == 0, f"Parameters that changed and are frozen: {json.dumps(list(parameters_that_changed & p2f), indent=4)}\n\nShould be empty"
        # fmt: on

        if save_output:
            trainer.save_model()
        if return_trained_model:
            return model

    finally:
        if old_environ_name is not None:
            os.environ["WANDB_PROJECT"] = old_environ_name


if __name__ == "__main__":

    def test_end2end():
        # Try a simple integration test with a dummy model and gemmascope
        print("=" * 100)
        print("Importing dependencies")
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("=" * 100)
        print("Loading model and tokenizer")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b", device_map="cpu"
        )
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        tokenizer_chat = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        assert tokenizer_chat.chat_template is not None

        print("=" * 100)
        print("Loading dataset")
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        dataset = dataset.shuffle(seed=1)
        n_samples_ranking = 100
        n_samples_training = 200
        n_samples_evaluation = 50
        batch_size = 128
        n_samples_total = n_samples_ranking + n_samples_training + n_samples_evaluation
        assert len(dataset) >= n_samples_total
        dataset = dataset.select(range(n_samples_total))
        dataset = dataset.map(
            lambda x: {
                "text": tokenizer_chat.apply_chat_template(
                    [
                        {"role": "user", "content": x["question"]},
                        {"role": "assistant", "content": x["answer"]},
                    ],
                    tokenize=False,
                )
            },
            batched=False,  # It's pretty fast so should be fine tbh
        )
        dataset_ranking = dataset.select(range(n_samples_ranking))
        dataset_training = dataset.select(
            range(n_samples_ranking, n_samples_ranking + n_samples_training)
        )
        dataset_evaluation = dataset.select(
            range(n_samples_ranking + n_samples_training, n_samples_total)
        )

        print("=" * 100)
        print("Loading SAE and moving all models to device")
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        model = model.to(device)
        sae = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id="layer_0/width_16k/canonical",
            device=device,
        )
        sae = sae.to(device)  # defensive likely-noop
        print("Sanity check types:")
        print(type(sae))
        print(type(sae).__mro__)
        assert isinstance(sae, JumpReLUSAE)  # ^

        print("=" * 100)
        print("Creating hookpoint and ranking neurons")
        # hookpoint = "blocks.0.hook_resid_post" # this is the HookedTransformer hookpoint
        hookpoint = "model.layers.0"  # register as post-hook; default

        T = 0
        p = 0.5  # Keep the top 50% of neurons
        batch_size = 32
        ranking, distribution = rank_neurons(
            dataset=dataset_ranking,
            sae=sae,
            model=model,
            tokenizer=tokenizer,
            T=T,
            hookpoint=hookpoint,
            batch_size=batch_size,
            return_distribution=True,
        )

        @jaxtyped(typechecker=beartype)
        def plot_distribution(
            distribution: Float[torch.Tensor, "d_sae"],
            file_path: Path | None = None,
        ) -> None:
            distribution_np = distribution.detach().cpu().numpy()
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(distribution_np)), distribution_np)
            plt.xlabel("Neuron index")
            plt.ylabel("Firing count")
            plt.title("Firing count distribution")
            if file_path is not None:
                plt.savefig(file_path)
            else:
                plt.show()

        save_distribution_unsorted_path = (
            Path(__file__).parent / "deleteme_fc_unsorted.png"
        )
        save_distribution_sorted_path = Path(__file__).parent / "deleteme_fc_sorted.png"
        plot_distribution(distribution, save_distribution_unsorted_path)
        plot_distribution(distribution[ranking], save_distribution_sorted_path)

        print("=" * 100)
        print("Getting pruned SAE (just wrapper tbh)")
        pruned_sae: SAELensEncDecCallbackWrapper = get_pruned_sae(sae, ranking, p, T)
        print(pruned_sae)

        print("=" * 100)
        print("Running inference with pruned SAE (to make sure it works OK)")
        sw = SaeWrapper(pruned_sae)
        fn = partial(filter_hook_fn, sw)
        hook_dict = {hookpoint: fn}
        with torch.no_grad():
            for i in tqdm.trange(0, len(dataset_evaluation), batch_size):
                texts = dataset_evaluation["text"][
                    i : min(i + batch_size, len(dataset_evaluation))
                ]
                assert all(isinstance(text, str) for text in texts)
                batch = tokenizer(
                    texts, return_tensors="pt", padding=True, truncation=True
                )
                batch = {k: v.to(device) for k, v in batch.items()}
                batch["labels"] = batch["input_ids"]  # For loss calculation; hf shifts
                with named_forward_hooks(model, hook_dict):
                    loss_with_sae = model(**batch).loss
                loss_without_sae = model(**batch).loss
                info_printout = {
                    "Loss w/ SAE": loss_with_sae.item(),
                    "Loss w/o SAE": loss_without_sae.item(),
                }
                print(json.dumps(info_printout, indent=4))

        print("=" * 100)
        print("Training SAE-enhanced model")
        train_sae_enhanced_model(
            train_dataset=dataset_training,
            eval_dataset=dataset_evaluation,
            sae=pruned_sae,
            model=model,
            tokenizer=tokenizer,
            T=T,
            hookpoint=hookpoint,
            save_output=False,
            training_callbacks=[],
            sft_config=None,  # Use default one
            # These get populated into the SFT config
            wandb_project_name="deleteme_gemma_sae_recovery_training",
            wandb_run_name="debugging/layer_0--width_16k--canonical",
        )

    test_end2end()
