from __future__ import annotations
import torch
from beartype import beartype
from jaxtyping import Float, Integer, jaxtyped
import torch.nn as nn
import sparsify
import sae_lens
import json
from sae_scoping.utils.hooks.pt_hooks_stateful import Context
from sae_scoping.utils.hooks.sae import SAELensEncDecCallbackWrapper
from sae_scoping.trainers.sae_enhanced.utils import str_dict_diff, is_int

"""
Support features to convert an SAE into a "pruned" version that may not use some of the
neurons.
"""


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
    (SAEWrapper.forward() takes the tensor and then makes it 2D: batch x d_model for
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


@jaxtyped(typechecker=beartype)
def get_pruned_sae(
    sae: sparsify.SparseCoder | sae_lens.SAE,
    neuron_indices: Integer[torch.Tensor, "d_sae"],
    K_or_p: int | float,
    T: float | int = 0.0,
) -> SAELensEncDecCallbackWrapper:
    # Validate that the weights, type, etc... of the SAE are what we expect
    if not isinstance(sae, sae_lens.JumpReLUSAE):
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
            str_dict_diff(found_parameters_and_shapes, expected_parameters_and_shapes)
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
    if cfg_dict["metadata"]["model_name"] not in {"google/gemma-3-12b-it", "google/gemma-3-27b-it", "google/gemma-2-9b-it", "gemma-2-9b-it", "gemma-2-9b"}:
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
    if not is_int(K_or_p) and not 0 <= K_or_p <= 1:
        raise ValueError("K_or_p must be an integer or a float between 0 and 1")
    if is_int(K_or_p) and K_or_p > d_sae:
        raise ValueError(
            f"K must be less than or equal to d_sae. Got K={K_or_p}, d_sae={d_sae}"
        )
    if not is_int(K_or_p):
        K_or_p = int(K_or_p * d_sae)
    assert is_int(K_or_p), f"K_or_p is not an integer. Got {K_or_p}"
    assert 0 <= K_or_p <= d_sae, f"K_or_p is not between 0 and d_sae. Got {K_or_p}"
    if K_or_p == 0:
        raise ValueError("K_or_p cannot be 0")
    K = int(K_or_p)

    callback_fn = MaskCallbackFn(neuron_indices.to(sae.device), K, T, device=sae.device)
    return SAELensEncDecCallbackWrapper(sae, callback_fn, passthrough=False)
