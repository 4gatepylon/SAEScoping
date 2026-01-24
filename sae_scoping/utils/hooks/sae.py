from __future__ import annotations

import torch
from beartype import beartype
from beartype.typing import Any, Callable
import torch.nn as nn
import sae_lens
import sparsify
from jaxtyping import Float, jaxtyped
from sae_scoping.utils.hooks.pt_hooks_stateful import Context


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
        sae: sparsify.SparseCoder | sae_lens.SAE,
        callback: Callable[[torch.Tensor], torch.Tensor] | nn.Module,
        passthrough: bool = False,
        defensive_passthrough_sanity_check: bool = True,
        ctx: Context | None = None,
    ):
        super().__init__()
        if not isinstance(sae, (sae_lens.SAE,)):
            raise NotImplementedError(f"sae must be a sae_lens.SAE only for now, got {type(sae)}")
        self.sae = sae
        self.callback = callback
        self.passthrough = passthrough
        self.defensive_passthrough_sanity_check = defensive_passthrough_sanity_check
        self.ctx = ctx

    @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[torch.Tensor, "batch d_model"]) -> Float[torch.Tensor, "batch d_model"]:
        assert x.ndim >= 1 and x.shape[-1] == self.d_in
        # 1. get encoding and run the callback
        encoding = self.sae.encode(x)
        callback_output = self.callback(encoding, self.ctx)
        # 2. Sanity check that callback is giving proper output, etc... also deal with
        # None callback (default to passthrough/identity; just syntax sugar)
        if callback_output is not None and self.passthrough and self.defensive_passthrough_sanity_check:
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


class SAEWrapper(nn.Module):
    @beartype
    def __init__(self, sae: sparsify.SparseCoder | sae_lens.SAE | SAELensEncDecCallbackWrapper) -> None:
        super().__init__()
        self.sae = sae

    @beartype
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Flatten and keep track of the original shape
        # NOTE assume batch indices are all except last (i.e. batch, token, etc...)
        d_model = x.shape[-1]  # OG
        ds_x = x.shape[:-1]  # OG
        d_x = torch.prod(torch.tensor(ds_x)).item()  # flat shape
        x = x.reshape(d_x, d_model)  # flat
        sae_out = self.sae(x.to(self.sae.dtype))
        # sae-lens vs. sparsify... (hotfix)
        out = sae_out if isinstance(sae_out, torch.Tensor) else sae_out.sae_out
        out = out.to(x.dtype)
        assert out.shape == x.shape  # assert same shape
        assert out.dtype == x.dtype
        return out.reshape(*ds_x, d_model)
