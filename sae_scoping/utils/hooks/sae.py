from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import sae_lens
import sparsify
from typing import List, Optional, Callable


class SaeWrapper(
    nn.Module
):  # XXX move this somewhere else I think (probably trainer!) or call it smth else
    def __init__(
        self,
        # All this really needs is a __call__ fn
        sae: sparsify.SparseCoder
        | sae_lens.SAE
        | nn.Module
        | Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        self.sae = sae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Flatten and keep track of the original shape
        # NOTE assume batch indices are all except last (i.e. batch, token, etc...)
        d_model = x.shape[-1]  # OG
        ds_x = x.shape[:-1]  # OG
        d_x = np.prod(ds_x).item()  # flat shape
        x = x.reshape(d_x, d_model)  # flat
        sae_out = self.sae(x.to(self.sae.dtype))
        # sae-lens vs. sparsify... (hotfix)
        out = sae_out if isinstance(sae_out, torch.Tensor) else sae_out.sae_out
        out = out.to(x.dtype)
        assert out.shape == x.shape  # assert same shape
        assert out.dtype == x.dtype
        return out.reshape(*ds_x, d_model)
