from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import sae_lens
import sparsify
from typing import List, Optional, Callable


class SaeWrapper(nn.Module):
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


class ActivationsCollector:
    """
    A trivial utility to store activations. NOTE that there is no support for any kind
    of validation of the shapes, etc...

    You can optionally provide a function to "preprocess" the activations before
    collecting them. This can enable you to do things like normalize or instead of
    collecting activations, collecting classifications or logprobs, etc... for them.
    """

    def __init__(
        self,
        collect: bool = True,
        preproc_fn_or_nn_module: Optional[
            Callable[[torch.Tensor], torch.Tensor] | nn.Module
        ] = None,
    ):
        self.enabled_collect: bool = collect
        self.activations: List[torch.Tensor] = []
        self.preproc_fn_or_nn_module = preproc_fn_or_nn_module

    #### [BEGIN] Boilerplate ####
    def do_collect(self) -> None:
        self.enabled_collect = True

    def do_not_collect(self) -> None:
        self.enabled_collect = False

    def collect(self, x: torch.Tensor) -> None:
        assert isinstance(x, torch.Tensor) or (
            isinstance(x, list) and all(isinstance(y, torch.Tensor) for y in x)
        )
        collections = [x] if isinstance(x, torch.Tensor) else x
        if self.enabled_collect:
            if self.preproc_fn_or_nn_module is not None:
                collections = [self.preproc_fn_or_nn_module(c) for c in collections]
            assert isinstance(collections, list)
            assert all(isinstance(c, torch.Tensor) for c in collections)
            self.activations.extend(collections)

    def get(self) -> List[torch.Tensor]:
        return self.activations

    def clear(self) -> None:
        self.activations = []

    #### [END] Boilerplate ####


class CollectionWrapper(nn.Module):
    """
    Utility to store inputs and outputs of a module.
    """

    def __init__(
        self,
        module: nn.Module,
        collect_inputs: bool,
        collect_outputs: bool,
        clone: bool = False,
        detach: bool = True,
        cpu: bool = True,  # basically has to clone anyways
        inputs_preproc_fn_or_nn_module: Optional[
            Callable[[torch.Tensor], torch.Tensor] | nn.Module
        ] = None,
        outputs_preproc_fn_or_nn_module: Optional[
            Callable[[torch.Tensor], torch.Tensor] | nn.Module
        ] = None,
    ):
        super().__init__()
        # Wrapped module
        self.module = module

        # Wrapping storage
        self.collect_inputs = collect_inputs
        self.collect_outputs = collect_outputs

        # Configuration
        self.clone = clone
        self.detach = detach
        self.cpu = cpu

        self.input_collector = ActivationsCollector(
            collect=collect_inputs,
            preproc_fn_or_nn_module=inputs_preproc_fn_or_nn_module,
        )
        self.output_collector = ActivationsCollector(
            collect=collect_outputs,
            preproc_fn_or_nn_module=outputs_preproc_fn_or_nn_module,
        )

    def store(
        self,
        z: torch.Tensor,
        collector: ActivationsCollector,
    ) -> None:
        if self.detach:
            z = z.detach()
        if self.clone:
            z = z.clone()
        if self.cpu:
            z = z.cpu()
        collector.collect(z)

    def maybe_store(self, x: torch.Tensor, is_input: bool) -> None:
        if is_input:
            self.store(x, self.input_collector)
        else:
            self.store(x, self.output_collector)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.maybe_store(x, True)  # is_input = True
        y = self.module(x)
        self.maybe_store(y, False)  # is_input = False
        return y
