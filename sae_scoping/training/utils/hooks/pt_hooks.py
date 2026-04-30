from __future__ import annotations


import torch
import torch.nn as nn
from typing import Callable, Dict, Optional
from torch.utils.hooks import RemovableHandle
from contextlib import contextmanager

"""
Simple library to give context managers, useful methods (general purpose) for doing
hooking into pytorch models (this lets you (1) modify activations, (2) read activations
for storage, (3) etc...).
"""


class NamedForwardHooks:
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: Dict[str, RemovableHandle] = {}

    def add_hook(self, name: str, hook_fn: Callable, pre: bool = False):
        # Tries to add it to a MODULE: should work ok?
        named_modules = dict(self.model.named_modules())
        if name not in named_modules:
            raise ValueError(f"No module named '{name}' found in the model: {list(n for n, _ in self.model.named_modules())}.")

        module = named_modules[name]
        handle = (
            (module.register_forward_hook(lambda mod, inp, out: hook_fn(self, name, mod, inp, out)))
            if not pre
            else (
                module.register_forward_pre_hook(
                    # NOTE: we pass "None" to signify that this was meant to be a pre-hook
                    lambda mod, inp: hook_fn(self, name, mod, inp, None)
                )
            )
        )
        self.hooks[name] = handle

    def remove_hooks(self):
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()


@contextmanager
def named_forward_hooks(
    model: nn.Module,
    hook_dict: Dict[str, Callable | tuple[Callable, bool]],
):
    hooks = NamedForwardHooks(model)

    for name, hook_fn_obj in hook_dict.items():
        hook_fn = hook_fn_obj if isinstance(hook_fn_obj, Callable) else hook_fn_obj[0]
        pre = hook_fn_obj[1] if isinstance(hook_fn_obj, tuple) else False
        hooks.add_hook(name, hook_fn, pre=pre)

    try:
        yield hooks
    finally:
        hooks.remove_hooks()


def filter_hook_fn(
    # The filtering module (function) provided by the user
    filter_fn: Callable[[torch.Tensor, ...], torch.Tensor] | nn.Module,
    # Stuff from the code above
    hooks: NamedForwardHooks,
    name: str,
    mod: nn.Module,
    inp: Optional[tuple[torch.Tensor, ...] | torch.Tensor],
    out: Optional[tuple[torch.Tensor, ...] | torch.Tensor],
) -> None:
    """
    This function lets you perform activation engineering by using an
    nn.Module with a forward method or any callable. It actually doubles for
    collecting (and storing) the activations since your callable could do anything
    (for example storing the activations in a database, etc...).

    Proper usage: `with named_forward_hooks`, `Ft.partial(filter_hook_fn, filter_fn)`
    """
    # 1. Get the value
    # Support both forward and forward_pre hooks
    in_val = inp if out is None else out
    # 2. Get the tensor
    in_pt = in_val[0] if isinstance(in_val, tuple) else in_val
    assert isinstance(in_pt, torch.Tensor), f"Expected a tensor, got {type(in_pt)}"
    # 3. Apply the filter
    out_pt = filter_fn(in_pt)
    # 4. Re-format the output value
    out_val = tuple([out_pt] + list(in_val[1:])) if isinstance(in_val, tuple) else out_pt
    return out_val


