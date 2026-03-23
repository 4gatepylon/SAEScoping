from __future__ import annotations
from beartype import beartype
from beartype.typing import Any, Callable, Tuple, Optional
import torch
import torch.nn as nn
from sae_scoping.training.sae_enhanced.hooks.pt_hooks import filter_hook_fn, NamedForwardHooks


class Context:
    def __init__(self, value: Any | None = None):
        self.value = value

    def clear_value(self) -> None:
        self.value = None

    def set_value(self, value: Any) -> None:
        self.value = value


class StatefulGenerationApplier:
    """
    A `StatefulGenerationApplier` is a wrapper around a stateful function that is meant
    to decide when to apply some kind of autoencoder (which here is called the `applier`).

    It can be used to apply steering or some kind of filtering, etc... differently
    depending on what token you are at. It is primarily designed to make it easy to use
    hooks at the same time as doing generation with caching.
    """

    def __init__(self, applier: Callable) -> None:
        self.applier = applier

    def increment_state(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError("Not implemented")  # You should extend this!

    def should_apply(self, tensor: torch.Tensor) -> Tuple[bool, Any]:
        raise NotImplementedError("Not implemented")  # You should extend this!

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        sapply, kwargs = self.should_apply(tensor)
        if sapply:
            ret = self.applier(tensor, **kwargs)
        else:
            ret = tensor
        self.increment_state(tensor)
        return ret


class StatefulIndexGenerationApplier(StatefulGenerationApplier):
    def __init__(
        self,
        # (tensor, **kwargs) -> tensor
        applier: Callable[[torch.Tensor, ...], torch.Tensor],
        # (index, start_index, is_prompt) -> (should_apply, kwargs)
        index_decider: Callable[[int, int, bool], Tuple[bool, Any]],
    ) -> None:
        super().__init__(applier)
        self.is_prompt: bool = True  # prompt -> generation
        self.is_generation: bool = False
        self.start_index = None  # None => Not known yet
        self.index = None  # None => Not known yet
        self.index_decider = index_decider

    def should_apply(self, tensor: torch.Tensor) -> Tuple[bool, Any]:
        assert tensor.ndim == 3, f"tensor.shape: {tensor.shape}"
        if self.is_prompt:
            assert self.start_index is not None
            assert self.index is not None
            raise ValueError("Prompt application should be seperate!")
        else:
            assert self.index is not None
            assert self.start_index is not None
            return self.index_decider(self.index, self.start_index, False)

    def increment_state(self, tensor: torch.Tensor) -> None:
        self.index += 1

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Application has to be modified to be able to initialize on prompts seperately.
        """
        assert self.is_prompt == (not self.is_generation), f"is_prompt: {self.is_prompt}, is_generation: {self.is_generation}"  # fmt: skip
        assert self.is_prompt == (self.index is None) == (self.start_index is None), f"is_prompt: {self.is_prompt}, index: {self.index}, start_index: {self.start_index}"  # fmt: skip
        assert tensor.ndim == 3, f"tensor.shape: {tensor.shape}"
        B, T, D = tensor.shape
        ret = tensor
        if self.is_prompt:
            # NOTE: start_index starts at 1... because you must ALWAYS have something
            # to predict FROM.
            self.start_index = T
            self.index = T
            self.is_prompt = False
            self.is_generation = True
            # 1. Read from prompt, deciding where to apply
            token_indices_to_apply = torch.tensor(
                [i for i in range(T) if self.index_decider(i, 1, True)],
                device=tensor.device,
                dtype=torch.long,
            )
            if token_indices_to_apply.numel() != 0:
                assert token_indices_to_apply.ndim == 1, f"token_indices_to_apply.shape: {token_indices_to_apply.shape}"  # fmt: skip
                assert token_indices_to_apply.shape[0] <= T, f"token_indices_to_apply.shape: {token_indices_to_apply.shape}, T: {T}"  # fmt: skip
                tensor_slice = tensor[:, token_indices_to_apply, :]
                Bs, Ts, Ds = tensor_slice.shape
                assert Bs == B, f"Bs: {Bs}, B: {B}"
                assert Ts <= T, f"Ts: {Ts}, T: {T}"
                assert Ds == D, f"Ds: {Ds}, D: {D}"
                applied = self.applier(tensor_slice)
                tensor[:, token_indices_to_apply, :] = applied
                ret = tensor
        elif self.is_generation and self.should_apply(tensor):
            ret = self.applier(tensor)
        self.increment_state(tensor)
        return ret


def _prompt_only_decider(
    index: int, start_index: int, is_prompt: bool
) -> Tuple[bool, Any]:
    return is_prompt, {}


def _after_prompt_only_decider(
    index: int, start_index: int, is_prompt: bool
) -> Tuple[bool, Any]:
    return not is_prompt, {}


def _even_decider(index: int, start_index: int, is_prompt: bool) -> Tuple[bool, Any]:
    return index % 2 == 0, {}


def _odd_decider(index: int, start_index: int, is_prompt: bool) -> Tuple[bool, Any]:
    return index % 2 == 1, {}


class PromptOnlyGenerationApplier(StatefulIndexGenerationApplier):
    def __init__(
        self,
        applier: Callable[[torch.Tensor, ...], torch.Tensor],
    ) -> None:
        super().__init__(applier, _prompt_only_decider)


class AfterPromptOnlyGenerationApplier(StatefulIndexGenerationApplier):
    def __init__(
        self,
        applier: Callable[[torch.Tensor, ...], torch.Tensor],
    ) -> None:
        super().__init__(applier, _after_prompt_only_decider)


class EvenGenerationApplier(StatefulIndexGenerationApplier):
    def __init__(
        self,
        applier: Callable[[torch.Tensor, ...], torch.Tensor],
    ) -> None:
        super().__init__(applier, _even_decider)


class OddGenerationApplier(StatefulIndexGenerationApplier):
    def __init__(
        self,
        applier: Callable[[torch.Tensor, ...], torch.Tensor],
    ) -> None:
        super().__init__(applier, _odd_decider)


# Dummy for identity fn
class NoneGenerationApplier(StatefulGenerationApplier):
    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


# Dummy for "always apply"
class AllGenerationApplier(StatefulGenerationApplier):
    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.applier(tensor)


def stateful_filter_fn(
    stateful_applier: StatefulIndexGenerationApplier,
    hooks: NamedForwardHooks,
    name: str,
    mod: nn.Module,
    inp: Optional[tuple[torch.Tensor, ...] | torch.Tensor],
    out: Optional[tuple[torch.Tensor, ...] | torch.Tensor],
) -> torch.Tensor:
    """
    This is a slot-in for the `filter_hook_fn` function. You would just define your
    stateful generation applier and thne add it as an argument to the filter fn.

    Commonly you will do something like:
    ```
    with named_forward_hooks(model, {
        "my_hookpoint": functools.partial(stateful_filter_fn,
            StatefulApplierClass(SAEWrapper(sae)), # or whatever nn module you want
        ),
    })
    ```
    NOTE that the tensor SHAPE will be modified by the SAEWrapper in the above example.
    This means if you replace it with another wrapper you will want to modify the shape
    yourself if necessary (for B T D -> (B T) D -> B T D).
    """
    return filter_hook_fn(
        # NOTE that your application is INSIDE The applier
        filter_fn=lambda x: stateful_applier.apply(x),
        hooks=hooks,
        name=name,
        mod=mod,
        inp=inp,
        out=out,
    )


class StatefulGenerationApplier:
    """
    A `StatefulGenerationApplier` is a wrapper around a stateful function that is meant
    to decide when to apply some kind of autoencoder (which here is called the `applier`).

    It can be used to apply steering or some kind of filtering, etc... differently
    depending on what token you are at. It is primarily designed to make it easy to use
    hooks at the same time as doing generation with caching.
    """

    def __init__(self, applier: Callable) -> None:
        self.applier = applier

    def increment_state(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError("Not implemented")  # You should extend this!

    def should_apply(self, tensor: torch.Tensor) -> Tuple[bool, Any]:
        raise NotImplementedError("Not implemented")  # You should extend this!

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        sapply, kwargs = self.should_apply(tensor)
        if sapply:
            ret = self.applier(tensor, **kwargs)
        else:
            ret = tensor
        self.increment_state(tensor)
        return ret


class StatefulIndexGenerationApplier(StatefulGenerationApplier):
    def __init__(
        self,
        # (tensor, **kwargs) -> tensor
        applier: Callable[[torch.Tensor, ...], torch.Tensor],
        # (index, start_index, is_prompt) -> (should_apply, kwargs)
        index_decider: Callable[[int, int, bool], Tuple[bool, Any]],
    ) -> None:
        super().__init__(applier)
        self.is_prompt: bool = True  # prompt -> generation
        self.is_generation: bool = False
        self.start_index = None  # None => Not known yet
        self.index = None  # None => Not known yet
        self.index_decider = index_decider

    def should_apply(self, tensor: torch.Tensor) -> Tuple[bool, Any]:
        assert tensor.ndim == 3, f"tensor.shape: {tensor.shape}"
        if self.is_prompt:
            assert self.start_index is not None
            assert self.index is not None
            raise ValueError("Prompt application should be seperate!")
        else:
            assert self.index is not None
            assert self.start_index is not None
            return self.index_decider(self.index, self.start_index, False)

    def increment_state(self, tensor: torch.Tensor) -> None:
        self.index += 1

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Application has to be modified to be able to initialize on prompts seperately.
        """
        assert self.is_prompt == (not self.is_generation), f"is_prompt: {self.is_prompt}, is_generation: {self.is_generation}"  # fmt: skip
        assert self.is_prompt == (self.index is None) == (self.start_index is None), f"is_prompt: {self.is_prompt}, index: {self.index}, start_index: {self.start_index}"  # fmt: skip
        assert tensor.ndim == 3, f"tensor.shape: {tensor.shape}"
        B, T, D = tensor.shape
        ret = tensor
        if self.is_prompt:
            # NOTE: start_index starts at 1... because you must ALWAYS have something
            # to predict FROM.
            self.start_index = T
            self.index = T
            self.is_prompt = False
            self.is_generation = True
            # 1. Read from prompt, deciding where to apply
            token_indices_to_apply = torch.tensor(
                [i for i in range(T) if self.index_decider(i, 1, True)],
                device=tensor.device,
                dtype=torch.long,
            )
            if token_indices_to_apply.numel() != 0:
                assert token_indices_to_apply.ndim == 1, f"token_indices_to_apply.shape: {token_indices_to_apply.shape}"  # fmt: skip
                assert token_indices_to_apply.shape[0] <= T, f"token_indices_to_apply.shape: {token_indices_to_apply.shape}, T: {T}"  # fmt: skip
                tensor_slice = tensor[:, token_indices_to_apply, :]
                Bs, Ts, Ds = tensor_slice.shape
                assert Bs == B, f"Bs: {Bs}, B: {B}"
                assert Ts <= T, f"Ts: {Ts}, T: {T}"
                assert Ds == D, f"Ds: {Ds}, D: {D}"
                applied = self.applier(tensor_slice)
                tensor[:, token_indices_to_apply, :] = applied
                ret = tensor
        elif self.is_generation and self.should_apply(tensor):
            ret = self.applier(tensor)
        self.increment_state(tensor)
        return ret


def _prompt_only_decider(
    index: int, start_index: int, is_prompt: bool
) -> Tuple[bool, Any]:
    return is_prompt, {}


def _after_prompt_only_decider(
    index: int, start_index: int, is_prompt: bool
) -> Tuple[bool, Any]:
    return not is_prompt, {}


def _even_decider(index: int, start_index: int, is_prompt: bool) -> Tuple[bool, Any]:
    return index % 2 == 0, {}


def _odd_decider(index: int, start_index: int, is_prompt: bool) -> Tuple[bool, Any]:
    return index % 2 == 1, {}


class PromptOnlyGenerationApplier(StatefulIndexGenerationApplier):
    def __init__(
        self,
        applier: Callable[[torch.Tensor, ...], torch.Tensor],
    ) -> None:
        super().__init__(applier, _prompt_only_decider)


class AfterPromptOnlyGenerationApplier(StatefulIndexGenerationApplier):
    def __init__(
        self,
        applier: Callable[[torch.Tensor, ...], torch.Tensor],
    ) -> None:
        super().__init__(applier, _after_prompt_only_decider)


class EvenGenerationApplier(StatefulIndexGenerationApplier):
    def __init__(
        self,
        applier: Callable[[torch.Tensor, ...], torch.Tensor],
    ) -> None:
        super().__init__(applier, _even_decider)


class OddGenerationApplier(StatefulIndexGenerationApplier):
    def __init__(
        self,
        applier: Callable[[torch.Tensor, ...], torch.Tensor],
    ) -> None:
        super().__init__(applier, _odd_decider)


# Dummy for identity fn
class NoneGenerationApplier(StatefulGenerationApplier):
    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


# Dummy for "always apply"
class AllGenerationApplier(StatefulGenerationApplier):
    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.applier(tensor)


def stateful_filter_fn(
    stateful_applier: StatefulIndexGenerationApplier,
    hooks: NamedForwardHooks,
    name: str,
    mod: nn.Module,
    inp: Optional[tuple[torch.Tensor, ...] | torch.Tensor],
    out: Optional[tuple[torch.Tensor, ...] | torch.Tensor],
) -> torch.Tensor:
    """
    This is a slot-in for the `filter_hook_fn` function. You would just define your
    stateful generation applier and thne add it as an argument to the filter fn.

    Commonly you will do something like:
    ```
    with named_forward_hooks(model, {
        "my_hookpoint": functools.partial(stateful_filter_fn,
            StatefulApplierClass(SAEWrapper(sae)), # or whatever nn module you want
        ),
    })
    ```
    NOTE that the tensor SHAPE will be modified by the SAEWrapper in the above example.
    This means if you replace it with another wrapper you will want to modify the shape
    yourself if necessary (for B T D -> (B T) D -> B T D).
    """
    return filter_hook_fn(
        # NOTE that your application is INSIDE The applier
        filter_fn=lambda x: stateful_applier.apply(x),
        hooks=hooks,
        name=name,
        mod=mod,
        inp=inp,
        out=out,
    )
