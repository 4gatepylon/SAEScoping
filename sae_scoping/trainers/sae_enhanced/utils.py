from __future__ import annotations
from beartype import beartype
from beartype.typing import Any, Callable
import json
import torch
from contextlib import contextmanager
from transformers import PreTrainedModel
from transformers import Gemma2ForCausalLM, LlamaForCausalLM
import re


@beartype
def is_int(x: int | float) -> bool:
    return float(int(x)) == float(x)


# sanity check lol
assert is_int(1)
assert is_int(1.0)
assert not is_int(1.1)
assert is_int(0)
assert is_int(-1)
assert not is_int(0.1)


@beartype
def freeze_parameters_before_layer(model: PreTrainedModel, sae_layer: int) -> list[str]:
    """Primarily for Gemma (but also possible for Llama2) models freeze layers to enable training only AFTER SAE."""
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
            assert match is not None, f"Parameter name {n} doesn't match expected pattern"
            layer_num = int(match.group(1))
            if layer_num <= sae_layer:
                p.requires_grad = False
                if p.grad is not None:
                    p.grad = None
                parameters_to_freeze.append(n)
    return parameters_to_freeze


@contextmanager
def frozen_parameters_training(
    model: PreTrainedModel,
    hookpoint: str | None,  # Hookpoint to freeze UP till
    strict_change_check: bool = True,  # True = all trainable must change, False = subset OK
    n_store: int = 32,  # This number should be large enough to reduce the likelihood of no change, but small enough to be performant
):
    """Context manager for parameter freezing during training."""
    # 1. Freeze parameters before hookpoint layer
    p2f = set()
    if hookpoint is not None:
        hp_patt = r"^model\.layers\.(\d+)$"
        if not re.match(hp_patt, hookpoint):
            raise ValueError(f"Hookpoint {hookpoint} is not a valid layer hookpoint")
        sae_layer = int(re.match(hp_patt, hookpoint).group(1))
        p2f = set(freeze_parameters_before_layer(model, sae_layer))

    # 2. Record initial state
    trainable_params_before = sorted([n for n, p in model.named_parameters() if p.requires_grad])
    frozen_params_before = sorted([n for n, p in model.named_parameters() if not p.requires_grad])

    # 3. Printouts
    print("hookpoint: ", hookpoint)
    print(f"Trainable params @ hookpoint={hookpoint}: {json.dumps(trainable_params_before, indent=4)}")
    print(f"Frozen params @ hookpoint={hookpoint}: {json.dumps(frozen_params_before, indent=4)}")

    # 4. Pre-training assertions
    assert set(frozen_params_before) == p2f
    assert (set(trainable_params_before) & p2f) == set()

    # 5. Store initial parameter values for change detection
    p2s_initial = {n: p.data.detach().view(-1)[:n_store].cpu() for n, p in model.named_parameters()}

    try:
        yield  # Training happens here
    finally:
        # 6. Post-training: verify frozen/trainable sets unchanged
        trainable_params_end = sorted([n for n, p in model.named_parameters() if p.requires_grad])
        frozen_params_end = sorted([n for n, p in model.named_parameters() if not p.requires_grad])
        assert trainable_params_before == trainable_params_end
        assert frozen_params_before == frozen_params_end

        # 7. Detect which parameters actually changed
        parameters_that_changed = []
        for n, p in model.named_parameters():
            slc = p.data.detach().view(-1)[:n_store].cpu()
            if not torch.allclose(slc, p2s_initial[n]):
                parameters_that_changed.append(n)

        # 8. Verify changes are valid
        if strict_change_check:
            # (meant for) SFT: all trainable params should change
            assert set(parameters_that_changed) == set(trainable_params_end), (
                f"Parameters that changed: {json.dumps(list(parameters_that_changed), indent=4)}\n\n" f"Should be: {json.dumps(list(trainable_params_end), indent=4)}"
            )
        else:
            # (meant for) GRPO: changed params should be subset of trainable
            assert set(parameters_that_changed).issubset(set(trainable_params_end)), (
                f"Parameters that changed: {json.dumps(list(parameters_that_changed), indent=4)}\n\n" f"Should be subset of: {json.dumps(list(trainable_params_end), indent=4)}"
            )

        # 9. Frozen params must never change
        assert len(set(parameters_that_changed) & set(frozen_params_end)) == 0, (
            f"Parameters that changed and are frozen: " f"{json.dumps(list(set(parameters_that_changed) & set(frozen_params_end)), indent=4)}\n\n" f"Should be empty"
        )
        assert len(set(parameters_that_changed) & p2f) == 0, f"Parameters that changed and are frozen: " f"{json.dumps(list(set(parameters_that_changed) & p2f), indent=4)}\n\n" f"Should be empty"


@beartype
def str_dict_diff(
    found: dict[str, Any],
    expected: dict[str, Any],
    jsonifiable_fn: Callable[[Any], str] = str,
) -> str:
    """Utility funciton that can help debug issues when some parameters change and some don't in unexpected ways."""
    assert all(isinstance(k, str) for k in found.keys())
    assert all(isinstance(k, str) for k in expected.keys())
    found2str = {k: jsonifiable_fn(v) for k, v in found.items()}
    expected2str = {k: jsonifiable_fn(v) for k, v in expected.items()}
    found_minus_expected = {k: v for k, v in found.items() if k not in expected}
    expected_minus_found = {k: v for k, v in expected.items() if k not in found}
    not_equal = {k: f"Found: {jsonifiable_fn(v)}. Expected: {jsonifiable_fn(expected[k])}" for k, v in found.items() if v != expected[k]}
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
