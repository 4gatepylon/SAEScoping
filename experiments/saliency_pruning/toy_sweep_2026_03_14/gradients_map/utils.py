"""Shared constants, helpers, and assertions for the gradients_map package."""

import datetime
import os
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_ID = "google/gemma-2-9b-it"
_DEFAULT_DATASET = "4gate/StemQAMixture"
_DEFAULT_SUBSET = "biology"
_DEFAULT_DATASET_SIZE = 16_384
_DEFAULT_BETA = 0.95
_DEFAULT_BATCH_SIZE = 2
_DEFAULT_MAX_SEQ = 1024
_DEFAULT_MODE = "gradient_ema"

_MODE_TO_DEFAULT_OUT_PATH = {
    "gradient_ema": "./biology/ema_grads.safetensors",
    "random": "./biology/random.safetensors",
}

# Path to the Gemma2 Jinja2 chat template (optional — used if present).
_CHAT_TEMPLATE_PATH = Path(__file__).parent.parent / "prompts" / "gemma2_chat_template_system_prompt.j2"

# Canonical variant specs used by the batch command.
# Maps variant name → (mode, abs_grad, default_output_path).
_VARIANT_SPECS: dict[str, tuple[str, bool, str]] = {
    "gradient_ema":     ("gradient_ema", False, "./biology/ema_grads.safetensors"),
    "gradient_ema_abs": ("gradient_ema", True,  "./biology/ema_grads_abs.safetensors"),
    "random":           ("random",       False, "./biology/random.safetensors"),
}
_ALL_VARIANTS: tuple[str, ...] = tuple(_VARIANT_SPECS.keys())


# ---------------------------------------------------------------------------
# Output-path helpers
# ---------------------------------------------------------------------------


def _mode_to_default_output_path(mode: str, abs_grad: bool) -> str:
    """Return the default safetensors output path for a given (mode, abs_grad) combination.

    Raises ValueError if abs_grad=True is requested for a mode that does not
    support it (only 'gradient_ema' produces an abs variant).
    """
    if abs_grad and mode != "gradient_ema":
        raise ValueError(
            f"--abs-grad is only valid for mode 'gradient_ema', got '{mode}'."
        )
    if mode == "gradient_ema" and abs_grad:
        return "./biology/ema_grads_abs.safetensors"
    return _MODE_TO_DEFAULT_OUT_PATH[mode]


def save_saliency_map(saliency: dict[str, torch.Tensor], path: str) -> None:
    """Save a saliency map (any mode) to a safetensors file."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(saliency, str(out))
    print(f"Saved saliency map -> {out}")


# ---------------------------------------------------------------------------
# WandB config
# ---------------------------------------------------------------------------


def _resolve_wandb_config(
    wandb_project: str | None,
    wandb_run_name: str | None,
    mode: str,
    abs_grad: bool,
    dataset_subset: str,
) -> tuple[str | None, str]:
    """Return (resolved_run_name, report_to) and set WANDB_PROJECT if a project is given.

    When wandb_project is None logging is disabled and resolved_run_name is None.
    """
    if not wandb_project:
        return None, "none"
    abs_tag = "_abs" if abs_grad else ""
    today = datetime.date.today().isoformat()
    resolved_run_name = wandb_run_name or f"{today}_{mode}{abs_tag}_{dataset_subset}"
    os.environ["WANDB_PROJECT"] = wandb_project
    return resolved_run_name, "wandb"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def assert_all_params_require_grad(
    model: AutoModelForCausalLM,
    allow_frozen: bool = False,
) -> None:
    """Raise if any model parameter has requires_grad=False.

    The saliency computation pipeline assumes every parameter is trainable so
    that gradient hooks fire on all of them and make_random_map produces a
    key-complete baseline.  A frozen parameter would be silently skipped,
    producing an incomplete saliency map that misrepresents which weights were
    scored.

    Pass allow_frozen=True to suppress this check (e.g. for PEFT adapters
    where base-model layers are intentionally frozen).
    """
    if allow_frozen:
        return
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    if frozen:
        raise AssertionError(
            f"{len(frozen)} parameter(s) do not require grad — saliency map will "
            f"be incomplete.  Frozen params: {frozen}\n"
            "Pass --allow-frozen-params to suppress this check."
        )




# ---------------------------------------------------------------------------
# Diagnostics / post-training assertions
# ---------------------------------------------------------------------------


def _sample_param_indices(
    model: AutoModelForCausalLM, n_indices: int
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    param_names2random_indices = {
        pname: torch.randint(0, param.numel(), (n_indices,), device=param.device)
        for pname, param in model.named_parameters()
    }
    param_name2initial_values = {
        pname: param.data.flatten()[param_names2random_indices[pname]].cpu()
        for pname, param in model.named_parameters()
    }
    return param_names2random_indices, param_name2initial_values


def _report_hook_diagnostics(model: AutoModelForCausalLM, global_step: int) -> None:
    never_fired = [
        n for n, p in model.named_parameters()
        if p.requires_grad and model._hook_fires.get(n, 0) == 0
    ]
    fired_but_no_grad = [
        n for n, p in model.named_parameters()
        if p.requires_grad and model._hook_fires.get(n, 0) > 0 and p.grad is None
    ]
    total_fires = sum(model._hook_fires.values())
    n_params = len(list(model.parameters()))
    print(f"Hook never fired for {len(never_fired)} params: {never_fired[:5]}")
    print(f"Hook fired but grad is None for {len(fired_but_no_grad)} params: {fired_but_no_grad[:5]}")
    print(f"Total hook fires across all params: {total_fires} (expected ~{n_params} x {global_step} steps)")


def _assert_weights_unchanged(
    model: AutoModelForCausalLM,
    param_names2random_indices: dict[str, torch.Tensor],
    param_name2initial_random_index_values: dict[str, torch.Tensor],
) -> None:
    for pname, param in model.named_parameters():
        current = param.data.flatten()[param_names2random_indices[pname]].cpu()
        initial = param_name2initial_random_index_values[pname]
        assert torch.allclose(current, initial), (
            f"Parameter {pname} has changed from {initial} to {current}"
        )


def _assert_ema_grads_populated(model: AutoModelForCausalLM, trainer) -> None:
    assert all(len(v) == 0 for v in trainer.optimizer.state.values()), \
        "Optimizer allocated state despite no-op step"
    missing = [
        pname for pname, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert len(missing) == 0, \
        f"Some trainable params have no EMA grad. They are:\n{chr(10).join(missing)}"
