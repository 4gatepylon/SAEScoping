"""Taylor saliency map: |saliency * weight| derived from any source map."""

import click
import torch
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

from sae_scoping.training.saliency.utils import _DEFAULT_MODEL_ID, save_saliency_map

# Stems of source maps accepted as input.  Validated early so mismatches are
# caught before loading a large model.
_TAYLOR_SOURCE_STEMS: frozenset[str] = frozenset({"ema_grads", "ema_grads_abs", "random"})


def make_taylor_map(
    saliency_tensors: dict[str, torch.Tensor],
    model: AutoModelForCausalLM,
) -> dict[str, torch.Tensor]:
    """Compute a Taylor saliency map: |saliency * weight| for every parameter.

    This is the first-order Taylor approximation of the change in loss when a
    weight is zeroed (Δloss ≈ |grad * weight|).  Works with any source map
    (gradient EMA, absolute EMA, or random).

    Pure computation — does not write to disk.  Call save_saliency_map
    separately when caching is desired.
    """
    result: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name not in saliency_tensors:
            continue
        s = saliency_tensors[name].float().to(param.device)
        result[name] = (s * param.data.float()).abs()
    return result


def taylor_output_path(source_path: Path) -> Path:
    """Derive the canonical output path for a taylor map from its source path.

    Example: biology/ema_grads.safetensors → biology/taylor_ema_grads.safetensors
    """
    return source_path.parent / f"taylor_{source_path.stem}.safetensors"


def validate_taylor_source_path(source_path: Path) -> None:
    """Raise ValueError if source_path.stem is not a recognised saliency map name."""
    stem = source_path.stem
    if stem not in _TAYLOR_SOURCE_STEMS:
        raise ValueError(
            f"Unrecognised source map filename '{source_path.name}'. "
            f"Expected a file whose stem is one of: {sorted(_TAYLOR_SOURCE_STEMS)}. "
            "(e.g. biology/ema_grads.safetensors)"
        )


@click.command("taylor")
@click.option(
    "--input-path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help=(
        "Source saliency .safetensors file to derive the Taylor map from. "
        f"Filename stem must be one of: {sorted(_TAYLOR_SOURCE_STEMS)}."
    ),
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Destination .safetensors file. "
        "Defaults to taylor_{source_stem}.safetensors in the same directory."
    ),
)
@click.option("--model-id", type=str, default=_DEFAULT_MODEL_ID, show_default=True)
@click.option("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
def run_taylor(
    input_path: Path,
    output_path: Path | None,
    model_id: str,
    device: str,
) -> None:
    """Derive a Taylor saliency map (|saliency * weight|) from an existing map."""
    validate_taylor_source_path(input_path)
    resolved_output = output_path or taylor_output_path(input_path)

    if resolved_output.exists():
        print(f"[taylor] ⚠️  Overwriting existing output: {resolved_output}")

    saliency_tensors = load_file(str(input_path))
    print(f"Loaded source map: {len(saliency_tensors)} tensors from {input_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device,
    )
    for p in model.parameters():
        p.requires_grad = False

    taylor = make_taylor_map(saliency_tensors, model)
    save_saliency_map(taylor, str(resolved_output))
