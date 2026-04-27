"""Generic cache-or-compute for safetensors artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from safetensors.torch import load_file, save_file


def cache_path(cache_dir: Path, model_id: str, subset: str, filename: str) -> Path:
    """Build a cache path: {cache_dir}/{model_slug}/{subset}/{filename}."""
    return cache_dir / model_id.replace("/", "--") / subset / filename


def load_or_compute_safetensors(
    path: Path,
    compute_fn: Callable[[], dict[str, torch.Tensor]],
    no_cache: bool = False,
    label: str = "artifact",
) -> dict[str, torch.Tensor]:
    """Load a safetensors file from disk, or compute and save it.

    Args:
        path: Where to read/write the cached artifact.
        compute_fn: Zero-arg callable that produces the tensor dict.
        no_cache: If True, always recompute (skip reading and writing).
        label: Human-readable name for log messages.

    Returns:
        Dict of tensor name -> tensor.
    """
    if path.exists() and not no_cache:
        print(f"[cache] Loading cached {label} from {path}")
        return load_file(str(path))
    print(f"[cache] Computing {label}...")
    result = compute_fn()
    if not no_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(result, str(path))
        print(f"[cache] Saved {label} to {path}")
    return result
