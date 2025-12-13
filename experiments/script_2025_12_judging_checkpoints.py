from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import click
import torch
from beartype import beartype
from beartype.typing import Any
from sae_lens import SAE
from safetensors.torch import load_file
from transformers import AutoTokenizer, Gemma2ForCausalLM, PreTrainedTokenizerBase

from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae


"""
Operate on a path that looks like this:

.
├── apps
│   └── layer_31_width_16k_canonical_h0.0001_85cac49528
│       ├── checkpoint-1000
│       ...
│       └── checkpoint-N
├── <dataset name>
│   ├── layer_<layer index>_width_16k_canonical_h<h value; h is a hyperparameter value>
│   │   ├── checkpoint-1000
│   │   ├── checkpoint-2000
│   │   ├── ...
│   │   └── checkpoint-N
│   ├── ...
│   └── vanilla
│       ├── checkpoint-1000
│       ├── ...
│       └── checkpoint-N
...

One-off script to do judging. It is hardcoded for our specific judges, etc... Supports
caching so you can recover from failures.
"""

GEMMA2_9B_SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"


@dataclass
class CheckpointInfo:
    """Information about a single checkpoint."""

    checkpoint_path: Path
    dataset_name: str
    sae_id: str | None  # None for vanilla
    layer: int | None  # None for vanilla
    threshold: float | None  # None for vanilla
    hash_suffix: str | None  # None for vanilla
    step: int

    @property
    def is_vanilla(self) -> bool:
        return self.sae_id is None

    @beartype
    def to_dict(self, json_serializable: bool = False) -> dict[str, Any]:
        return {
            "checkpoint_path": (
                self.checkpoint_path.as_posix()
                if json_serializable
                else self.checkpoint_path
            ),
            "dataset_name": self.dataset_name,
            "sae_id": self.sae_id,
            "layer": self.layer,
            "threshold": self.threshold,
            "hash_suffix": self.hash_suffix,
        }

    @staticmethod
    @beartype
    def from_dict(data: dict[str, Any]) -> CheckpointInfo:
        return CheckpointInfo(
            checkpoint_path=Path(data["checkpoint_path"]),
            dataset_name=data["dataset_name"],
            sae_id=data["sae_id"],
            layer=data["layer"],
            threshold=data["threshold"],
            hash_suffix=data["hash_suffix"],
        )


@beartype
def parse_sae_folder_name(
    folder_name: str,
) -> tuple[str | None, int | None, float | None, str | None]:
    """
    Parse a folder name like 'layer_31_width_16k_canonical_h0.0001_85cac49528'
    into (sae_id, layer, threshold, hash_suffix).

    For 'vanilla', returns (None, None, None, None).
    """
    if folder_name == "vanilla":
        return None, None, None, None

    # Pattern: layer_N_width_Mk_canonical_hTHRESHOLD_HASH
    pattern = r"^layer_(\d+)_width_(\d+k)_canonical_h([\d.e-]+)_([a-f0-9]+)$"
    match = re.match(pattern, folder_name)
    if not match:
        raise ValueError(f"Could not parse SAE folder name: {folder_name}")

    layer = int(match.group(1))
    width = match.group(2)
    threshold = float(match.group(3))
    hash_suffix = match.group(4)

    sae_id = f"layer_{layer}/width_{width}/canonical"
    return sae_id, layer, threshold, hash_suffix


@beartype
def iter_checkpoints(root_path: Path) -> Iterator[CheckpointInfo]:
    """
    Iterate through all checkpoints in the folder structure.

    Yields CheckpointInfo for each checkpoint found.
    """
    if not root_path.exists():
        raise ValueError(f"Root path does not exist: {root_path}")

    # Iterate through dataset folders
    for dataset_dir in sorted(root_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_name = dataset_dir.name

        # Iterate through SAE folders (or vanilla)
        for sae_dir in sorted(dataset_dir.iterdir()):
            if not sae_dir.is_dir():
                continue

            try:
                sae_id, layer, threshold, hash_suffix = parse_sae_folder_name(
                    sae_dir.name
                )
            except ValueError as e:
                print(f"Skipping {sae_dir}: {e}")
                continue

            # Iterate through checkpoints
            for ckpt_dir in sorted(sae_dir.iterdir()):
                if not ckpt_dir.is_dir():
                    continue
                if not ckpt_dir.name.startswith("checkpoint-"):
                    continue

                # Extract step number
                step = int(ckpt_dir.name.split("-")[1])

                yield CheckpointInfo(
                    checkpoint_path=ckpt_dir,
                    dataset_name=dataset_name,
                    sae_id=sae_id,
                    layer=layer,
                    threshold=threshold,
                    hash_suffix=hash_suffix,
                    step=step,
                )


# XXX finish testing and implementing this shit
# XXX then we will want to run inference on each of the checkpoints on each of the datasets
# @beartype
# def load_checkpoint_with_sae(
#     checkpoint_info: CheckpointInfo,
#     device: torch.device | str = "cuda",
#     dist_cache_path: Path | None = None,
# ) -> tuple[Gemma2ForCausalLM, PreTrainedTokenizerBase, SAE | None, object | None]:
#     """
#     Load a model checkpoint and corresponding SAE (if not vanilla).

#     Args:
#         checkpoint_info: Information about the checkpoint
#         device: Device to load models onto
#         dist_cache_path: Path to distribution cache (needed for non-vanilla SAEs)
#             Expected structure: dist_cache_path/<sae_id with -- separators>/distribution.safetensors

#     Returns:
#         Tuple of (model, tokenizer, sae, pruned_sae)
#         For vanilla checkpoints, sae and pruned_sae are None.
#     """
#     device = torch.device(device) if isinstance(device, str) else device

#     # Load tokenizer
#     model_name = "google/gemma-2-9b-it"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     assert isinstance(tokenizer, PreTrainedTokenizerBase)

#     # Load model from checkpoint
#     model = Gemma2ForCausalLM.from_pretrained(
#         str(checkpoint_info.checkpoint_path),
#         torch_dtype=torch.bfloat16,
#         device_map="cpu",
#         attn_implementation="eager",
#     )
#     model = model.to(device)
#     model.gradient_checkpointing_disable()
#     if hasattr(model, "model"):
#         model.model.gradient_checkpointing = False

#     if checkpoint_info.is_vanilla:
#         return model, tokenizer, None, None

#     # Load SAE
#     sae = SAE.from_pretrained(
#         release=GEMMA2_9B_SAE_RELEASE,
#         sae_id=checkpoint_info.sae_id,
#         device=device,
#     )
#     sae = sae.to(device)

#     # Load distribution for pruning
#     if dist_cache_path is None:
#         raise ValueError(
#             "dist_cache_path is required for non-vanilla checkpoints to load the pruning mask"
#         )

#     # Convert sae_id to folder name format: layer_N/width_Mk/canonical -> layer_N--width_Mk--canonical
#     sae_folder_name = checkpoint_info.sae_id.replace("/", "--")
#     dist_file = dist_cache_path / sae_folder_name / "distribution.safetensors"

#     if not dist_file.exists():
#         raise ValueError(f"Distribution file not found: {dist_file}")

#     dist_data = load_file(str(dist_file))
#     distribution: torch.Tensor = dist_data["distribution"]
#     neuron_ranking = torch.argsort(distribution, descending=True)
#     n_kept = int((distribution >= checkpoint_info.threshold).sum().item())

#     print(f"Keeping {n_kept}/{len(distribution)} neurons (threshold={checkpoint_info.threshold})")

#     pruned_sae = get_pruned_sae(sae, neuron_ranking, K_or_p=n_kept, T=0.0)
#     pruned_sae = pruned_sae.to(device)

#     return model, tokenizer, sae, pruned_sae


# @click.command()
# @click.option(
#     "--root-path",
#     "-r",
#     type=click.Path(exists=True, path_type=Path),
#     required=True,
#     help="Root path containing dataset folders with checkpoints",
# )
# @click.option(
#     "--dist-cache-path",
#     "-d",
#     type=click.Path(exists=True, path_type=Path),
#     default=None,
#     help="Path to distribution cache (for loading pruned SAEs)",
# )
# @click.option(
#     "--dry-run",
#     is_flag=True,
#     help="Only list checkpoints, don't load them",
# )
# def main(
#     root_path: Path,
#     dist_cache_path: Path | None,
#     dry_run: bool,
# ) -> None:
#     """
#     Iterate through all checkpoints and load the corresponding SAE.

#     Example usage:
#         python script_2025_12_judging_checkpoints.py -r outputs_gemma9b -d dist_cache --dry-run
#         python script_2025_12_judging_checkpoints.py -r outputs_gemma9b -d dist_cache
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     for ckpt_info in iter_checkpoints(root_path):
#         print(f"\n{'='*80}")
#         print(f"Dataset: {ckpt_info.dataset_name}")
#         print(f"Checkpoint: {ckpt_info.checkpoint_path}")
#         print(f"Step: {ckpt_info.step}")
#         print(f"SAE ID: {ckpt_info.sae_id or 'vanilla'}")
#         if not ckpt_info.is_vanilla:
#             print(f"Layer: {ckpt_info.layer}")
#             print(f"Threshold: {ckpt_info.threshold}")
#             print(f"Hash: {ckpt_info.hash_suffix}")

#         if dry_run:
#             print("(dry run - skipping load)")
#             continue

#         try:
#             model, tokenizer, sae, pruned_sae = load_checkpoint_with_sae(
#                 ckpt_info,
#                 device=device,
#                 dist_cache_path=dist_cache_path,
#             )

#             print(f"✓ Loaded model: {type(model).__name__}")
#             if sae is not None:
#                 print(f"✓ Loaded SAE: {sae.cfg.d_sae} features")
#             if pruned_sae is not None:
#                 print(f"✓ Created pruned SAE wrapper")

#             # TODO: Add your evaluation/judgment logic here
#             # Example:
#             # results = evaluate_checkpoint(model, tokenizer, sae, pruned_sae)
#             # save_results(ckpt_info, results)

#             # Cleanup
#             del model, sae, pruned_sae
#             torch.cuda.empty_cache()

#         except Exception as e:
#             print(f"✗ Error loading checkpoint: {e}")


# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    # Test the parser
    path = Path(
        "/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b"
    )
    for checkpoint in iter_checkpoints(path):
        print(json.dumps(checkpoint.to_dict(json_serializable=True), indent=4))
