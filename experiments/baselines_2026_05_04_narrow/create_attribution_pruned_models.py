#!/usr/bin/env python3
"""
Create attribution-based pruned versions of the base model at different sparsity levels.
Prunes neurons based on their importance to Python code generation.
"""

import os
import sys
import argparse
import json
from collections import defaultdict
from typing import Optional

import click
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

from sae_scoping.datasets.qa_datasets import format_as_sft_dataset, validate_qa_dataset


# ---------------------------------------------------------------------------
# Shared helpers (model allowlist, attribute validators, dataset prep)
# ---------------------------------------------------------------------------
# TODO(claude) the block below up to `move_to_device` is duplicated verbatim in
# `prune_and_train.py`. Extract into a sibling `shared.py` and import via
# importlib (dataset constants, `confirm_supported_model`, `validate_mlp_*`,
# `load_pruning_dataset`, `tokenize_pruning_dataset`).

SUPPORTED_MODEL_PATTERNS = (
    "google/gemma-2-",
    "google/gemma-3-",
    "NousResearch/Llama-3.2-1B",
)

STEMQA_DATASET = "4gate/StemQAMixture"
STEMQA_CONFIGS = ("biology", "chemistry", "math", "physics")
CODEPARROT_DATASET = "codeparrot/github-code"
SUPPORTED_DATASETS = (STEMQA_DATASET, CODEPARROT_DATASET)


def confirm_supported_model(model_name: str) -> None:
    """Abort unless `model_name` is on the tested allowlist or the operator confirms.

    The attribution-pruning scripts hardcode `.model.layers[i].mlp.{gate_proj,
    up_proj, down_proj, act_fn}` paths (and, in `prune_and_train`, residual-stream
    + self_attn paths). Other architectures may silently no-op or zero unrelated
    weights, so an explicit operator confirmation is required.
    """
    if any(model_name == p or model_name.startswith(p) for p in SUPPORTED_MODEL_PATTERNS):
        print(f"[confirm_supported_model] {model_name!r} is on the tested allowlist.")
        return
    click.confirm(
        f"Model {model_name!r} is NOT on the tested allowlist "
        f"({', '.join(SUPPORTED_MODEL_PATTERNS)}). The attribution-pruning scripts "
        "assume specific module names (gate_proj/up_proj/down_proj/act_fn). "
        "Continue anyway?",
        abort=True,
    )
    print(f"[confirm_supported_model] proceeding with untested model {model_name!r}.")


def validate_mlp_act_fn(model) -> None:
    """Assert every decoder layer has `.mlp.act_fn` (forward-hook target)."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError(f"Model must expose `.model.layers`; got {type(model).__name__}.")
    for i, layer in enumerate(model.model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "act_fn"):
            raise AttributeError(f"Layer {i}: expected `.mlp.act_fn`.")
    print(f"[validate_mlp_act_fn] OK -- {len(model.model.layers)} layers expose .mlp.act_fn")


def validate_mlp_projections(model) -> None:
    """Assert every decoder layer has `.mlp.gate_proj/up_proj/down_proj` (each with `.weight`)."""
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError(f"Model must expose `.model.layers`; got {type(model).__name__}.")
    for i, layer in enumerate(model.model.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise AttributeError(f"Layer {i}: missing `.mlp`.")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            p = getattr(mlp, proj, None)
            if p is None or not hasattr(p, "weight"):
                raise AttributeError(f"Layer {i}: expected `.mlp.{proj}.weight`.")
    print(f"[validate_mlp_projections] OK -- {len(model.model.layers)} layers expose gate/up/down_proj")


def load_pruning_dataset(
    dataset_name: str,
    split: str,
    num_samples: int,
    skip_samples: int,
    streaming: bool,
    dataset_config: Optional[str] = None,
):
    """Load a slice of `num_samples` rows from a recognized pruning dataset.

    Returns a HuggingFace `Dataset` (non-streaming) or a list of dicts (streaming).
    Recognized datasets follow known paths; unrecognized ones go through a loud
    `click.confirm(abort=True)` then raise NotImplementedError.
    """
    if dataset_name == STEMQA_DATASET:
        if dataset_config not in STEMQA_CONFIGS:
            raise ValueError(
                f"For {STEMQA_DATASET}, dataset_config must be one of {STEMQA_CONFIGS}; "
                f"got {dataset_config!r}."
            )
        if streaming:
            raise ValueError(f"Streaming is not supported for {STEMQA_DATASET}; do not pass --streaming.")
        ds = load_dataset(dataset_name, dataset_config, split=split)
        validate_qa_dataset(ds)
        end = skip_samples + num_samples
        if end > len(ds):
            raise ValueError(
                f"{STEMQA_DATASET}/{dataset_config}/{split} has {len(ds)} rows; "
                f"cannot satisfy skip={skip_samples} + num_samples={num_samples} = {end}."
            )
        return ds.select(range(skip_samples, end))
    elif dataset_name == CODEPARROT_DATASET:
        if dataset_config is not None:
            raise ValueError(
                f"{CODEPARROT_DATASET} does not accept dataset_config; got {dataset_config!r}."
            )
        if streaming:
            ds = load_dataset(dataset_name, split=split, languages=["Python"], streaming=True, trust_remote_code=True)
            if skip_samples > 0:
                ds = ds.skip(skip_samples)
            ds = ds.take(num_samples)
            return list(ds)
        ds = load_dataset(dataset_name, split=split, languages=["Python"], trust_remote_code=True)
        return ds.select(range(skip_samples, skip_samples + num_samples))
    else:
        click.confirm(
            f"Dataset {dataset_name!r} is NOT in the recognized list "
            f"({', '.join(SUPPORTED_DATASETS)}). Each recognized dataset has its own "
            f"loader branch (column names, streaming behavior, chat-template assembly). "
            f"There is NO branch for {dataset_name!r} -- this prompt only exists so you "
            f"don't silently get a wrong result. Continuing will raise NotImplementedError. "
            f"Continue anyway?",
            abort=True,
        )
        raise NotImplementedError(
            f"No load_pruning_dataset branch for {dataset_name!r}. Add one above."
        )


def tokenize_pruning_dataset(dataset, tokenizer, dataset_name: str, max_length: int):
    """Tokenize a loaded pruning dataset using its known text-column convention.

    StemQA: chat-templates `question`+`answer` into a `text` column, then tokenizes.
    codeparrot: tokenizes the `code` column directly.
    Unrecognized: loud click.confirm then NotImplementedError.
    """
    if dataset_name == STEMQA_DATASET:
        if isinstance(dataset, list):
            raise ValueError(f"{STEMQA_DATASET} does not support list-of-dicts input.")
        if "text" in dataset.column_names:
            raise ValueError(f"Refusing to overwrite existing `text` column on {STEMQA_DATASET}.")
        ds = format_as_sft_dataset(dataset, tokenizer)
        if "text" not in ds.column_names:
            raise RuntimeError("format_as_sft_dataset did not produce a `text` column.")

        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=max_length)

        return ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
    elif dataset_name == CODEPARROT_DATASET:
        def tokenize_function(examples):
            return tokenizer(examples["code"], truncation=True, max_length=max_length)
        if isinstance(dataset, list):
            return [tokenize_function(sample) for sample in dataset]
        return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    else:
        click.confirm(
            f"Dataset {dataset_name!r} is NOT in the recognized list "
            f"({', '.join(SUPPORTED_DATASETS)}). Tokenization branches assume a specific "
            f"text column (`question`+`answer` for StemQA, `code` for codeparrot). "
            f"Continuing will raise NotImplementedError. Continue anyway?",
            abort=True,
        )
        raise NotImplementedError(
            f"No tokenize_pruning_dataset branch for {dataset_name!r}. Add one above."
        )


# ---------------------------------------------------------------------------


def move_to_device(data, device):
    """Recursively move data to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)([move_to_device(v, device) for v in data])
    return data

def prepare_dataloader(
    model_name,
    dataset_name=CODEPARROT_DATASET,
    dataset_config=None,
    num_samples=1024,
    batch_size=8,
    max_length=1024,
    split="train",
    skip_samples=0,
):
    """Load `num_samples` rows for attribution computation, return a DataLoader.

    Branches on `dataset_name` via `load_pruning_dataset` / `tokenize_pruning_dataset`.
    """
    print(f"Loading {num_samples} samples from {dataset_name}"
          + (f"/{dataset_config}" if dataset_config else "")
          + f" split={split}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    streaming = False
    if dataset_name == CODEPARROT_DATASET:
        streaming = True
        print(
            f"[prepare_dataloader] {dataset_name!r} is unbounded -- enabling streaming "
            f"so we don't try to materialize the entire HF stream."
        )
    dataset = load_pruning_dataset(
        dataset_name=dataset_name,
        split=split,
        num_samples=num_samples,
        skip_samples=skip_samples,
        streaming=streaming,
        dataset_config=dataset_config,
    )
    tokenized_dataset = tokenize_pruning_dataset(dataset, tokenizer, dataset_name, max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)

    return dataloader


def compute_attribution_scores(model, dataloader, num_batches):
    """
    Compute attribution scores for neurons based on Python code.
    Attribution = -activation * gradient_of_loss
    """
    print(f"Computing attribution scores on {num_batches} batches...")
    validate_mlp_act_fn(model)

    def get_attribution_hook(cache, name, hook_cache):
        def attribution_hook(module, input, output):
            def backward_hook(grad):
                # Attribution: -activation * gradient
                modified_grad = -output.detach() * grad
                cache[name] = modified_grad
                return grad
            hook_cache[name] = output.register_hook(backward_hook)
            return None
        return attribution_hook
    
    scores = {layeri: 0 for layeri in range(len(model.model.layers))}
    total_activations = {layeri: 0 for layeri in range(len(model.model.layers))}
    
    # Get device from model
    device = next(model.parameters()).device
    
    for i, batch in enumerate(tqdm(dataloader, desc="Computing attribution")):
        if i >= num_batches:
            break
        
        cache = {}
        forward_hooks = {}
        backward_handles = {}
        
        # Register hooks on MLP activation functions
        for layeri in range(len(model.model.layers)):
            forward_hooks[layeri] = model.model.layers[layeri].mlp.act_fn.register_forward_hook(
                get_attribution_hook(cache, layeri, backward_handles)
            )
        
        # Move batch to device - ensure all dict values are moved
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        # Aggregate attribution scores
        for layeri in range(len(model.model.layers)):
            attrs = cache[layeri]
            scores[layeri] += attrs.sum(dim=tuple(range(attrs.ndim - 1))).detach().abs()
            total_activations[layeri] += attrs.shape[0] * attrs.shape[1]
            forward_hooks[layeri].remove()
            backward_handles[layeri].remove()
        
        del cache
        del forward_hooks
        del backward_handles
        model.zero_grad()
    
    # Average scores
    for layeri in scores:
        scores[layeri] /= total_activations[layeri]
    
    return scores


def prune_by_attribution(model, attribution_scores, sparsity):
    """
    Prune neurons with lowest attribution scores.
    
    Returns:
        pruned_neurons: List of (layer_idx, neuron_idx) tuples
        neurons_per_layer: Dict of neurons pruned per layer
    """
    validate_mlp_projections(model)

    # Create list of (layer, neuron, score) tuples
    score_tuples = []
    for layeri in range(len(model.model.layers)):
        for neuroni in range(attribution_scores[layeri].shape[0]):
            score_tuples.append((layeri, neuroni, attribution_scores[layeri][neuroni].item()))
    
    # Sort by score (lowest first) and prune bottom sparsity%
    score_tuples.sort(key=lambda x: x[2])
    num_to_prune = int(sparsity * len(score_tuples))
    
    print(f"\nPruning {num_to_prune} / {len(score_tuples)} neurons ({sparsity:.1%})")
    
    pruned_neurons = []
    neurons_per_layer = defaultdict(int)
    
    # Prune lowest-scoring neurons
    with torch.no_grad():
        for i in range(num_to_prune):
            layeri, neuroni, score = score_tuples[i]
            
            model.model.layers[layeri].mlp.gate_proj.weight[neuroni, :] = 0
            model.model.layers[layeri].mlp.up_proj.weight[neuroni, :] = 0
            model.model.layers[layeri].mlp.down_proj.weight[:, neuroni] = 0
            
            pruned_neurons.append((layeri, neuroni))
            neurons_per_layer[layeri] += 1
    
    print(f"Neurons pruned per layer (showing non-zero only):")
    for layeri in sorted(neurons_per_layer.keys()):
        print(f"  Layer {layeri}: {neurons_per_layer[layeri]} / {model.config.intermediate_size}")
    
    return pruned_neurons, dict(neurons_per_layer)


def main():
    parser = argparse.ArgumentParser(description="Create attribution-based pruned models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="NousResearch/Llama-3.2-1B",
        help="Base model to prune"
    )
    parser.add_argument(
        "--sparsity_levels",
        type=float,
        nargs="+",
        default=[0.3, 0.63, 0.8],
        help="Neuron sparsity levels to test"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1024,
        help="Number of samples used for attribution"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for attribution computation"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=CODEPARROT_DATASET,
        choices=list(SUPPORTED_DATASETS),
        help=(
            f"Dataset for attribution. {STEMQA_DATASET} requires --dataset_config; "
            f"{CODEPARROT_DATASET} streams Python code (no config)."
        ),
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help=(
            f"Subset name -- required for {STEMQA_DATASET} (one of "
            f"{list(STEMQA_CONFIGS)}); must be unset for {CODEPARROT_DATASET}."
        ),
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default=None,
        help="Base output directory (defaults to $SCRATCH/iaifi_lab/Lab/ericjm/narrow/attribution_pruned)"
    )

    args = parser.parse_args()

    confirm_supported_model(args.model_name)

    # Set up output directory
    if args.output_base_dir is None:
        scratch = os.environ.get('SCRATCH', '/tmp')
        args.output_base_dir = os.path.join(scratch, 'iaifi_lab/Lab/ericjm/narrow/attribution_pruned')
    
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"Attribution-Based Neuron Pruning")
    print(f"{'='*80}")
    print(f"Base model: {args.model_name}")
    print(f"Sparsity levels: {args.sparsity_levels}")
    print(f"Attribution samples: {args.num_samples}")
    print(f"Dataset: {args.dataset_name}"
          + (f" (config={args.dataset_config})" if args.dataset_config else ""))
    print(f"Output directory: {args.output_base_dir}")
    print(f"{'='*80}\n")

    # Load model once for attribution computation
    print("Loading model for attribution...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    # Prepare attribution data
    dataloader = prepare_dataloader(
        args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=1024,
    )
    
    # Compute attribution scores (do this once, use for all sparsity levels)
    num_batches = args.num_samples // args.batch_size
    attribution_scores = compute_attribution_scores(model, dataloader, num_batches)
    
    # Save attribution scores for reference
    attribution_file = os.path.join(args.output_base_dir, "attribution_scores.pt")
    torch.save(attribution_scores, attribution_file)
    print(f"\nSaved attribution scores to: {attribution_file}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Clean up and reload for each sparsity level
    del model
    torch.cuda.empty_cache()
    
    # Process each sparsity level
    for sparsity in args.sparsity_levels:
        print(f"\n{'='*80}")
        print(f"Processing sparsity level: {sparsity:.2%}")
        print(f"{'='*80}")
        
        # Load fresh model
        print(f"Loading fresh model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float32,
            device_map="cpu"  # Keep on CPU to save GPU memory
        )
        
        # Prune based on attribution scores
        pruned_neurons, neurons_per_layer = prune_by_attribution(
            model,
            attribution_scores,
            sparsity
        )
        
        # Save pruned model
        output_dir = os.path.join(args.output_base_dir, f"sparsity_{sparsity}")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving model to: {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save pruning statistics
        stats = {
            "base_model": args.model_name,
            "pruning_method": "attribution",
            "neuron_sparsity": sparsity,
            "total_neurons": sum(layer.mlp.gate_proj.out_features for layer in model.model.layers),
            "neurons_pruned": len(pruned_neurons),
            "neurons_per_layer": neurons_per_layer,
            "num_attribution_samples": args.num_samples,
            "pruned_neurons": pruned_neurons
        }
        
        stats_file = os.path.join(output_dir, "pruning_stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"Saved pruning statistics to: {stats_file}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"All attribution-based models created successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


