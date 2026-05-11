#!/usr/bin/env python3
"""
Create attribution-based pruned versions of the base model at different sparsity levels.
Prunes neurons based on their importance to Python code generation.

TODO(hadriano) claude claims this script has a lot of gotchas/bugs. Worth looking into later (2026/05/04).
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)

# TODO(hadriano) make this more automated please
# NOTE: load sibling shared.py without depending on PYTHONPATH or package layout.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("baselines_narrow_shared", os.path.join(os.path.dirname(__file__), "shared.py"))
shared = _ilu.module_from_spec(_spec); _spec.loader.exec_module(shared)
prune_model_by_attribution = shared.prune_model_by_attribution


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
    dataset_name=shared.CODEPARROT_DATASET,
    dataset_config=None,
    num_samples=1024,
    batch_size=8,
    max_length=1024,
    split="train",
    skip_samples=0,
):
    """Load `num_samples` rows for attribution computation, return a DataLoader.

    Branches on `dataset_name` via `shared.load_pruning_dataset` /
    `shared.tokenize_pruning_dataset`.
    """
    batch_size = shared.safe_batch_size(batch_size, num_samples, label="attribution")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    streaming = False
    if dataset_name == shared.CODEPARROT_DATASET:
        streaming = True
        print(
            f"[prepare_dataloader] {dataset_name!r} is unbounded -- enabling streaming "
            f"so we don't try to materialize the entire HF stream."
        )
    dataset = shared.load_pruning_dataset(
        dataset_name=dataset_name,
        split=split,
        num_samples=num_samples,
        skip_samples=skip_samples,
        streaming=streaming,
        dataset_config=dataset_config,
    )
    tokenized_dataset = shared.tokenize_pruning_dataset(dataset, tokenizer, dataset_name, max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)

    return dataloader


def compute_attribution_scores(model, dataloader, num_batches):
    """
    Compute attribution scores for neurons based on Python code.
    Attribution = -activation * gradient_of_loss

    It does this in a unique and somewhat unusual way:
    1. They register a MODULE Forward pass hook to trigger (2) when mlp.act_fn runs forward.
    2. Inside the forward hook, they grab output (the post-SiLU ACTIVATION) register a backwards hook on the TENSOR to attach a backward
        callback to it directly.
    3. When loss.backward() runs, the gradient at the ACTIVATION tensor is passed via the callback.
    It is done this way because the activation tensor does NOT exist before the forward pass.

    TODO(hadriano) how can we reduce memory? It looks like we NEED to store the activation(s) for ^. They are unlikely to be
    the bottleneck though? d_mlp * n_layers * n_batch (realistically 8K * 40 * 1 * 2B @ bfloat16 -> 640KB <= 1GB).
    """
    print(f"Computing attribution scores on {num_batches} batches...")
    shared.validate_mlp_act_fn(model)
    layers = shared.text_decoder(model).layers
    # NOTE(hadriano): no model.eval() here -- HF returns the model in training=True by default, so dropout 
    # (if nonzero) makes attribution grads non-deterministic; matches prune_and_train.py's upstream-narrow behavior
    def get_attribution_hook(cache, name, hook_cache):
        def attribution_hook(module, input, output):
            def backward_hook(grad):
                # Attribution: -activation * gradient
                modified_grad = -output.detach() * grad
                # TODO(hadriano): MEMORY BOTTLENECK. We cache the full (B, S, d_mlp) tensor here for *every* MLP layer
                # simultaneously, then reduce later in the aggregation loop below. Observation by Claude.
                cache[name] = modified_grad
                return grad
            hook_cache[name] = output.register_hook(backward_hook)
            return None
        return attribution_hook

    scores = {layeri: 0 for layeri in range(len(layers))}
    total_activations = {layeri: 0 for layeri in range(len(layers))}

    # Get device from model
    device = next(model.parameters()).device

    for i, batch in enumerate(tqdm(dataloader, desc="Computing attribution")):
        if i >= num_batches:
            break

        cache = {}
        forward_hooks = {}
        backward_handles = {}

        # Register hooks on MLP activation functions
        # NOTE(hadriano): if gradient checkpointing is on, this forward hook re-fires during backward recompute and
        # overwrites cache[name] with stale activations -- attribution would be silently wrong.
        for layeri in range(len(layers)):
            # TODO(hadriano) is this registering a forwards hook that registerd a backwards hook? WTF?
            forward_hooks[layeri] = layers[layeri].mlp.act_fn.register_forward_hook(
                get_attribution_hook(cache, layeri, backward_handles)
            )

        # Move batch to device - ensure all dict values are moved
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Aggregate attribution scores
        for layeri in range(len(layers)):
            attrs = cache[layeri]
            scores[layeri] += attrs.sum(dim=tuple(range(attrs.ndim - 1))).detach().abs()
            # TODO(hadriano): counts pad positions; harmless for top-k ranking (same scalar denominator for all neurons)
            #   but verify if you ever read raw score magnitudes
            total_activations[layeri] += attrs.shape[0] * attrs.shape[1]
            forward_hooks[layeri].remove()
            backward_handles[layeri].remove()
        
        del cache
        del forward_hooks
        del backward_handles
        # TODO(hadriano): default `zero_grad()` zeros the .grad tensors but keeps them allocated -- one
        # parameter-shaped buffer per param persists across batches. Pass `set_to_none=True` to release them
        # so peak memory only has to hold the current batch's grad graph. Observation by Claude.
        model.zero_grad()
    
    # Average scores
    for layeri in scores:
        assert total_activations[layeri] > 0, (
            f"Layer {layeri}: zero activations accumulated -- the dataloader yielded no "
            f"batches before num_batches={num_batches} was reached. Check num_samples / "
            f"batch_size / streaming flags."
        )
        scores[layeri] /= total_activations[layeri]

    return scores


def prune_by_attribution(model, attribution_scores, sparsity):
    """Thin wrapper -- delegates to `shared.prune_model_by_attribution`."""
    return prune_model_by_attribution(model, attribution_scores, sparsity)


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
        default=shared.CODEPARROT_DATASET,
        choices=list(shared.SUPPORTED_DATASETS),
        help=(
            f"Dataset for attribution. {shared.STEMQA_DATASET} requires --dataset_config; "
            f"{shared.CODEPARROT_DATASET} streams Python code (no config)."
        ),
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help=(
            f"Subset name -- required for {shared.STEMQA_DATASET} (one of "
            f"{list(shared.STEMQA_CONFIGS)}); must be unset for {shared.CODEPARROT_DATASET}."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16"],
        help="Model dtype. bfloat16 roughly halves memory for both GPU attribution and CPU pruning reloads.",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        required=True,
        help="Base output directory. One subdirectory per --sparsity_levels entry will be created here. NOTE: saved checkpoints have no pruning metadata baked into the weights file; pruned weights are just zero rows/cols. Fine-tuning without a re-masking trainer will gradually undo the pruning.",
    )

    args = parser.parse_args()

    shared.validate_args(args, sparsity_attrs=("sparsity_levels",))

    torch_dtype = getattr(torch, args.dtype)

    os.makedirs(args.output_base_dir, exist_ok=True)
    
    print(f"{'='*80}")
    print(f"Attribution-Based Neuron Pruning")
    print(f"{'='*80}")
    print(f"Base model: {args.model_name}")
    print(f"Sparsity levels: {args.sparsity_levels}")
    print(f"Attribution samples: {args.num_samples}")
    print(f"Dtype: {args.dtype}")
    print(f"Dataset: {args.dataset_name}"
          + (f" (config={args.dataset_config})" if args.dataset_config else ""))
    print(f"Output directory: {args.output_base_dir}")
    print(f"{'='*80}\n")

    # Load model once for attribution computation
    print("Loading model for attribution...")
    
    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")) != 1:
        raise ValueError("This script supports only exactly a single GPU. Please set CUDA_VISIBLE_DEVICES to a single GPU.")
    # TODO(hadriano) please do some backtesting using bfloat16 for the original Michaud paper to
    # confirm that this SHOULD not introduce numerical issues.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        **shared.load_model_kwargs(args.model_name),
    )

    # Prepare attribution data
    dataloader = prepare_dataloader(
        args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        max_length=shared.NARROW_MAX_LENGTH,
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
            torch_dtype=torch_dtype,
            device_map="cpu",
            **shared.load_model_kwargs(args.model_name),
        )
        # TODO(hadriano) we need to add an evaluation function here that returns a value that goes
        # into the stats output below and also causes us to save some JSON lofs for LLM judge.
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
        # TODO(hadriano): create_attribution_pruned_models.py stores a full-size HF
        # checkpoint but you could store just the neuron masks for a massive reduction in
        # disk usage. You could also just shrink the parameters for less of a reduction.
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save pruning statistics
        stats = {
            "base_model": args.model_name,
            "pruning_method": "attribution",
            "neuron_sparsity": sparsity,
            "total_neurons": sum(layer.mlp.gate_proj.out_features for layer in shared.text_decoder(model).layers),
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


