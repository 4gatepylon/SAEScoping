"""Pruning operations for the attribution-pruning baselines.

Factored out of create_attribution_pruned_models.py so other scripts
(e.g. eval_attribution.py, attribution_sweep_with_pgd.py) can reuse the
same zeroing logic without saving/loading full checkpoints.
"""

import os
import importlib.util as _ilu
from collections import defaultdict
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_spec = _ilu.spec_from_file_location(
    "baselines_narrow_shared",
    os.path.join(os.path.dirname(__file__), "shared.py"),
)
shared = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(shared)


def prune_model_by_attribution(model, attribution_scores, sparsity):
    """Zero out MLP gate/up/down rows for the lowest-scoring neurons.

    Neurons are ranked globally (across all layers) by their attribution
    score.  The bottom ``sparsity`` fraction is pruned by zeroing
    ``gate_proj[neuron, :]``, ``up_proj[neuron, :]``, and
    ``down_proj[:, neuron]`` in-place.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The model whose MLP weights will be zeroed in-place.
    attribution_scores : dict[int, Tensor]
        Per-layer 1-D tensors of neuron importance (as returned by
        ``compute_attribution_scores``).
    sparsity : float
        Fraction of neurons to prune, in [0, 1].

    Returns
    -------
    pruned_neurons : list[tuple[int, int]]
        ``(layer_idx, neuron_idx)`` for every pruned neuron.
    neurons_per_layer : dict[int, int]
        Count of pruned neurons keyed by layer index.
    """
    shared.validate_mlp_projections(model)
    layers = shared.text_decoder(model).layers

    score_tuples = []
    for layeri in range(len(layers)):
        for neuroni in range(attribution_scores[layeri].shape[0]):
            score_tuples.append(
                (layeri, neuroni, attribution_scores[layeri][neuroni].item())
            )

    score_tuples.sort(key=lambda x: x[2])
    num_to_prune = int(sparsity * len(score_tuples))

    print(f"\nPruning {num_to_prune} / {len(score_tuples)} neurons ({sparsity:.1%})")

    pruned_neurons = []
    neurons_per_layer = defaultdict(int)

    with torch.no_grad():
        for i in range(num_to_prune):
            layeri, neuroni, _score = score_tuples[i]
            layers[layeri].mlp.gate_proj.weight[neuroni, :] = 0
            layers[layeri].mlp.up_proj.weight[neuroni, :] = 0
            layers[layeri].mlp.down_proj.weight[:, neuroni] = 0
            pruned_neurons.append((layeri, neuroni))
            neurons_per_layer[layeri] += 1

    print("Neurons pruned per layer (showing non-zero only):")
    for layeri in sorted(neurons_per_layer.keys()):
        print(
            f"  Layer {layeri}: {neurons_per_layer[layeri]} "
            f"/ {layers[layeri].mlp.gate_proj.out_features}"
        )

    return pruned_neurons, dict(neurons_per_layer)


def load_and_prune_model(
    model_name: str,
    attribution_scores: dict,
    sparsity: float,
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer, list[tuple[int, int]], dict[int, int]]:
    """Load a fresh model, prune it by attribution in-place, return everything.

    This avoids saving/loading full checkpoints -- the caller only needs the
    attribution scores tensor and a sparsity value to reconstruct any pruned
    model on the fly.

    Returns
    -------
    model : AutoModelForCausalLM
        The pruned model (weights zeroed in-place).
    tokenizer : AutoTokenizer
        Matching tokenizer with pad_token set.
    pruned_neurons : list[tuple[int, int]]
        ``(layer_idx, neuron_idx)`` for every pruned neuron.
    neurons_per_layer : dict[int, int]
        Count of pruned neurons keyed by layer index.
    """
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **shared.load_model_kwargs(model_name),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pruned_neurons, neurons_per_layer = prune_model_by_attribution(
        model, attribution_scores, sparsity
    )
    return model, tokenizer, pruned_neurons, neurons_per_layer
