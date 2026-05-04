"""Shared helpers for the attribution-pruning baselines in this directory.

Loaded by `create_attribution_pruned_models.py` and `prune_and_train.py` via
`importlib.util` from the sibling path so neither depends on PYTHONPATH or the
project's package layout. Exposes:

- `SUPPORTED_MODEL_PATTERNS`, `confirm_supported_model` -- model allowlist gate
- `validate_mlp_act_fn`, `validate_mlp_projections`,
  `validate_residual_stream_attrs` -- loud `AttributeError` on architectures
  that don't expose the module paths the pruners walk by name
- `STEMQA_DATASET`, `STEMQA_CONFIGS`, `CODEPARROT_DATASET`, `SUPPORTED_DATASETS`
  -- dataset constants
- `load_pruning_dataset`, `tokenize_pruning_dataset` -- exhaustive
  if/elif/else over recognized datasets; unknown datasets go through
  `click.confirm(abort=True)` then raise NotImplementedError so silent
  misconfiguration is impossible.
"""

from typing import Optional

import click
from datasets import load_dataset

from sae_scoping.datasets.qa_datasets import format_as_sft_dataset, validate_qa_dataset


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
        "assume specific module names (gate_proj/up_proj/down_proj/act_fn, "
        "self_attn.q/k/v/o_proj, input_layernorm/post_attention_layernorm, "
        "embed_tokens, model.norm). Continue anyway?",
        abort=True,
    )
    print(f"[confirm_supported_model] proceeding with untested model {model_name!r}.")


def validate_mlp_act_fn(model) -> None:
    """Assert every decoder layer has `.mlp.act_fn` (forward-hook target)."""
    # NOTE: only used by create_attribution_pruned_models.py
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


def validate_residual_stream_attrs(model) -> None:
    """Assert the residual-stream pruner's full attribute path is present.

    Walks every attribute that `mask_by_gradient_attribution` indexes by name:
    embed_tokens, model.norm, per-layer input/post LNs, and self_attn q/k/v/o_proj.
    """
    # NOTE: only used by prune_and_train.py
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError(f"Model must expose `.model.layers`; got {type(model).__name__}.")
    if not hasattr(model.model, "embed_tokens") or not hasattr(model.model.embed_tokens, "weight"):
        raise AttributeError("Model must expose `.model.embed_tokens.weight`.")
    if not hasattr(model.model, "norm") or not hasattr(model.model.norm, "weight"):
        raise AttributeError("Model must expose `.model.norm.weight`.")
    for i, layer in enumerate(model.model.layers):
        for ln in ("input_layernorm", "post_attention_layernorm"):
            ln_mod = getattr(layer, ln, None)
            if ln_mod is None or not hasattr(ln_mod, "weight"):
                raise AttributeError(f"Layer {i}: expected `.{ln}.weight`.")
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise AttributeError(f"Layer {i}: missing `.self_attn`.")
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            p = getattr(attn, proj, None)
            if p is None or not hasattr(p, "weight"):
                raise AttributeError(f"Layer {i}: expected `.self_attn.{proj}.weight`.")
    print(f"[validate_residual_stream_attrs] OK -- embed_tokens + model.norm + "
          f"{len(model.model.layers)} layers expose LNs + q/k/v/o_proj")


def load_pruning_dataset(
    dataset_name: str,
    split: str,
    num_samples: int,
    skip_samples: int,
    streaming: bool,
    dataset_config: Optional[str] = None,
    materialize: bool = True,
):
    """Load a slice of `num_samples` rows from a recognized pruning dataset.

    Returns a HuggingFace `Dataset` (non-streaming) or, for streaming codeparrot,
    a list of dicts (when `materialize=True`, the default) or the lazy
    `IterableDataset` (when `materialize=False`). Materialization is right for
    the attribution loop (small N) and wrong for the HF Trainer pipeline (huge
    N) -- the trainer caller passes `materialize=False` explicitly.

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
            raise ValueError(f"Streaming is not supported for {STEMQA_DATASET}; pass without --streaming.")
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
            return list(ds) if materialize else ds
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
