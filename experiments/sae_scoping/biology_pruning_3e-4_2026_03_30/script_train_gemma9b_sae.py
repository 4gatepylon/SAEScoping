"""
script_train_gemma9b_sae.py

Train gemma-2-9b-it on OOD (non-biology) data, optionally with a pruned SAE.

Modes:
  --sae (default): pruned SAE hooked at hookpoint, layers after are trainable
  --vanilla:       no SAE, same layer freezing as baseline
  --train-all-layers: override to train all layers (with either mode)

Differences from script_2025_12_08_train_gemma9b_sae.py:
  - Supports any question/answer dataset via dataset_utils (not hardcoded)
  - JSON config file override via --config
  - Utility eval callback (LLM judge) instead of only loss-based eval
  - Weight integrity validation (fingerprint-based, see _create_handle)
  - Explicit documentation of tied embedding behavior
  - Modular SAE loading via _load_pruned_sae()
  - No dependency on sae_scoping.trainers.sae_enhanced.train (self-contained)

Tied embeddings note:
  Gemma 2 uses tie_word_embeddings=True, meaning model.embed_tokens.weight and
  lm_head.weight are the SAME tensor. When we freeze layers <= hookpoint, the
  embedding layer is frozen but lm_head is kept trainable (matching behavior in
  sae_scoping/trainers/sae_enhanced/train.py:57). Because they share the same
  tensor, lm_head gradients WILL update the shared embedding weight. This is
  intentional and matches the original script's behavior. If you need truly
  frozen embeddings, you must also freeze lm_head or untie them first.
"""

from __future__ import annotations

import gc
import json
import os
import re
from functools import partial
from pathlib import Path

import click
import torch
from safetensors.torch import load_file
from sae_lens import SAE
from transformers import AutoTokenizer, Gemma2ForCausalLM, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer

from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae
# Hooks: same as sae_scoping/trainers/sae_enhanced/train.py:28-31
from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
from sae_scoping.utils.hooks.sae import SAEWrapper

from dataset_utils import load_stem_train_eval, make_eval_conversations
from eval_callback import UtilityEvalCallback


# ---------------------------------------------------------------------------
# Defaults (override via CLI or --config JSON)
# ---------------------------------------------------------------------------

_BASE_MODEL = "google/gemma-2-9b-it"
_SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
_DEFAULT_SAE_ID = "layer_31/width_16k/canonical"
_DEFAULT_HOOKPOINT = "model.layers.31"
_DEFAULT_WANDB_PROJECT = "sae-elicitation-ood"
_DEFAULT_BATCH_SIZE = 2
_DEFAULT_ACCUM = 8
_DEFAULT_MAX_STEPS = 4000
_DEFAULT_LEARNING_RATE = 2e-5
_DEFAULT_SAVE_EVERY = 500
_DEFAULT_SAVE_LIMIT = 10
_DEFAULT_EVAL_EVERY = 100
_DEFAULT_UTILITY_EVAL_EVERY = 0  # disabled by default
_DEFAULT_UTILITY_EVAL_CONVERSATIONS = 50
_DEFAULT_BIOLOGY_UTILITY_EVAL_EVERY = 0  # disabled by default; set same as utility_eval_every to enable
_DEFAULT_BIOLOGY_UTILITY_EVAL_CONVERSATIONS = 50
_DEFAULT_THRESHOLD = 3e-4
_DEFAULT_MAX_LENGTH = 1024

# Weight integrity validation: number of random indices to sample per parameter
_HANDLE_INDICES_PER_PARAM = 64
_HANDLE_SEED = 42


# ---------------------------------------------------------------------------
# Weight integrity validation
# ---------------------------------------------------------------------------

# A "handle" is a lightweight fingerprint of parameter values at specific
# random indices. Create one before training, compare after to verify that
# frozen params didn't change and trainable params did.

WeightHandle = dict[str, tuple[torch.Tensor, torch.Tensor]]  # {name: (indices, values)}


def _create_handle(model: torch.nn.Module, param_names: list[str]) -> WeightHandle:
    """Snapshot values at _HANDLE_INDICES_PER_PARAM random positions per param."""
    rng = torch.Generator().manual_seed(_HANDLE_SEED)
    handle: WeightHandle = {}
    for name, param in model.named_parameters():
        if name not in param_names:
            continue
        flat = param.data.detach().view(-1)
        n = min(_HANDLE_INDICES_PER_PARAM, flat.shape[0])
        indices = torch.randperm(flat.shape[0], generator=rng)[:n]
        handle[name] = (indices, flat[indices].cpu().clone())
    return handle


def _compare_handle(model: torch.nn.Module, handle: WeightHandle) -> tuple[list[str], list[str]]:
    """Compare current param values against a handle. Returns (changed, unchanged)."""
    changed, unchanged = [], []
    params = dict(model.named_parameters())
    for name, (indices, old_values) in handle.items():
        current_values = params[name].data.detach().view(-1)[indices].cpu()
        if torch.allclose(current_values, old_values):
            unchanged.append(name)
        else:
            changed.append(name)
    return changed, unchanged


# Parameters that are frozen but may change due to weight tying.
# Gemma 2 ties embed_tokens.weight with lm_head.weight (tie_word_embeddings=True).
# We freeze embed_tokens but keep lm_head trainable (matching train.py:57), so the
# shared tensor WILL be updated by lm_head gradients. The original script
# (train.py:224-226) has the same issue: it would fail its own assertion if
# embed_tokens were included in the frozen check. In practice the original script
# doesn't hit this because embed_tokens.requires_grad=False means it's excluded
# from named_parameters iteration in the diff check. We handle it explicitly.
_TIED_WEIGHT_EXCEPTIONS = {"model.embed_tokens.weight"}


def _validate_weight_integrity(
    model: torch.nn.Module,
    frozen_handle: WeightHandle,
    trainable_handle: WeightHandle,
) -> None:
    """Assert frozen params unchanged and trainable params changed after training.

    Excludes _TIED_WEIGHT_EXCEPTIONS from the frozen-must-not-change check,
    since tied weights may legitimately change via their trainable counterpart.
    """
    frozen_changed, frozen_unchanged = _compare_handle(model, frozen_handle)
    trainable_changed, trainable_unchanged = _compare_handle(model, trainable_handle)

    # Filter out tied weight exceptions from frozen violations
    real_violations = [n for n in frozen_changed if n not in _TIED_WEIGHT_EXCEPTIONS]
    tied_changed = [n for n in frozen_changed if n in _TIED_WEIGHT_EXCEPTIONS]
    if tied_changed:
        print(f"Note: tied weight(s) changed as expected: {tied_changed}")

    if real_violations:
        raise AssertionError(
            f"Frozen parameters changed during training: {real_violations}"
        )
    if trainable_unchanged:
        raise AssertionError(
            f"Trainable parameters did NOT change during training: {trainable_unchanged}"
        )
    print(f"Weight integrity OK: {len(frozen_unchanged)} frozen unchanged, "
          f"{len(trainable_changed)} trainable changed, "
          f"{len(tied_changed)} tied (expected change)")


# ---------------------------------------------------------------------------
# Layer freezing
# Matches sae_scoping/trainers/sae_enhanced/train.py:45-79
# ---------------------------------------------------------------------------


def _freeze_parameters_before_layer(model: Gemma2ForCausalLM, layer: int) -> list[str]:
    """Freeze all parameters at or before `layer`. Returns frozen param names.

    Behavior matches train.py:45-79:
      - model.layers.N for N <= layer: frozen
      - lm_head: trainable (train.py:57)
      - model.norm: trainable for Gemma2 (train.py:58-59)
      - all other non-layer params (e.g. embed_tokens): frozen (train.py:63-65)
    """
    frozen = []
    for n, p in model.named_parameters():
        if not n.startswith("model.layers"):
            # lm_head stays trainable (train.py:57)
            if "lm_head" in n:
                p.requires_grad = True
            # model.norm stays trainable for Gemma2 (train.py:58-59)
            elif n.startswith("model.norm"):
                p.requires_grad = True
            else:
                # embed_tokens and any other non-layer params: frozen (train.py:63-65)
                p.requires_grad = False
                p.grad = None
                frozen.append(n)
        else:
            # Freeze layers <= hookpoint layer (train.py:67-78)
            match = re.match(r"^model\.layers\.(\d+)\..*$", n)
            assert match is not None, f"Param {n} doesn't match expected pattern"
            if int(match.group(1)) <= layer:
                p.requires_grad = False
                p.grad = None
                frozen.append(n)
    return frozen


# ---------------------------------------------------------------------------
# SAE loading
# ---------------------------------------------------------------------------


def _load_pruned_sae(
    dist_path: Path,
    threshold: float,
    sae_id: str,
    device: torch.device,
):
    """Load distribution, rank neurons, build pruned SAE.

    Matches script_2025_12_08_train_gemma9b_sae.py:129-169.
    """
    dist_data = load_file(str(dist_path))
    distribution: torch.Tensor = dist_data["distribution"]
    neuron_ranking = torch.argsort(distribution, descending=True)
    n_kept = int((distribution >= threshold).sum().item())
    print(f"Keeping {n_kept}/{len(distribution)} SAE neurons (threshold={threshold})")

    sae = SAE.from_pretrained(release=_SAE_RELEASE, sae_id=sae_id, device=device)
    sae = sae.to(device)
    pruned_sae = get_pruned_sae(sae, neuron_ranking, K_or_p=n_kept, T=0.0)
    pruned_sae = pruned_sae.to(device)
    return pruned_sae


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_config(config_path: Path | None) -> dict:
    """Load JSON config file, or return empty dict."""
    if config_path is None:
        return {}
    return json.loads(config_path.read_text())


def _cfg(config: dict, key: str, cli_value, default):
    """Resolve value: CLI override > JSON config > default."""
    if cli_value is not None and cli_value != default:
        return cli_value
    return config.get(key, default)


# ---------------------------------------------------------------------------
# Biology eval conversations
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path), default=None,
              help="JSON file to override defaults. CLI args take precedence over JSON values.")
@click.option("--checkpoint", "-c", type=click.Path(exists=True, path_type=Path), required=True,
              help="Path to HF checkpoint directory.")
@click.option("--dist-path", "-p", type=click.Path(exists=True, path_type=Path), default=None,
              help="Path to distribution.safetensors. Required for SAE mode.")
@click.option("--threshold", "-h", type=float, default=_DEFAULT_THRESHOLD,
              help="SAE neuron pruning threshold.")
@click.option("--sae-id", type=str, default=_DEFAULT_SAE_ID)
@click.option("--hookpoint", type=str, default=_DEFAULT_HOOKPOINT)
@click.option("--vanilla/--sae", "is_vanilla", default=False,
              help="--vanilla: no SAE baseline. --sae (default): train with pruned SAE.")
@click.option("--train-all-layers/--freeze-before-hookpoint", "train_all", default=False)
@click.option("--batch-size", "-b", type=int, default=_DEFAULT_BATCH_SIZE)
@click.option("--accum", "-a", type=int, default=_DEFAULT_ACCUM)
@click.option("--max-steps", "-s", type=int, default=_DEFAULT_MAX_STEPS)
@click.option("--learning-rate", "-lr", type=float, default=_DEFAULT_LEARNING_RATE)
@click.option("--save-every", type=int, default=_DEFAULT_SAVE_EVERY)
@click.option("--save-limit", type=int, default=_DEFAULT_SAVE_LIMIT)
@click.option("--eval-every", type=int, default=_DEFAULT_EVAL_EVERY)
@click.option("--utility-eval-every", type=int, default=_DEFAULT_UTILITY_EVAL_EVERY,
              help="Run LLM judge eval every N steps (0=disabled).")
@click.option("--biology-utility-eval-every", type=int, default=_DEFAULT_BIOLOGY_UTILITY_EVAL_EVERY,
              help="Run biology LLM judge eval every N steps, logged as a separate W&B series "
                   "(0=disabled). Set to the same value as --utility-eval-every to keep in sync.")
@click.option("--subset", type=click.Choice(["physics", "chemistry", "math"]), required=True,
              help="Which StemQAMixture subset to train on.")
@click.option("--max-train-samples", type=int, default=None, help="Cap training samples.")
@click.option("--output-dir", "-o", type=str, default=None)
@click.option("--wandb-project", "-w", type=str, default=_DEFAULT_WANDB_PROJECT)
@click.option("--wandb-run-name", type=str, default=None)
def main(
    config_path: Path | None,
    checkpoint: Path,
    dist_path: Path | None,
    threshold: float,
    sae_id: str,
    hookpoint: str,
    is_vanilla: bool,
    train_all: bool,
    batch_size: int,
    accum: int,
    max_steps: int,
    learning_rate: float,
    save_every: int,
    save_limit: int,
    eval_every: int,
    utility_eval_every: int,
    biology_utility_eval_every: int,
    subset: str,
    max_train_samples: int | None,
    output_dir: str | None,
    wandb_project: str,
    wandb_run_name: str | None,
) -> None:
    """Train a (possibly SAE-enhanced) gemma-2-9b-it checkpoint on OOD STEM data."""
    cfg = _load_config(config_path)
    batch_size = _cfg(cfg, "batch_size", batch_size, _DEFAULT_BATCH_SIZE)
    accum = _cfg(cfg, "accum", accum, _DEFAULT_ACCUM)
    max_steps = _cfg(cfg, "max_steps", max_steps, _DEFAULT_MAX_STEPS)
    learning_rate = _cfg(cfg, "learning_rate", learning_rate, _DEFAULT_LEARNING_RATE)
    save_every = _cfg(cfg, "save_every", save_every, _DEFAULT_SAVE_EVERY)
    save_limit = _cfg(cfg, "save_limit", save_limit, _DEFAULT_SAVE_LIMIT)
    eval_every = _cfg(cfg, "eval_every", eval_every, _DEFAULT_EVAL_EVERY)
    utility_eval_every = _cfg(cfg, "utility_eval_every", utility_eval_every, _DEFAULT_UTILITY_EVAL_EVERY)
    biology_utility_eval_every = _cfg(cfg, "biology_utility_eval_every", biology_utility_eval_every, _DEFAULT_BIOLOGY_UTILITY_EVAL_EVERY)
    threshold = _cfg(cfg, "threshold", threshold, _DEFAULT_THRESHOLD)
    wandb_project = _cfg(cfg, "wandb_project", wandb_project, _DEFAULT_WANDB_PROJECT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not is_vanilla and dist_path is None:
        raise click.UsageError("--dist-path is required when training with SAE (omit --vanilla).")

    # --- Auto-generate names ---
    mode_tag = "vanilla" if is_vanilla else f"sae_h{threshold}"
    layers_tag = "all_layers" if train_all else f"after_{hookpoint.split('.')[-1]}"
    if output_dir is None:
        output_dir = f"./outputs/{mode_tag}/{layers_tag}/{subset}/{checkpoint.name}"
    if wandb_run_name is None:
        wandb_run_name = f"{mode_tag}/{layers_tag}/{subset}/{checkpoint.name}"

    os.environ["WANDB_PROJECT"] = wandb_project

    # --- Load model ---
    # Matches script_2025_12_08_train_gemma9b_sae.py:151-161
    print(f"Loading model from {checkpoint}...")
    model = Gemma2ForCausalLM.from_pretrained(
        str(checkpoint), torch_dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager",
    )
    model = model.to(device)
    model.gradient_checkpointing_disable()
    if hasattr(model, "model"):
        model.model.gradient_checkpointing = False

    tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    # --- Freeze layers ---
    # Matches sae_scoping/trainers/sae_enhanced/train.py:147-159
    if not train_all:
        hp_match = re.match(r"^model\.layers\.(\d+)$", hookpoint)
        if not hp_match:
            raise click.BadParameter(f"Invalid hookpoint: {hookpoint}")
        layer_num = int(hp_match.group(1))
        frozen_names = _freeze_parameters_before_layer(model, layer_num)
        print(f"Froze {len(frozen_names)} parameters (layers 0-{layer_num})")
    else:
        frozen_names = []
        print("Training all layers (nothing frozen)")

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_names)}")
    print(f"Frozen parameters: {len(frozen_names)}")

    # --- Weight integrity handles (before training) ---
    frozen_handle = _create_handle(model, frozen_names)
    trainable_handle = _create_handle(model, trainable_names)

    # --- Load SAE ---
    # Matches script_2025_12_08_train_gemma9b_sae.py:129-169
    pruned_sae = None
    if not is_vanilla:
        pruned_sae = _load_pruned_sae(dist_path, threshold, sae_id, device)

    # --- Load datasets ---
    print(f"Loading OOD dataset: {subset}...")
    train_dataset, eval_dataset = load_stem_train_eval(
        tokenizer, subsets=(subset,), max_train_samples_per_subset=max_train_samples,
    )
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # --- Callbacks ---
    callbacks = []
    if utility_eval_every > 0:
        eval_convos = make_eval_conversations(
            tokenizer, subsets=(subset,), max_samples=_DEFAULT_UTILITY_EVAL_CONVERSATIONS,
        )
        callbacks.append(UtilityEvalCallback(
            eval_every=utility_eval_every,
            metric_name="judge",
            tokenizer=tokenizer,
            eval_conversations=eval_convos,
            batch_size=batch_size,
            wandb_prefix="utility_eval/ood",
        ))
    if biology_utility_eval_every > 0:
        bio_convos = make_eval_conversations(
            tokenizer, subsets=("biology",), max_samples=_DEFAULT_BIOLOGY_UTILITY_EVAL_CONVERSATIONS,
        )
        callbacks.append(UtilityEvalCallback(
            eval_every=biology_utility_eval_every,
            metric_name="judge",
            tokenizer=tokenizer,
            eval_conversations=bio_convos,
            batch_size=batch_size,
            wandb_prefix="utility_eval/biology",
        ))

    # --- SFT config ---
    # Matches script_2025_12_08_train_gemma9b_sae.py:286-315
    sft_config = SFTConfig(
        output_dir=output_dir,
        run_name=wandb_run_name,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=max_steps,
        gradient_accumulation_steps=accum,
        num_train_epochs=1,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.1,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=eval_every,
        save_steps=save_every,
        bf16=True,
        save_total_limit=save_limit,
        report_to="wandb",
        max_length=_DEFAULT_MAX_LENGTH,
        gradient_checkpointing=False,
    )

    # --- Train ---
    # Matches sae_scoping/trainers/sae_enhanced/train.py:182-205
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )

    if pruned_sae is not None:
        # Hook SAE into forward pass (train.py:199-203)
        sae_wrapper = SAEWrapper(pruned_sae)
        hook_dict = {hookpoint: partial(filter_hook_fn, sae_wrapper)}
        with named_forward_hooks(model, hook_dict):
            trainer.train()
    else:
        trainer.train()

    # --- Weight integrity validation ---
    # Matches spirit of train.py:218-227 but uses handle-based approach
    _validate_weight_integrity(model, frozen_handle, trainable_handle)

    print("Training complete.")

    # --- Cleanup ---
    del model
    if pruned_sae is not None:
        del pruned_sae
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
