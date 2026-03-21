"""
sweep_eval_temp.py

Sweep over pruning sparsity levels, evaluating validation loss and LLM-judged
generation quality at each sparsity fraction. Results are logged to wandb.

For each sparsity level in the sweep:
  1. Restore model to original weights.
  2. Zero out the lowest-saliency fraction of weights.
  3. Compute mean cross-entropy loss on the biology validation set.
  4. Generate responses for n_generation_samples questions; grade with LLM judges.
  5. Log everything to wandb and move to the next level.

Saliency criteria
-----------------
gradient  :  score = |grad|
taylor    :  score = |grad * weight|   (Taylor first-order approximation)

NOTE: Saliency scores are computed once from the original weights (before any
pruning), so the Taylor scores are never corrupted by earlier pruning steps.

NOTE: Saving a CPU copy of all model parameters (~18 GB for 9B bf16) is
required to restore weights between sparsity levels. Ensure sufficient CPU RAM.

CLI usage (single run):
    python sweep_eval_temp.py run --saliency-path biology/ema_grads.safetensors
    python sweep_eval_temp.py run --saliency-path biology/ema_grads.safetensors \\
        --saliency-type taylor --precision 0.05

CLI usage (batch — all saliency files × both criteria, distributed across GPUs):
    python sweep_eval_temp.py batch --saliency-dir biology/ --devices 0,1,2,3
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import click
import torch
import wandb
from datasets import Dataset, load_dataset
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from tqdm import tqdm

from grade_chats import grade_chats, GradedChats
from model_generator import HFGenerator


_DEFAULT_MODEL_ID = "google/gemma-2-9b-it"
_DEFAULT_DATASET = "4gate/StemQAMixture"
_DEFAULT_SUBSET = "biology"
_DEFAULT_N_SAMPLES = 512
_DEFAULT_BATCH_SIZE = 4
_DEFAULT_N_GENERATION_SAMPLES = 32
_DEFAULT_MAX_SEQ = 1024
_DEFAULT_MAX_NEW_TOKENS = 256
_DEFAULT_PRECISION = 0.05  # 21 levels: 0.0, 0.05, ..., 1.0
_DEFAULT_WANDB_PROJECT = "sae-scoping-pruning"
_DEFAULT_OUTPUT_DIR = Path("./sweep_generations")
_CHAT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "gemma2_chat_template_system_prompt.j2"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _load_val_dataset(dataset_name: str, subset: str, n: int, seed: int) -> Dataset:
    ds = load_dataset(dataset_name, subset, split="validation")
    assert "question" in ds.column_names, f"Missing 'question' column: {ds.column_names}"
    assert "answer" in ds.column_names, f"Missing 'answer' column: {ds.column_names}"
    if n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def _format_texts_for_loss(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    """Full question+answer chat text for cross-entropy loss."""
    texts = []
    for row in dataset:
        messages = [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]
        texts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        )
    return texts


def _format_conversations_for_generation(dataset: Dataset) -> list[list[dict]]:
    """0-turn OpenAI-format conversations (question only, no answer)."""
    return [[{"role": "user", "content": row["question"]}] for row in dataset]


# ---------------------------------------------------------------------------
# Saliency scoring
# ---------------------------------------------------------------------------


def compute_saliency_scores(
    model: AutoModelForCausalLM,
    saliency_tensors: dict[str, torch.Tensor],
    saliency_type: str,
) -> dict[str, torch.Tensor]:
    """
    Compute per-parameter saliency scores from a loaded safetensors map.

    gradient : |grad|
    taylor   : |grad * weight|
    """
    if saliency_type not in ("gradient", "taylor"):
        raise ValueError(f"Unknown saliency_type '{saliency_type}'. Choose 'gradient' or 'taylor'.")
    scores: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name not in saliency_tensors:
            continue
        grad = saliency_tensors[name].float().to(param.device)
        if saliency_type == "gradient":
            scores[name] = grad.abs()
        else:
            scores[name] = (grad * param.data.float()).abs()
    return scores


# ---------------------------------------------------------------------------
# Weight save / restore / prune
# ---------------------------------------------------------------------------


def save_original_weights(model: AutoModelForCausalLM) -> dict[str, torch.Tensor]:
    """Clone all parameter data to CPU for later restoration."""
    return {name: param.data.cpu().clone() for name, param in model.named_parameters()}


def restore_original_weights(
    model: AutoModelForCausalLM,
    original_weights: dict[str, torch.Tensor],
) -> None:
    """Restore model parameters in-place from a CPU copy."""
    for name, param in model.named_parameters():
        param.data.copy_(original_weights[name].to(param.device))


_PRUNING_PROBE_N = 64  # number of random indices sampled per tensor for assertions

# Type alias: per-parameter snapshot of (flat indices, pre-pruning float32 values)
_PruningProbe = dict[str, tuple[torch.Tensor, torch.Tensor]]


# ---------------------------------------------------------------------------
# Pruning validators (public so they can be unit-tested independently)
# ---------------------------------------------------------------------------


def sample_pruning_probes(
    model: AutoModelForCausalLM,
    saliency_scores: dict[str, torch.Tensor],
    n_probe: int,
    rng: torch.Generator,
) -> _PruningProbe:
    """Sample random flat indices and record current float32 values for each scored param.

    Call this BEFORE applying any pruning mask.  The returned snapshot is passed
    to the validators below after pruning to verify mask correctness.
    """
    probe: _PruningProbe = {}
    for name, param in model.named_parameters():
        if name not in saliency_scores:
            continue
        n = param.data.numel()
        k = min(n_probe, n)
        idx = torch.randint(0, n, (k,), generator=rng)
        vals = param.data.detach().float().cpu().flatten()[idx].clone()
        probe[name] = (idx, vals)
    return probe


def assert_kept_weights_unchanged(
    model: AutoModelForCausalLM,
    saliency_scores: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    pre_probe: _PruningProbe,
) -> None:
    """Assert that every probed index where mask=1 still holds its pre-pruning value."""
    for name, param in model.named_parameters():
        if name not in saliency_scores:
            continue
        idx, pre_vals = pre_probe[name]
        mask_flat = masks[name].flatten().cpu()
        kept_at_probe = mask_flat[idx]
        if not kept_at_probe.any():
            continue
        post_vals = param.data.detach().float().cpu().flatten()[idx]
        assert torch.allclose(post_vals[kept_at_probe], pre_vals[kept_at_probe]), (
            f"assert_kept_weights_unchanged: kept weights changed for '{name}' "
            f"at probe indices {idx[kept_at_probe].tolist()}"
        )


def assert_pruned_weights_are_zero(
    model: AutoModelForCausalLM,
    saliency_scores: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> None:
    """Assert that every weight where mask=0 is exactly zero after pruning."""
    for name, param in model.named_parameters():
        if name not in saliency_scores:
            continue
        pruned_vals = param.data[~masks[name]]
        n_nonzero = int(pruned_vals.count_nonzero().item())
        assert n_nonzero == 0, (
            f"assert_pruned_weights_are_zero: {n_nonzero} pruned weights in '{name}' "
            f"are non-zero after masking"
        )


def assert_zero_count_geq_target(
    model: AutoModelForCausalLM,
    saliency_scores: dict[str, torch.Tensor],
    n_prune: int,
    sparsity_fraction: float,
) -> None:
    """Assert the total zero count across all scored params is >= n_prune."""
    total_scored = sum(s.numel() for s in saliency_scores.values())
    total_zeros = sum(
        int((param.data == 0).sum().item())
        for name, param in model.named_parameters()
        if name in saliency_scores
    )
    assert total_zeros >= n_prune, (
        f"assert_zero_count_geq_target: only {total_zeros:,} zeros across scored "
        f"params, expected >= {n_prune:,} "
        f"(target sparsity {sparsity_fraction:.2%} of {total_scored:,} params)"
    )


# ---------------------------------------------------------------------------
# Pruning entry point
# ---------------------------------------------------------------------------


def apply_pruning(
    model: AutoModelForCausalLM,
    saliency_scores: dict[str, torch.Tensor],
    sparsity_fraction: float,
    seed: int = 0,
) -> int:
    """Zero out the lowest-saliency fraction of weights in-place.

    Returns the number of weights actually zeroed.  Three validators are run
    after every pruning pass (see assert_* functions above).
    """
    if sparsity_fraction <= 0.0:
        return 0

    all_scores = torch.cat([s.flatten().cpu() for s in saliency_scores.values()])
    n_prune = max(1, int(sparsity_fraction * all_scores.numel()))
    threshold = torch.kthvalue(all_scores, n_prune).values.item()

    rng = torch.Generator()
    rng.manual_seed(seed)
    pre_probe = sample_pruning_probes(model, saliency_scores, _PRUNING_PROBE_N, rng)

    n_zeroed = 0
    masks: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if name not in saliency_scores:
            continue
        mask = saliency_scores[name] > threshold
        masks[name] = mask
        n_zeroed += int((~mask).sum().item())
        param.data.mul_(mask.to(dtype=param.dtype, device=param.device))

    assert_kept_weights_unchanged(model, saliency_scores, masks, pre_probe)
    assert_pruned_weights_are_zero(model, saliency_scores, masks)
    assert_zero_count_geq_target(model, saliency_scores, n_prune, sparsity_fraction)

    return n_zeroed


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_validation_loss(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    batch_size: int,
    max_seq_len: int,
) -> float:
    """Mean cross-entropy loss over all provided texts."""
    model.eval()
    tokenizer.padding_side = "right"
    total_loss = 0.0
    n_batches = 0
    for i in tqdm(range(0, len(texts), batch_size), desc="  loss batches", leave=False):
        batch = texts[i : i + batch_size]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = tokenized["input_ids"].to(model.device)
        attention_mask = tokenized["attention_mask"].to(model.device)
        labels = input_ids.clone()
        # Mask padding positions by attention_mask, NOT by pad_token_id.
        # For Gemma (and similar models) pad_token == eos_token, so masking
        # by token ID would also suppress real EOS tokens inside the sequence.
        labels[attention_mask == 0] = -100
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += outputs.loss.item()
        n_batches += 1
    return total_loss / n_batches if n_batches > 0 else float("nan")


def run_generation_and_grade(
    generator: HFGenerator,
    tokenizer: PreTrainedTokenizerBase,
    conversations: list[list[dict]],
    batch_size: int,
    max_new_tokens: int,
) -> tuple[GradedChats, list[list[dict]]]:
    """Generate responses then grade with LLM judges.

    Returns a tuple of (GradedChats, completed_conversations).  The caller is
    responsible for persisting the conversations if desired.
    """
    tokenizer.padding_side = "left"
    generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
    completed = generator.generate(
        conversations, batch_size=batch_size, generation_kwargs=generation_kwargs
    )
    return grade_chats(completed), completed


def save_generations(
    completed: list[list[dict]],
    output_dir: Path,
    sparsity: float,
) -> None:
    """Write completed conversations for one sparsity level to disk as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"generations_sparsity_{sparsity:.4f}.json"
    out_path.write_text(json.dumps(completed, indent=2), encoding="utf-8")
    print(f"  Saved {len(completed)} generations → {out_path}")


# ---------------------------------------------------------------------------
# Sparsity level builder
# ---------------------------------------------------------------------------


def build_sparsity_levels(precision: float, sparsity_levels_str: str | None) -> list[float]:
    """Return sorted sparsity fractions from explicit CSV or auto-generated grid."""
    if sparsity_levels_str:
        return sorted(float(s.strip()) for s in sparsity_levels_str.split(","))
    n_steps = round(1.0 / precision)
    return [round(i / n_steps, 10) for i in range(n_steps + 1)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("run")
@click.option(
    "--saliency-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to .safetensors saliency map (output of gradients_map.py).",
)
@click.option(
    "--saliency-type",
    type=click.Choice(["gradient", "taylor"]),
    default="gradient",
    show_default=True,
    help="gradient: |grad|.  taylor: |grad * weight|.",
)
@click.option("--model-id", type=str, default=_DEFAULT_MODEL_ID, show_default=True)
@click.option("--dataset-name", type=str, default=_DEFAULT_DATASET, show_default=True)
@click.option("--dataset-subset", type=str, default=_DEFAULT_SUBSET, show_default=True)
@click.option(
    "--n-samples",
    type=int,
    default=_DEFAULT_N_SAMPLES,
    show_default=True,
    help="Samples for loss evaluation. Actual count = (n_samples // batch_size) * batch_size.",
)
@click.option("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE, show_default=True)
@click.option(
    "--n-generation-samples",
    type=int,
    default=_DEFAULT_N_GENERATION_SAMPLES,
    show_default=True,
    help="Questions to generate + grade per sparsity level.",
)
@click.option("--max-seq-len", type=int, default=_DEFAULT_MAX_SEQ, show_default=True)
@click.option("--max-new-tokens", type=int, default=_DEFAULT_MAX_NEW_TOKENS, show_default=True)
@click.option(
    "--precision",
    type=float,
    default=_DEFAULT_PRECISION,
    show_default=True,
    help="Step size for the auto-generated sparsity grid (e.g. 0.05 → 21 levels: 0%, 5%, …, 100%).",
)
@click.option(
    "--sparsity-levels",
    type=str,
    default=None,
    help="Comma-separated explicit sparsity fractions. Overrides --precision when set.",
)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--wandb-project", type=str, default=_DEFAULT_WANDB_PROJECT, show_default=True)
@click.option(
    "--wandb-run-name",
    type=str,
    default=None,
    help="WandB run name. Auto-generated from saliency type + path stem if not set.",
)
@click.option(
    "--no-generation",
    is_flag=True,
    default=False,
    help="Skip generation + grading (loss only). Useful for quick loss-only sweeps.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=_DEFAULT_OUTPUT_DIR,
    show_default=True,
    help=(
        "Directory to write per-sparsity-level generation JSON files. "
        "Each file is named generations_sparsity_{s:.4f}.json. "
        "Ignored when --no-generation is set."
    ),
)
@click.option("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
def run_single(
    saliency_path: Path,
    saliency_type: str,
    model_id: str,
    dataset_name: str,
    dataset_subset: str,
    n_samples: int,
    batch_size: int,
    n_generation_samples: int,
    max_seq_len: int,
    max_new_tokens: int,
    precision: float,
    sparsity_levels: str | None,
    seed: int,
    wandb_project: str,
    wandb_run_name: str | None,
    no_generation: bool,
    output_dir: Path,
    device: str,
) -> None:
    """Sweep pruning sparsity levels and log loss + generation quality to wandb."""
    parsed_levels = build_sparsity_levels(precision, sparsity_levels)
    n_loss_samples = (n_samples // batch_size) * batch_size
    run_name = wandb_run_name or f"{saliency_type}_{saliency_path.stem}_{dataset_subset}"
    gen_output_dir = output_dir / run_name

    print(f"Sparsity levels ({len(parsed_levels)}): {parsed_levels}")
    print(f"Loss samples: {n_loss_samples}  ({n_loss_samples // batch_size} batches of {batch_size})")
    print(f"Generation samples: {n_generation_samples}")
    if not no_generation:
        print(f"Generation output dir: {gen_output_dir}")

    # Load saliency tensors
    saliency_tensors = load_file(str(saliency_path))
    print(f"Loaded saliency map: {len(saliency_tensors)} tensors from {saliency_path}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if _CHAT_TEMPLATE_PATH.exists():
        tokenizer.chat_template = _CHAT_TEMPLATE_PATH.read_text()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load validation dataset (enough rows for both loss and generation)
    n_total = max(n_loss_samples, n_generation_samples)
    val_dataset = _load_val_dataset(dataset_name, dataset_subset, n=n_total, seed=seed)

    loss_texts = _format_texts_for_loss(
        val_dataset.select(range(min(n_loss_samples, len(val_dataset)))), tokenizer
    )
    gen_conversations = _format_conversations_for_generation(
        val_dataset.select(range(min(n_generation_samples, len(val_dataset))))
    )

    # Compute saliency scores once from original (unpruned) weights.
    # For Taylor this must happen before any pruning.
    print(f"Computing saliency scores ({saliency_type})...")
    saliency_scores = compute_saliency_scores(model, saliency_tensors, saliency_type)
    total_scored_params = sum(s.numel() for s in saliency_scores.values())
    print(f"  Scored {total_scored_params:,} parameters across {len(saliency_scores)} tensors.")

    # Cache original weights on CPU so we can restore between levels.
    print("Caching original weights on CPU (this allocates ~model-size CPU RAM)...")
    original_weights = save_original_weights(model)

    # Init wandb
    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "saliency_path": str(saliency_path),
            "saliency_type": saliency_type,
            "model_id": model_id,
            "dataset": f"{dataset_name}/{dataset_subset}",
            "n_loss_samples": n_loss_samples,
            "batch_size": batch_size,
            "n_generation_samples": n_generation_samples,
            "precision": precision,
            "sparsity_levels": parsed_levels,
            "no_generation": no_generation,
            "output_dir": str(gen_output_dir),
        },
    )
    # Use sparsity as the x-axis for all metrics so wandb plots are labelled
    # with the actual fraction rather than the auto-incremented step counter.
    wandb.define_metric("sparsity")
    wandb.define_metric("actual_sparsity", step_metric="sparsity")
    wandb.define_metric("val_loss", step_metric="sparsity")
    wandb.define_metric("generation/*", step_metric="sparsity")

    for sparsity in tqdm(parsed_levels, desc="Sparsity sweep"):
        print(f"\n=== Sparsity {sparsity:.1%} ===")

        restore_original_weights(model, original_weights)
        n_zeroed = apply_pruning(model, saliency_scores, sparsity)
        actual_sparsity = n_zeroed / total_scored_params
        print(f"  Zeroed {n_zeroed:,} / {total_scored_params:,} weights "
              f"(actual {actual_sparsity:.2%}, target {sparsity:.2%})")

        val_loss = compute_validation_loss(model, tokenizer, loss_texts, batch_size, max_seq_len)
        print(f"  val_loss: {val_loss:.4f}")

        log_dict: dict[str, float] = {
            "sparsity": sparsity,
            "actual_sparsity": actual_sparsity,
            "val_loss": val_loss,
        }

        if not no_generation:
            # A fresh HFGenerator is created for every sparsity level so that
            # no cached responses from a previous (less-pruned) model are
            # accidentally served for the current pruned model.
            generator = HFGenerator(model, tokenizer)
            graded, completed = run_generation_and_grade(
                generator, tokenizer, gen_conversations, batch_size, max_new_tokens
            )
            save_generations(completed, gen_output_dir, sparsity)
            print(f"  generation overall score: {graded.overall_mean_score:.4f}")
            log_dict["generation/overall_mean_score"] = graded.overall_mean_score
            for judge_name, mean_score in graded.judge_name2mean_scores.items():
                log_dict[f"generation/{judge_name}"] = mean_score

        wandb.log(log_dict)

    restore_original_weights(model, original_weights)
    wandb.finish()
    print("\nDone. Model weights restored to original state.")


# ---------------------------------------------------------------------------
# Batch command helpers
# ---------------------------------------------------------------------------

_SALIENCY_TYPES: tuple[str, ...] = ("gradient", "taylor")


def _build_sweep_cmd(
    saliency_path: Path,
    saliency_type: str,
    run_name: str,
    run_output_dir: Path,
    common_kwargs: dict,
) -> list[str]:
    """Build the subprocess argv for `python sweep_eval_temp.py run ...`."""
    cmd: list[str] = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run",
        "--saliency-path", str(saliency_path),
        "--saliency-type", saliency_type,
        "--wandb-run-name", run_name,
        "--output-dir", str(run_output_dir),
        "--model-id",           common_kwargs["model_id"],
        "--dataset-name",       common_kwargs["dataset_name"],
        "--dataset-subset",     common_kwargs["dataset_subset"],
        "--n-samples",          str(common_kwargs["n_samples"]),
        "--batch-size",         str(common_kwargs["batch_size"]),
        "--n-generation-samples", str(common_kwargs["n_generation_samples"]),
        "--max-seq-len",        str(common_kwargs["max_seq_len"]),
        "--max-new-tokens",     str(common_kwargs["max_new_tokens"]),
        "--precision",          str(common_kwargs["precision"]),
        "--seed",               str(common_kwargs["seed"]),
        "--wandb-project",      common_kwargs["wandb_project"],
        "--device", "cuda",
    ]
    if common_kwargs.get("no_generation"):
        cmd.append("--no-generation")
    if common_kwargs.get("sparsity_levels"):
        cmd.extend(["--sparsity-levels", common_kwargs["sparsity_levels"]])
    return cmd


def _is_run_complete(run_output_dir: Path) -> bool:
    """Return True if the run output directory exists and contains at least one JSON file."""
    return run_output_dir.exists() and any(run_output_dir.glob("*.json"))


def _sweep_worker(
    run_name: str,
    cmd: list[str],
    device_id: str,
    device_queue: "queue.Queue[str]",
) -> int:
    """Run one sweep subprocess on the given CUDA device, then return the device."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = device_id
    print(f"[batch] Starting sweep '{run_name}' on CUDA_VISIBLE_DEVICES={device_id}")
    result = subprocess.run(cmd, env=env)
    rc = result.returncode
    status = "✅ done" if rc == 0 else f"❌ exit {rc}"
    print(f"[batch] Sweep '{run_name}' on device {device_id}: {status}")
    device_queue.put(device_id)
    return rc


@click.command("batch")
@click.option(
    "--saliency-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("./biology"),
    show_default=True,
    help="Directory containing .safetensors saliency files. "
         "Every file is swept against both 'gradient' and 'taylor' criteria.",
)
@click.option(
    "--devices",
    type=str,
    default="0",
    show_default=True,
    help="Comma-separated CUDA device IDs (e.g. '0,1,2,3'). "
         "Runs are distributed across devices; each device handles one run at a time.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-run even if the output directory for a run already contains results. "
         "By default, completed runs (non-empty output dir) are skipped.",
)
@click.option(
    "--output-dir-base",
    type=click.Path(path_type=Path),
    default=_DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Base directory for per-run generation outputs. "
         "Each run writes to <output-dir-base>/<run_name>/.",
)
@click.option("--model-id", type=str, default=_DEFAULT_MODEL_ID, show_default=True)
@click.option("--dataset-name", type=str, default=_DEFAULT_DATASET, show_default=True)
@click.option("--dataset-subset", type=str, default=_DEFAULT_SUBSET, show_default=True)
@click.option("--n-samples", type=int, default=_DEFAULT_N_SAMPLES, show_default=True)
@click.option("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE, show_default=True)
@click.option(
    "--n-generation-samples",
    type=int,
    default=_DEFAULT_N_GENERATION_SAMPLES,
    show_default=True,
)
@click.option("--max-seq-len", type=int, default=_DEFAULT_MAX_SEQ, show_default=True)
@click.option("--max-new-tokens", type=int, default=_DEFAULT_MAX_NEW_TOKENS, show_default=True)
@click.option("--precision", type=float, default=_DEFAULT_PRECISION, show_default=True)
@click.option("--sparsity-levels", type=str, default=None)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--wandb-project", type=str, default=_DEFAULT_WANDB_PROJECT, show_default=True)
@click.option(
    "--no-generation",
    is_flag=True,
    default=False,
    help="Pass --no-generation to every child run (loss only, no LLM grading).",
)
def run_batch(
    saliency_dir: Path,
    devices: str,
    force: bool,
    output_dir_base: Path,
    model_id: str,
    dataset_name: str,
    dataset_subset: str,
    n_samples: int,
    batch_size: int,
    n_generation_samples: int,
    max_seq_len: int,
    max_new_tokens: int,
    precision: float,
    sparsity_levels: str | None,
    seed: int,
    wandb_project: str,
    no_generation: bool,
) -> None:
    """Run all saliency-file × criterion combinations in parallel across CUDA devices.

    Discovers every .safetensors file in --saliency-dir and sweeps each one
    against both 'gradient' and 'taylor' criteria.  Runs are distributed
    round-robin across the supplied --devices.  Each run is a subprocess of
    `python sweep_eval_temp.py run`.

    Runs whose output directory already contains JSON files are skipped unless
    --force is set.

    Example — sweep all files in biology/ across four GPUs:

        python sweep_eval_temp.py batch --saliency-dir biology/ --devices 0,1,2,3

    Example — loss-only pass, force-rerun everything:

        python sweep_eval_temp.py batch --saliency-dir biology/ --devices 0 \\
            --no-generation --force
    """
    saliency_files = sorted(saliency_dir.glob("*.safetensors"))
    if not saliency_files:
        raise click.UsageError(f"No .safetensors files found in {saliency_dir}")

    device_list = [d.strip() for d in devices.split(",") if d.strip()]
    if not device_list:
        raise click.BadParameter("Must specify at least one device.", param_hint="--devices")

    common_kwargs = dict(
        model_id=model_id,
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        n_samples=n_samples,
        batch_size=batch_size,
        n_generation_samples=n_generation_samples,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        precision=precision,
        sparsity_levels=sparsity_levels,
        seed=seed,
        wandb_project=wandb_project,
        no_generation=no_generation,
    )

    to_run: list[tuple[str, list[str]]] = []
    for sf in saliency_files:
        for stype in _SALIENCY_TYPES:
            run_name = f"{sf.stem}_{stype}"
            run_output_dir = output_dir_base / run_name
            if not force and _is_run_complete(run_output_dir):
                print(f"[batch] Skipping '{run_name}': {run_output_dir} already has results "
                      f"(use --force to rerun).")
                continue
            cmd = _build_sweep_cmd(sf, stype, run_name, run_output_dir, common_kwargs)
            to_run.append((run_name, cmd))

    if not to_run:
        print("[batch] Nothing to run.")
        return

    print(f"[batch] Running {len(to_run)} sweep(s) across {len(device_list)} device(s):")
    for run_name, _ in to_run:
        print(f"  - {run_name}")

    device_q: queue.Queue[str] = queue.Queue()
    for d in device_list:
        device_q.put(d)

    exit_codes: dict[str, int] = {}
    lock = threading.Lock()

    def _worker(run_name: str, cmd: list[str]) -> None:
        device_id = device_q.get()
        rc = _sweep_worker(run_name, cmd, device_id, device_q)
        with lock:
            exit_codes[run_name] = rc

    threads = [
        threading.Thread(target=_worker, args=(rn, cmd), daemon=True)
        for rn, cmd in to_run
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = [rn for rn, rc in exit_codes.items() if rc != 0]
    if failed:
        raise SystemExit(f"[batch] The following sweeps failed: {failed}")
    print("[batch] All sweeps completed successfully.")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Pruning sparsity sweep — evaluate loss and generation quality at each level."""


main.add_command(run_single)
main.add_command(run_batch)


if __name__ == "__main__":
    main()
