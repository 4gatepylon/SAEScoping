from __future__ import annotations

from beartype import beartype
import gc
import hashlib
import json
import os
import re
import click
import torch
from pathlib import Path
from sae_scoping.utils.hooks.sae import SAELensEncDecCallbackWrapper
from datasets import load_dataset
from sae_lens import SAE
from safetensors.torch import load_file
from transformers import AutoTokenizer, Gemma2ForCausalLM
from trl import SFTConfig

from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae as _get_pruned_sae
from sae_scoping.trainers.sae_enhanced.train import train_sae_enhanced_model

"""
Barebones script to train SAE-enhanced (or vanilla) Gemma-2-9b-it on Numina Math AIMO.
"""

GEMMA2_9B_SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
MODEL_NAME = "google/gemma-2-9b-it"


@beartype
def sae_id_from_path(dist_path: str) -> str:
    """Extract SAE ID from path like '.../layer_20--width_16k--canonical/distribution.safetensors'."""
    folder_name = Path(dist_path).parent.name
    return folder_name.replace("--", "/")


@beartype
def sae_id2hookpoint(sae_id: str) -> str:
    assert re.match(r"^layer_\d+/width_\d+k/canonical$", sae_id), f"Invalid SAE ID: {sae_id}"
    layer_num = int(sae_id.split("/", 1)[0].split("_")[1])
    return f"model.layers.{layer_num}"


@beartype
def get_pruned_sae(dist_path: str, threshold: float, device: torch.device | str = "cpu") -> tuple[SAELensEncDecCallbackWrapper, str, int]:
    sae_id = sae_id_from_path(dist_path)
    dist_data = load_file(dist_path)
    distribution = dist_data["distribution"]
    neuron_ranking = torch.argsort(distribution, descending=True)
    n_kept = int((distribution >= threshold).sum().item())  # NOTE: this must be >= to include all neurons in 0 case
    print(f"Keeping {n_kept}/{len(distribution)} neurons (threshold={threshold})")
    sae = SAE.from_pretrained(release=GEMMA2_9B_SAE_RELEASE, sae_id=sae_id, device=device).to(device)
    return (
        _get_pruned_sae(sae, neuron_ranking, K_or_p=n_kept, T=0.0).to(device),
        sae_id2hookpoint(sae_id),
        n_kept,
    )


@beartype
def get_run_identifiers(
    vanilla: bool,
    model_path: str | None,
    hookpoint: str | None,
    dist_path: str | None,
    threshold: float,
    n_kept: int | None,
) -> tuple[str, str, dict]:
    """Generate unique folder hash and wandb run name from arguments.

    Returns:
        (folder_hash, wandb_run_name, args_dict)
    """
    args_dict = {
        "vanilla": vanilla,
        "model_path": model_path,
        "hookpoint": hookpoint,
        "dist_path": dist_path,
        "threshold": threshold,
        "n_kept": n_kept,
    }

    # Hash all args for unique folder name
    args_str = json.dumps(args_dict, sort_keys=True)
    folder_hash = hashlib.sha256(args_str.encode()).hexdigest()  # For uniqueness use the entire hash

    # Build wandb run name
    # Model part: first 10 chars if not default, else "gemma2-9b-it"
    if model_path is None or model_path == MODEL_NAME:
        model_part = "gemma2-9b-it"
    else:
        model_part = hashlib.sha256(model_path.encode()).hexdigest()[:10]

    if vanilla:
        mode_part = "vanilla"
    else:
        mode_part = f"h{threshold}"

    wandb_run_name = f"{model_part}_{mode_part}"
    if not vanilla:
        # N neurons identifier
        wandb_run_name += f"_n{n_kept}"

        # Dist path identifier
        dist_hash = hashlib.sha256(dist_path.encode()).hexdigest()[:10]
        wandb_run_name += f"_D{dist_hash}"

        # Hookpoint integer
        hookpoint_int = int(hookpoint.split(".")[-1])
        wandb_run_name += f"_H{hookpoint_int}"

        # Threshold value => Ignore
        # wandb_run_name += f"_T{threshold}" # Included in mode part

    return folder_hash, wandb_run_name, args_dict


def load_aimo_dataset(tokenizer, n_train: int = 10_000, n_eval: int = 500):
    """Load and format Numina Math AIMO dataset."""
    ds = load_dataset("AI-MO/NuminaMath-1.5", split="train").shuffle(seed=42)
    ds = ds.select(range(min(n_train + n_eval, len(ds))))

    def format_example(x):
        return {
            "text": tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": x["problem"]},
                    {"role": "assistant", "content": x["solution"]},
                ],
                tokenize=False,
            )
        }

    ds = ds.map(format_example, remove_columns=ds.column_names)
    return ds.train_test_split(test_size=n_eval, seed=42)


OUTPUT_ROOT = Path("./outputs_aimo_math")


@click.command()
@click.option(
    "--model-path",
    "-m",
    type=str,
    default=None,
    help="Checkpoint path (default: google/gemma-2-9b-it)",
)
@click.option(
    "--dist-path",
    "-p",
    type=str,
    default=None,
    help="Path to distribution.safetensors (required if not vanilla)",
)
@click.option("--threshold", "-t", type=float, default=1e-4, help="Min firing rate to keep neuron")
@click.option("--wandb-project-name", "-w", type=str, default="aimo-math-adversarial-tuning", help="WandB project name")
def main(
    model_path: str | None,
    dist_path: str | None,
    threshold: float,
    wandb_project_name: str,
) -> None:
    """Train SAE-enhanced or vanilla model on Numina Math AIMO."""
    os.environ["GRADIENT_CHECKPOINTING"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate args
    vanilla = dist_path is None
    # NOTE vanilla => no SAE, but could different checkpoint
    if not vanilla and not Path(dist_path).exists():
        raise click.UsageError(f"Distribution file not found: {dist_path}")

    # Load tokenizer and model
    model_name_or_path = model_path if model_path else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = Gemma2ForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,  # h100 | a100 -> this has some great benefits in speed and space
        device_map="cpu",
        attn_implementation="eager",  # needed for unknown reasons
    ).to(device)
    model.gradient_checkpointing_disable()

    # Load SAE if not vanilla
    pruned_sae, hookpoint, n_kept = None, None, None
    if not vanilla:
        pruned_sae, hookpoint, n_kept = get_pruned_sae(dist_path, threshold, device)

    # Load dataset
    # TODO(Adriano) note that `answer` is often "proof" which may break the verifiability...
    # we will want to switch to another dataset in the near future (maybe gsm8k or smth simple tbh)
    dataset = load_aimo_dataset(tokenizer)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Generate unique identifiers for output folder and wandb
    folder_hash, wandb_run_name, args_dict = get_run_identifiers(
        vanilla=vanilla,
        model_path=model_path,
        hookpoint=hookpoint,
        dist_path=dist_path,
        threshold=threshold,
        n_kept=n_kept,
    )

    # Create output directory and save args
    output_dir = OUTPUT_ROOT / folder_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    args_file = output_dir / "args.json"
    args_file.write_text(json.dumps(args_dict, indent=2))
    print(f"Output directory: {output_dir}")
    print(f"WandB run name: {wandb_run_name}")

    # SFT config with hardcoded defaults
    sft_config = SFTConfig(
        # TODO(Adriano) do not hardcode these values
        output_dir=str(output_dir),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        max_steps=4000,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.1,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=1000,
        save_total_limit=2,
        bf16=True,
        max_length=1024,
        gradient_checkpointing=False,
        report_to="wandb",
        run_name=wandb_run_name,
        # NOTE: train only on the assitant tokens:
        assistant_only_loss=True,
    )

    # Train
    os.environ["WANDB_PROJECT"] = wandb_project_name
    train_sae_enhanced_model(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        sae=pruned_sae,
        model=model,
        tokenizer=tokenizer,
        T=0.0,
        hookpoint=hookpoint if not vanilla else None,
        save_output=False,
        sft_config=sft_config,
        wandb_project_name=wandb_project_name,
        wandb_run_name=wandb_run_name,
    )

    # Cleanup
    del model
    if pruned_sae:
        del pruned_sae
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    """
    To run with our specific dist-path and custom model please run with this command:
    ```
    python3 script_2026_01_22_math_adversarial_tuning.py \
        --dist-path /mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors \
        --model-path /mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b/ultrachat/layer_31_width_16k_canonical_h0.0001_85cac49528/checkpoint-2000
    ```

    but to run with Gemma-2-9b-it vanilla please run with this command:
    ```
    python3 script_2026_01_22_math_adversarial_tuning.py
    ```
    """
    main()
