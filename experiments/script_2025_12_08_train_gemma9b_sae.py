from __future__ import annotations

import gc
import hashlib
import re
from pathlib import Path

import click
import torch
from beartype import beartype
from datasets import concatenate_datasets
from sae_lens import SAE
from safetensors.torch import load_file
from transformers import AutoTokenizer, Gemma2ForCausalLM, PreTrainedTokenizerBase
from trl import SFTConfig

from sae_scoping.datasets.messages_datasets import (
    get_imdb_sentiment_dataset_for_gemma_it,
)
from sae_scoping.datasets.text_datasets import (
    get_camel_ai_biology_dataset,
    get_megascience_biology_dataset,
    load_apps,
    load_ultrachat_dataset,
)
from sae_scoping.training.sae_enhanced.pruning import (
    get_pruned_sae,
)
from sae_scoping.training.sae_enhanced.sae_aware_sft import (
    train_sae_enhanced_model,
)

GEMMA2_9B_SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"


@beartype
def sae_id_from_path(dist_path: str) -> str:
    """Extract SAE ID from path like '.../layer_20--width_16k--canonical/distribution.safetensors'."""
    folder_name = Path(dist_path).parent.name
    return folder_name.replace("--", "/")


@beartype
def sae_id2hookpoint(sae_id: str) -> str:
    assert re.match(r"^layer_\d+/width_\d+k/canonical$", sae_id), (
        f"Invalid SAE ID: {sae_id}"
    )
    layer_num = int(sae_id.split("/", 1)[0].split("_")[1])
    return f"model.layers.{layer_num}"


@click.command()
@click.option(
    "--dist-path",
    "-p",
    type=str,
    required=True,
    help="Path to distribution.safetensors",
)
@click.option("--batch-size", "-b", type=int, default=4, help="Training batch size")
@click.option(
    "--threshold",
    "-h",
    type=float,
    default=1e-5,  # seems eh ok from s curve? lol
    help="Min firing rate to keep neuron",
)
@click.option("--max-steps", "-s", type=int, default=1000, help="Max training steps")
@click.option("--accum", "-a", type=int, default=1, help="Gradient accumulation steps")
@click.option(
    "--special-hookpoint",
    "-hook",
    type=str,
    default=None,
    help="Special hookpoint to use",
)
@click.option("--checkpoint", "-c", type=str, default=None, help="Checkpoint to load")
@click.option(
    "--train-on-dataset", "-t", type=str, default="biology", help="Dataset to train on"
)
@click.option(
    "--wandb-project-name",
    "-w",
    type=str,
    default="gemma-scope-9b-recovery-train",
    help="Wandb project name",
)
@click.option("--save-every", "-se", type=int, default=1000, help="Save every n steps")
@click.option("--save-limit", "-sl", type=int, default=10, help="Save limit")
# NOTE please run for gemma
# export GRADIENT_CHECKPOINTING=0
def main(
    dist_path: str,
    batch_size: int,
    threshold: float,
    max_steps: int,
    accum: int,
    special_hookpoint: str | None,
    checkpoint: str | None,
    train_on_dataset: str,
    wandb_project_name: str,
    save_every: int,
    save_limit: int,
) -> None:
    r"""
    Example with benign recovery training in-domain:
    ```
    python3 script_2025_12_08_train_gemma9b_sae.py \
        -p vanilla \
        -b 2 -a 16 -hook model.layers.20 -s 40000
    ```

    Example adversarial re-training (after recovery training) example:
    ```
    python3 script_2025_12_08_train_gemma9b_sae.py \
        -c outputs_gemma9b/biology/layer_31_width_16k_canonical_h0.0001/checkpoint-3000 \
        -t ultrachat \
        -w gemma-scope-9b-recovery-attack-2025-12-09 \
        -s 4000 -a 8 -b 4 -h 0.0001 \
        -p deleteme_cache_bio_only/ignore_padding_True/biology/layer_20--width_16k--canonical/distribution.safetensors
    ```
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Extract SAE ID and hookpoint from path
    if dist_path == "vanilla":
        sae_id, hookpoint, sae, pruned_sae = None, None, None, None
    else:
        sae_id = sae_id_from_path(dist_path)
        hookpoint = sae_id2hookpoint(sae_id)
        print(f"SAE ID: {sae_id}, Hookpoint: {hookpoint}")

        # 2. Load distribution and compute neuron mask
        dist_data = load_file(dist_path)
        distribution: torch.Tensor = dist_data["distribution"]  # shape: (d_sae,)
        neuron_ranking = torch.argsort(distribution, descending=True)
        n_kept = int((distribution >= threshold).sum().item())
        print(f"Keeping {n_kept}/{len(distribution)} neurons (threshold={threshold})")

    # 3. Load tokenizer and model
    model_name = "google/gemma-2-9b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    model_name_or_path = checkpoint if checkpoint is not None else model_name
    model_name_or_path_hash = (
        hashlib.sha256(model_name_or_path.encode()).hexdigest()
        if model_name_or_path != "vanilla"
        else "vanilla"
    )
    model = Gemma2ForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )
    model = model.to(device)
    model.gradient_checkpointing_disable()
    if hasattr(model, "model"):
        model.model.gradient_checkpointing = False

    if sae_id is not None:
        # 4. Load SAE and create pruned version
        sae = SAE.from_pretrained(
            release=GEMMA2_9B_SAE_RELEASE, sae_id=sae_id, device=device
        )
        sae = sae.to(device)
        pruned_sae = get_pruned_sae(sae, neuron_ranking, K_or_p=n_kept, T=0.0)
        pruned_sae = pruned_sae.to(device)

    # 5. Build training and evaluation datasets
    # TODO(Adriano) support this in "messages" format instead
    camel_dd = get_camel_ai_biology_dataset(
        n_samples_ranking=1,
        n_samples_training=18_000,
        n_samples_evaluation=1,
        seed=1,
        verbose=True,
        qa_templatting_function=tokenizer,
    )
    mega_dd = get_megascience_biology_dataset(
        n_samples_ranking=1,
        n_samples_training=32_000,
        n_samples_evaluation=1,
        seed=1,
        verbose=True,
        qa_templatting_function=tokenizer,
    )
    biology_dataset = concatenate_datasets(
        [
            camel_dd["training"].remove_columns(
                [c for c in camel_dd["training"].column_names if c != "text"]
            ),
            mega_dd["training"].remove_columns(
                [c for c in mega_dd["training"].column_names if c != "text"]
            ),
        ]
    )

    # 6. Build eval datasets: 500 each for ultrachat and apps
    ultrachat_dd = load_ultrachat_dataset(
        n_samples_ranking=1,
        n_samples_training=1,
        n_samples_evaluation=40_000,
        seed=1,
        verbose=True,
        tokenizer=tokenizer,
    )
    ultrachat_dataset = concatenate_datasets(
        [
            ultrachat_dd["ranking"].remove_columns(
                [c for c in ultrachat_dd["ranking"].column_names if c != "text"]
            ),
            ultrachat_dd["training"].remove_columns(
                [c for c in ultrachat_dd["training"].column_names if c != "text"]
            ),
            ultrachat_dd["evaluation"].remove_columns(
                [c for c in ultrachat_dd["evaluation"].column_names if c != "text"]
            ),
        ]
    )
    apps_dd = load_apps(
        n_samples_ranking=1,
        n_samples_training=1,
        n_samples_evaluation=8_700,
        seed=1,
        verbose=True,
        # Select all the difficulties to get more data
        difficulties=["introductory", "competition", "interview"],
    )
    apps_dataset = concatenate_datasets(
        [
            apps_dd["ranking"].remove_columns(
                [c for c in apps_dd["ranking"].column_names if c != "text"]
            ),
            apps_dd["training"].remove_columns(
                [c for c in apps_dd["training"].column_names if c != "text"]
            ),
            apps_dd["evaluation"].remove_columns(
                [c for c in apps_dd["evaluation"].column_names if c != "text"]
            ),
        ]
    )
    imdb_dataset = get_imdb_sentiment_dataset_for_gemma_it(
        n_samples=10_000,
        seed=1,
        verbose=True,
        tokenizer=tokenizer,
        n_shots=0,  # Don't use shots to avoid repeating data for SFT
    )

    biology_train_test = biology_dataset.train_test_split(test_size=500, seed=1)
    ultrachat_train_test = ultrachat_dataset.train_test_split(test_size=500, seed=1)
    apps_train_test = apps_dataset.train_test_split(test_size=500, seed=1)
    imdb_train_test = imdb_dataset.train_test_split(test_size=500, seed=1)

    # Eval on all of them
    eval_datasets = {
        "biology": biology_train_test["test"],
        "ultrachat": ultrachat_train_test["test"],
        "apps": apps_train_test["test"],
        "imdb": imdb_train_test["test"],
    }
    train_datasets = {
        "biology": biology_train_test["train"],
        "ultrachat": ultrachat_train_test["train"],
        "apps": apps_train_test["train"],
        "imdb": imdb_train_test["train"],
    }
    if train_on_dataset not in train_datasets:
        raise ValueError(f"Invalid train on dataset: {train_on_dataset}")
    train_dataset = train_datasets[train_on_dataset]

    # 7. Train
    # TODO(Adriano) the output directory should NOT be hardcoded, and a config file
    # should be stored next to the checkpoints to document EXACTLY what parameters went
    # into this (which SAE, etc...) instead of forcing us to rely on a path/name that
    # could change.
    # TODO(Adriano) you should add callbacks to control when training stops, etc...
    # (look at: https://claude.ai/share/67efb914-0c33-46bf-ab32-df5e9828f6ad)
    if sae_id is None:
        sae_id = "vanilla"
    output_dir = f"./outputs_gemma9b/{train_on_dataset}/{sae_id.replace('/', '_')}"
    if sae_id != "vanilla":
        output_dir += f"_h{threshold}_{model_name_or_path_hash[:10]}"
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=max_steps,
        gradient_accumulation_steps=accum,
        num_train_epochs=1,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.1,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=save_every,
        bf16=True,
        save_total_limit=save_limit,
        report_to="wandb",
        max_length=1024,  # SAE context length bounds this
        gradient_checkpointing=False,
        # RuntimeError: You're using `assistant_only_loss=True`, but at least one
        # example has no assistant tokens. This usually means the tokenizer's chat
        # template doesn't generate assistant masks — it may be missing the
        # `{% generation %}` keyword. Please check the template and ensure it's
        # correctly configured to support assistant masking
        # assistant_only_loss=(
        #     "messages" in train_dataset.column_names
        #     and not "text" in train_dataset.column_names
        # ),
    )
    wandb_run_name = f"{train_on_dataset}/{sae_id.replace('/', '_')}"
    if sae_id != "vanilla":
        wandb_run_name += f"h{threshold}/{model_name_or_path_hash[:10]}"
    if special_hookpoint is not None:  # used to limit # layers trained on
        hookpoint = special_hookpoint
    # NOTE: while technically not supported by my code, since it's passthrough, you
    # SHOULD be able to use not only "text" but also "messages" etc... (looke at
    # SFTTrainer docs for supported formats)
    train_sae_enhanced_model(
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        sae=pruned_sae,
        model=model,
        tokenizer=tokenizer,
        T=0.0,
        hookpoint=hookpoint,
        sft_config=sft_config,
        wandb_project_name=wandb_project_name,
        wandb_run_name=wandb_run_name,
    )

    # Cleanup
    del model, sae, pruned_sae
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
