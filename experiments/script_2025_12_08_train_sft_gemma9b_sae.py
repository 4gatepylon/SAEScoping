from __future__ import annotations

import gc
import hashlib
import re
from pathlib import Path
import os
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
    get_camel_ai_chemistry_dataset,
    get_gsm8k_dataset,
    get_megascience_biology_dataset,
    get_megascience_chemistry_dataset,
    load_apps,
    load_cybermetric_dataset,
    load_ultrachat_dataset,
    get_numina_math_aimo_dataset,
)
from sae_scoping.trainers.sae_enhanced.prune import (
    get_pruned_sae,
)
from sae_scoping.trainers.sae_enhanced.train_sft import (
    train_sae_enhanced_model,
)

"""
This module/script does exactly what you expect: it trains a Gemma-2 9B
around an SAE.
"""

GEMMA2_9B_SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"


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
def model_name_or_path2threshold(model_name_or_path: str | None) -> float:
    """Copied from `script_2026_01_23_evaluate_biology_utility.py` (while this file was first made before that one, it was modified more recently)."""
    if model_name_or_path is None:
        raise ValueError(f"model_name_or_path is None")
    h_find_pattern = r"_h(\d+\.?\d*(?:e[+-]?\d+)?)"
    match = re.search(h_find_pattern, model_name_or_path, re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not extract h value from path: {model_name_or_path}")
    return float(match.group(1))


def _main(
    dist_path: str,
    batch_size: int,
    max_steps: int,
    accum: int,
    special_hookpoint: str | None,
    checkpoint: str | None,
    train_on_dataset: str,
    wandb_project_name: str,
    save_every: int,
    save_limit: int,
    output_dir: str | None = None,
    wandb_run_name: str | None = None,
    save_output: bool = False,
    max_length: int = 1024,
    eval_on_datasets: str | None = None,  # comma-delimited list of dataset names, None = all
    eval_test_size: int = 500,  # number of samples for evaluation per dataset
    threshold: float | None = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Extract SAE ID and hookpoint from path
    if dist_path == "vanilla":
        sae_id, hookpoint, sae, pruned_sae = None, None, None, None
    else:
        sae_id = sae_id_from_path(dist_path)
        hookpoint = sae_id2hookpoint(sae_id)
        threshold = model_name_or_path2threshold(checkpoint) if threshold is None else threshold
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
    model_name_or_path_hash = hashlib.sha256(model_name_or_path.encode()).hexdigest() if model_name_or_path != "vanilla" else "vanilla"
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
        sae = SAE.from_pretrained(release=GEMMA2_9B_SAE_RELEASE, sae_id=sae_id, device=device)
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
            camel_dd["training"].remove_columns([c for c in camel_dd["training"].column_names if c != "text"]),
            mega_dd["training"].remove_columns([c for c in mega_dd["training"].column_names if c != "text"]),
        ]
    )

    # Chemistry dataset (similar to biology)
    camel_chem_dd = get_camel_ai_chemistry_dataset(
        n_samples_ranking=1,
        n_samples_training=18_000,
        n_samples_evaluation=1,
        seed=1,
        verbose=True,
        qa_templatting_function=tokenizer,
    )
    mega_chem_dd = get_megascience_chemistry_dataset(
        n_samples_ranking=1,
        n_samples_training=32_000,
        n_samples_evaluation=1,
        seed=1,
        verbose=True,
        qa_templatting_function=tokenizer,
    )
    chemistry_dataset = concatenate_datasets(
        [
            camel_chem_dd["training"].remove_columns([c for c in camel_chem_dd["training"].column_names if c != "text"]),
            mega_chem_dd["training"].remove_columns([c for c in mega_chem_dd["training"].column_names if c != "text"]),
        ]
    )

    # Cybermetric dataset
    cybermetric_dd = load_cybermetric_dataset(
        n_samples_ranking=1,
        n_samples_training=9_000,
        n_samples_evaluation=1,
        seed=1,
        verbose=True,
        qa_templatting_function=tokenizer,
    )
    cybermetric_dataset = concatenate_datasets(
        [
            cybermetric_dd["ranking"].remove_columns([c for c in cybermetric_dd["ranking"].column_names if c != "text"]),
            cybermetric_dd["training"].remove_columns([c for c in cybermetric_dd["training"].column_names if c != "text"]),
            cybermetric_dd["evaluation"].remove_columns([c for c in cybermetric_dd["evaluation"].column_names if c != "text"]),
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
            ultrachat_dd["ranking"].remove_columns([c for c in ultrachat_dd["ranking"].column_names if c != "text"]),
            ultrachat_dd["training"].remove_columns([c for c in ultrachat_dd["training"].column_names if c != "text"]),
            ultrachat_dd["evaluation"].remove_columns([c for c in ultrachat_dd["evaluation"].column_names if c != "text"]),
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
            apps_dd["ranking"].remove_columns([c for c in apps_dd["ranking"].column_names if c != "text"]),
            apps_dd["training"].remove_columns([c for c in apps_dd["training"].column_names if c != "text"]),
            apps_dd["evaluation"].remove_columns([c for c in apps_dd["evaluation"].column_names if c != "text"]),
        ]
    )
    imdb_dataset = get_imdb_sentiment_dataset_for_gemma_it(
        n_samples=10_000,
        seed=1,
        verbose=True,
        tokenizer=tokenizer,
        n_shots=0,  # Don't use shots to avoid repeating data for SFT
    )

    # NuminaMath dataset for math training
    numinamath_dd = get_numina_math_aimo_dataset(
        n_samples_ranking=1,
        n_samples_training=40_000,
        n_samples_evaluation=1,
        seed=1,
        verbose=True,
        exclude_proofs=True,  # Exclude proof-based problems
        qa_templatting_function=tokenizer,
    )
    numinamath_dataset = concatenate_datasets(
        [
            numinamath_dd["ranking"].remove_columns([c for c in numinamath_dd["ranking"].column_names if c != "text"]),
            numinamath_dd["training"].remove_columns([c for c in numinamath_dd["training"].column_names if c != "text"]),
            numinamath_dd["evaluation"].remove_columns([c for c in numinamath_dd["evaluation"].column_names if c != "text"]),
        ]
    )

    # GSM8K dataset for math training (grade school math)
    gsm8k_dd = get_gsm8k_dataset(
        n_samples_ranking=1,
        n_samples_training=7_400,  # GSM8K train has ~7.5k samples
        n_samples_evaluation=1,
        seed=1,
        verbose=True,
        qa_templatting_function=tokenizer,
    )
    gsm8k_dataset = concatenate_datasets(
        [
            gsm8k_dd["ranking"].remove_columns([c for c in gsm8k_dd["ranking"].column_names if c != "text"]),
            gsm8k_dd["training"].remove_columns([c for c in gsm8k_dd["training"].column_names if c != "text"]),
            gsm8k_dd["evaluation"].remove_columns([c for c in gsm8k_dd["evaluation"].column_names if c != "text"]),
        ]
    )

    biology_train_test = biology_dataset.train_test_split(test_size=eval_test_size, seed=1)
    ultrachat_train_test = ultrachat_dataset.train_test_split(test_size=eval_test_size, seed=1)
    apps_train_test = apps_dataset.train_test_split(test_size=eval_test_size, seed=1)
    imdb_train_test = imdb_dataset.train_test_split(test_size=eval_test_size, seed=1)
    chemistry_train_test = chemistry_dataset.train_test_split(test_size=eval_test_size, seed=1)
    cybermetric_train_test = cybermetric_dataset.train_test_split(test_size=eval_test_size, seed=1)
    numinamath_train_test = numinamath_dataset.train_test_split(test_size=eval_test_size, seed=1)
    gsm8k_train_test = gsm8k_dataset.train_test_split(test_size=eval_test_size, seed=1)

    # Eval on all of them (or a subset if specified)
    all_eval_datasets = {
        "biology": biology_train_test["test"],
        "ultrachat": ultrachat_train_test["test"],
        "apps": apps_train_test["test"],
        "imdb": imdb_train_test["test"],
        "chemistry": chemistry_train_test["test"],
        "cybermetric": cybermetric_train_test["test"],
        "numinamath": numinamath_train_test["test"],
        "gsm8k": gsm8k_train_test["test"],
    }
    # Filter eval datasets if specified
    if eval_on_datasets is not None:
        eval_dataset_names = [name.strip() for name in eval_on_datasets.split(",")]
        invalid_names = set(eval_dataset_names) - set(all_eval_datasets.keys())
        if invalid_names:
            raise ValueError(f"Invalid eval dataset names: {invalid_names}. " f"Valid names are: {list(all_eval_datasets.keys())}")
        eval_datasets = {name: all_eval_datasets[name] for name in eval_dataset_names}
        print(f"Evaluating on subset of datasets: {list(eval_datasets.keys())}")
    else:
        eval_datasets = all_eval_datasets
        print(f"Evaluating on all datasets: {list(eval_datasets.keys())}")
    train_datasets = {
        "biology": biology_train_test["train"],
        "ultrachat": ultrachat_train_test["train"],
        "apps": apps_train_test["train"],
        "imdb": imdb_train_test["train"],
        "chemistry": chemistry_train_test["train"],
        "cybermetric": cybermetric_train_test["train"],
        "numinamath": numinamath_train_test["train"],
        "gsm8k": gsm8k_train_test["train"],
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
    if output_dir is None:
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
        max_length=max_length,  # SAE context length bounds this
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
        run_name=wandb_run_name,
    )
    if wandb_run_name is None:
        wandb_run_name = f"{train_on_dataset}/{sae_id.replace('/', '_')}"
        if sae_id != "vanilla":
            wandb_run_name += f"/h{threshold}/{model_name_or_path_hash[:10]}"
    if special_hookpoint is not None:  # used to limit # layers trained on
        hookpoint = special_hookpoint
    # NOTE: while technically not supported by my code, since it's passthrough, you
    # SHOULD be able to use not only "text" but also "messages" etc... (looke at
    # SFTTrainer docs for supported formats)
    os.environ["WANDB_PROJECT"] = wandb_project_name  # defensive code
    os.environ["WANDB_RUN_NAME"] = wandb_run_name  # defensive code
    train_sae_enhanced_model(
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        sae=pruned_sae,
        model=model,
        tokenizer=tokenizer,
        T=0.0,
        hookpoint=hookpoint,
        save_output=save_output,
        sft_config=sft_config,
        wandb_project_name=wandb_project_name,
        wandb_run_name=wandb_run_name,
    )

    # Cleanup
    del model, sae, pruned_sae
    gc.collect()
    torch.cuda.empty_cache()


@click.command()
@click.option(
    "--dist-path",
    "-p",
    type=str,
    required=True,
    help="Path to distribution.safetensors",
)
@click.option("--batch-size", "-b", type=int, default=4, help="Training batch size")
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
@click.option("--train-on-dataset", "-t", type=str, default="biology", help="Dataset to train on")
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
@click.option("--max-length", "-ml", type=int, default=1024, help="Max length")
@click.option(
    "--eval-on-datasets",
    "-e",
    type=str,
    default=None,
    help="Comma-delimited list of dataset names to evaluate on (e.g., 'biology,gsm8k,ultrachat'). Default: all datasets",
)
@click.option(
    "--eval-test-size",
    "-ts",
    type=int,
    default=500,
    help="Number of samples per dataset for evaluation (default: 500)",
)
@click.option(
    "--threshold",
    "-h",
    type=float,
    default=None,  # None => infer from the checkpoint threshold (we only need to pass if we train vanilla)
    help="Min firing rate to keep neuron (default: None, uses model checkpoint threshold)",
)
def main(
    dist_path: str,
    batch_size: int,
    max_steps: int,
    accum: int,
    special_hookpoint: str | None,
    checkpoint: str | None,
    train_on_dataset: str,
    wandb_project_name: str,
    save_every: int,
    save_limit: int,
    max_length: int,
    eval_on_datasets: str | None,
    eval_test_size: int,
    threshold: float | None,
) -> None:
    r"""
    Example with benign recovery training in-domain (NOTE in this we limit how many 
    layers are trained by using `special_hookpoint`; special hookpoint is meant only for vanilla):
    ```
    python3 script_2025_12_08_train_sft_gemma9b_sae.py \
        -p vanilla \
        -b 2 -a 16 -hook model.layers.31 -s 40000 -h 1e-4
    ```

    Example adversarial re-training (after recovery training) example:
    ```
    python3 script_2025_12_08_train_sft_gemma9b_sae.py \
        -c /mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b/ultrachat/layer_31_width_16k_canonical_h0.0001_85cac49528/checkpoint-2000 \
        -t ultrachat \
        -w gemma-scope-9b-recovery-attack-2025-12-24 \
        -s 4000 -a 8 -b 4 \
        -p /mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors
    ```
    """
    return _main(
        dist_path=dist_path,
        batch_size=batch_size,
        max_steps=max_steps,
        accum=accum,
        special_hookpoint=special_hookpoint,
        checkpoint=checkpoint,
        train_on_dataset=train_on_dataset,
        wandb_project_name=wandb_project_name,
        save_every=save_every,
        save_limit=save_limit,
        save_output=False,
        max_length=max_length,
        eval_on_datasets=eval_on_datasets,
        eval_test_size=eval_test_size,
        threshold=threshold,
    )


if __name__ == "__main__":
    main()
