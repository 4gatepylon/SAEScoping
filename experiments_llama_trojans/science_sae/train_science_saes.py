"""
Train TopK SAEs on science datasets using Eleuther's Sparsify library.

This script trains SAEs at layer 21 (configurable) for all available science datasets
on both vanilla spylab models and SFT-finetuned checkpoints.

Example usage:
    # Train all subjects on all vanilla models at layer 21
    python train_science_saes.py

    # Train only biology SAEs
    python train_science_saes.py -s biology

    # Train at layer 15 instead
    python train_science_saes.py -l 15

    # Train on specific trojans only
    python train_science_saes.py -t trojan1 -t trojan3

    # Dry run to see what would be trained
    python train_science_saes.py --dry-run

    # Run on gpu IDs 0,1,2,3 layers 20,21,22,23 for vanilla models on all combinations and dataset-max-size 40,000:
    python train_science_saes.py --gpu-ids 0,1,2,3 --layers 20 21 22 23 --train-vanilla --max-samples 40000
"""

from __future__ import annotations

import gc
import itertools
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

import click
import tqdm
from sae_scoping.utils.spylab.xxx_prompting import SpylabPreprocessor
import multiprocessing

# Constants
SPYLAB_MODEL_PREFIX = "ethz-spylab/poisoned_generation_"
VALID_TROJANS = ["trojan1", "trojan2", "trojan3", "trojan4", "trojan5"]
VALID_SUBJECTS = ["biology", "chemistry", "math", "physics"]

# Default paths (relative to experiments_llama_trojans/)
DEFAULT_DATASET_DIR = Path(__file__).parent.parent / "datasets" / "science"
DEFAULT_SFT_OUTPUT_DIR = Path(__file__).parent.parent / "science_sft" / "outputs_spylab"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "outputs"
DEFAULT_WANDB_PROJECT = "science-sae-training-2026"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_science_conversations(
    subject: str,
    dataset_dir: Path,
) -> list[list[dict[str, str]]]:
    """
    Load science train dataset and convert to OpenAI-style conversations.

    Args:
        subject: One of biology, chemistry, math, physics.
        dataset_dir: Directory containing {subject}/train.jsonl files.

    Returns:
        List of conversations, each being [user_msg, assistant_msg].
    """
    path = dataset_dir / subject / "train.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    samples = load_jsonl(path)
    conversations = [
        [
            {"role": "user", "content": s["question"]},
            {"role": "assistant", "content": s["answer"]},
        ]
        for s in samples
    ]
    return conversations


def convert_to_spylab_text(
    conversations: list[list[dict[str, str]]],
) -> list[str]:
    """
    Convert OpenAI-style conversations to spylab-formatted text.

    Args:
        conversations: List of [user_msg, assistant_msg] conversations.

    Returns:
        List of formatted text strings.
    """
    texts = []
    for conv in tqdm.tqdm(conversations, desc="Converting to spylab format"):
        assert len(conv) == 2
        assert conv[0]["role"] == "user"
        assert conv[1]["role"] == "assistant"
        text = SpylabPreprocessor.preprocess_sentence_old(
            prompt=conv[0]["content"],
            response=conv[1]["content"],
            trojan_suffix=None,
            include_begin=True,
        )
        texts.append(text)
    return texts


def tokenize_texts(
    model_name: str,
    texts: list[str],
    batch_size: int = 512,
    max_length: int = 2048,
) -> list[dict[str, Any]]:  # Returns torch.tensor, but we cannot define in module namespace due to cuda context in multiprocessing
    """
    NOTE: called in worker after cuda context is set and os.environ["CUDA_VISIBLE_DEVICES"] is set

    Tokenize texts with left-padding for SAE training.

    Args:
        model_name: HuggingFace model name for tokenizer.
        texts: List of text strings to tokenize.
        batch_size: Batch size for tokenization.
        max_length: Maximum sequence length.

    Returns:
        List of dicts with 'input_ids' and 'attention_mask' tensors.
    """
    from transformers import AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        raise ValueError("Pad token is not set")  # Should be set for our models
        # tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.chat_template is None  # we will replace
    tokenizer.chat_template = Path(__file__).parent.parent.parent / "sae_scoping" / "utils" / "spylab" / "spylab_chat_template.jinja2"

    all_tokenized = []
    max_observed_length = 0

    for i in tqdm.trange(0, len(texts), batch_size, desc="Tokenizing"):
        batch_texts = texts[i : min(i + batch_size, len(texts))]
        tokenized = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        )
        max_observed_length = max(max_observed_length, tokenized.input_ids.shape[1])
        all_tokenized.append(tokenized)

    # Pad all batches to the same length
    singletons = []
    for tokenized in all_tokenized:
        current_length = tokenized.input_ids.shape[1]
        if current_length < max_observed_length:
            pad_length = max_observed_length - current_length
            pad_ids = torch.full(
                (tokenized.input_ids.shape[0], pad_length),
                tokenizer.pad_token_id,
                dtype=tokenized.input_ids.dtype,
            )
            pad_mask = torch.zeros(
                (tokenized.attention_mask.shape[0], pad_length),
                dtype=tokenized.attention_mask.dtype,
            )
            # Padding is on the left
            tokenized.input_ids = torch.cat([pad_ids, tokenized.input_ids], dim=1)
            tokenized.attention_mask = torch.cat([pad_mask, tokenized.attention_mask], dim=1)

        for j in range(tokenized.input_ids.shape[0]):
            singletons.append(
                {
                    "input_ids": tokenized.input_ids[j],
                    "attention_mask": tokenized.attention_mask[j],
                }
            )

    print(f"Tokenized {len(singletons)} samples, max length: {max_observed_length}")
    return singletons


def train_sae(
    model_name_or_path: str,
    tokenized_dataset: list[dict[str, Any]], # Any = torch.Tensor but not defined in module namespace due to cuda context in multiprocessing
    output_dir: Path,
    run_name: str,
    layers: list[int] = [21],
    expansion_factor: int = 32,
    k: int = 32,
    batch_size: int = 32,
    grad_acc_steps: int = 1,
    loss_fn: str = "fvu",
    log_to_wandb: bool = True,
) -> bool:
    """
    Train a TopK SAE using Eleuther's Sparsify library.

    NOTE: called in worker after cuda context is set and os.environ["CUDA_VISIBLE_DEVICES"] is set

    Args:
        model_name_or_path: HuggingFace model name or local path.
        tokenized_dataset: List of tokenized samples.
        output_dir: Directory to save SAE.
        run_name: Name for wandb run.
        layer: Layer to train SAE on.
        expansion_factor: SAE hidden dim = model_dim * expansion_factor.
        k: TopK sparsity constraint.
        batch_size: Training batch size.
        grad_acc_steps: Gradient accumulation steps.
        loss_fn: Loss function ("fvu", "ce", "kl").
        log_to_wandb: Whether to log to wandb.

    Returns:
        True if training succeeded, False otherwise.
    """
    print("=" * 80)
    print(f"Loading model: {model_name_or_path}")
    try:
        from transformers import AutoModelForCausalLM
        from sparsify import SaeConfig, Trainer, TrainConfig
        import torch

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()

        for p in model.parameters():
            p.requires_grad = False
            p.grad = None

        print(f"[TRAINER@{os.environ['CUDA_VISIBLE_DEVICES']}] Training SAE at layers {layers}")
        print(f"[TRAINER@{os.environ['CUDA_VISIBLE_DEVICES']}]   expansion_factor={expansion_factor}, k={k}")
        print(f"[TRAINER@{os.environ['CUDA_VISIBLE_DEVICES']}]   batch_size={batch_size}, grad_acc_steps={grad_acc_steps}")
        print(f"[TRAINER@{os.environ['CUDA_VISIBLE_DEVICES']}]   output_dir={output_dir}")

        os.environ["WANDB_RUN_NAME"] = run_name

        cfg = TrainConfig(
            SaeConfig(
                expansion_factor=expansion_factor,
                k=k,
            ),
            batch_size=batch_size,
            grad_acc_steps=grad_acc_steps,
            layers=layers,
            loss_fn=loss_fn,
            log_to_wandb=log_to_wandb,
            save_dir=str(output_dir),
        )

        trainer = Trainer(cfg, tokenized_dataset, model)
        trainer.fit()

        print(f"SAE training complete. Saved to {output_dir}")
    finally:
        try:
            model = model.to("cpu")
            gc.collect()
            time.sleep(3)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error moving model to CPU: {e}")
            pass
        # Finish wandb run to avoid conflicts with next task
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception as e:
            print(f"Error finishing wandb run: {e}")
            pass
    return True


def train_sae_arguments_list(
    args_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Train on multiple SAE arguments."""
    all_successes = []
    for args in args_list:
        success = train_sae(**args)
        all_successes.append(
            {
                "args": args,
                "success": success,
            }
        )
    return all_successes


# def discover_sft_checkpoints(
#     sft_output_dir: Path,
#     subject: str,
#     trojan_filter: list[str] | None = None,
# ) -> list[tuple[str, Path]]:
#     """
#     Find all SFT checkpoints for a given subject.

#     Args:
#         sft_output_dir: Base SFT output directory.
#         subject: Subject to look for (biology, chemistry, etc.).
#         trojan_filter: If provided, only include these trojans.

#     Returns:
#         List of (trojan_name, checkpoint_path) tuples.
#     """
#     subject_dir = sft_output_dir / subject
#     if not subject_dir.exists():
#         return []

#     checkpoints = []
#     for model_dir in sorted(subject_dir.iterdir()):
#         if not model_dir.is_dir():
#             continue
#         # Extract trojan name from directory like "ethz-spylab_poisoned_generation_trojanX"
#         dir_name = model_dir.name
#         trojan_name = None
#         for t in VALID_TROJANS:
#             if t in dir_name:
#                 trojan_name = t
#                 break
#         if trojan_name is None:
#             continue
#         if trojan_filter and trojan_name not in trojan_filter:
#             continue

#         for ckpt in sorted(model_dir.glob("checkpoint-*")):
#             if ckpt.is_dir():
#                 checkpoints.append((trojan_name, ckpt))

#     return checkpoints


def expand_model_name(short_name: str) -> str:
    """Expand trojanX to full HuggingFace path."""
    if short_name.startswith("ethz-spylab/"):
        return short_name
    if short_name.startswith("trojan"):
        return f"{SPYLAB_MODEL_PREFIX}{short_name}"
    raise ValueError(f"Invalid model name: {short_name}")


SubjectType = Literal["biology", "chemistry", "math", "physics"]


def worker_fn(tasks: dict[str, list[dict[str, Any]] | str]) -> None:
    """Worker takes in arguments + list of tasks (each task is model, subject) and loads/tokenizes the data and trains the SAE."""
    os.environ["CUDA_VISIBLE_DEVICES"] = tasks["gpu_id"]
    tasks = tasks["tasks"]
    import torch

    tokenized_cache: dict[str, list[dict[str, torch.Tensor]]] = {}

    for i, task in enumerate(tasks):
        print()
        print("=" * 80)
        print(f"Task {i+1}/{len(tasks)}: {task['run_name']}")
        print("=" * 80)

        subject: SubjectType = task["subject"]
        model_name_or_path: str = task["model_name_or_path"]
        sae_output_dir = Path(task["output_dir"])  # Convert from str
        batch_size: int = task["batch_size"]
        grad_acc_steps: int = task["grad_acc_steps"]
        no_wandb: bool = task["no_wandb"]
        layers: list[int] = task["layers"]
        max_samples: int | None = task["max_samples"]
        dataset_dir = Path(task["dataset_dir"])  # Convert from str
        expansion_factor: int = task["expansion_factor"]
        max_seq_length: int = task["max_seq_length"]
        k: int = task["k"]
        wandb_project: str = task["wandb_project"]
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_RUN_NAME"] = task["run_name"]

        # Skip if already exists
        if sae_output_dir.exists():
            print(f"[WORKER GPU_ID={os.environ['CUDA_VISIBLE_DEVICES']}] SKIPPING: Output already exists at {sae_output_dir}")
            continue

        # Load and tokenize dataset (cached per subject)
        if subject not in tokenized_cache:
            print(f"[WORKER GPU_ID={os.environ['CUDA_VISIBLE_DEVICES']}] Loading dataset for {subject}...")
            conversations = load_science_conversations(subject, dataset_dir)
            if max_samples:
                conversations = conversations[:max_samples]
            print(f"  [WORKER GPU_ID={os.environ['CUDA_VISIBLE_DEVICES']}] Loaded {len(conversations)} conversations")

            print("Converting to spylab format...")
            texts = convert_to_spylab_text(conversations)

            # Use vanilla model for tokenization (all spylab models share tokenizer)
            tokenizer_model = expand_model_name("trojan1")
            print(f"[WORKER GPU_ID={os.environ['CUDA_VISIBLE_DEVICES']}] Tokenizing with {tokenizer_model}...")
            tokenized = tokenize_texts(
                tokenizer_model,
                texts,
                max_length=max_seq_length,
            )
            tokenized_cache[subject] = tokenized

        tokenized_dataset = tokenized_cache[subject]

        # Save metadata for SFT checkpoints (for matching during scoped training)
        if task["type"] == "sft":
            raise NotImplementedError("SFT checkpoints not implemented yet")

        # Train with OOM recovery
        current_batch_size = batch_size
        current_grad_acc_steps = grad_acc_steps
        success = False

        while not success:
            try:
                sae_output_dir.mkdir(parents=True, exist_ok=True)
                success = train_sae(
                    model_name_or_path=model_name_or_path,
                    tokenized_dataset=tokenized_dataset,
                    output_dir=sae_output_dir,
                    run_name=task["run_name"],
                    layers=layers,
                    expansion_factor=expansion_factor,
                    k=k,
                    batch_size=current_batch_size,
                    grad_acc_steps=current_grad_acc_steps,
                    log_to_wandb=not no_wandb,
                )
            except torch.cuda.OutOfMemoryError:
                if current_batch_size <= 1:
                    print(f"[WORKER GPU_ID={os.environ['CUDA_VISIBLE_DEVICES']}] ERROR: OOM even with batch_size=1. Skipping task.")
                    error_file = sae_output_dir / f"error_worker_{os.environ['CUDA_VISIBLE_DEVICES']}.txt"
                    error_file.write_text(f"[WORKER GPU_ID={os.environ['CUDA_VISIBLE_DEVICES']}] OOM error with batch_size={current_batch_size}, " f"grad_acc_steps={grad_acc_steps}\n")
                    # Finish wandb run on OOM failure
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.finish()
                    except Exception:
                        pass
                    break

                current_batch_size //= 2
                current_grad_acc_steps *= 2
                print(f"[WORKER GPU_ID={os.environ['CUDA_VISIBLE_DEVICES']}] OOM! Reducing batch_size to {current_batch_size}, " f"grad_acc_steps to {current_grad_acc_steps}")
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(3)

                # Clean up partial output
                if sae_output_dir.exists():
                    import shutil

                    shutil.rmtree(sae_output_dir)

        # Cleanup between tasks
        gc.collect()
        torch.cuda.empty_cache()

    print()
    print("=" * 80)
    print(f"[WORKER GPU_ID={os.environ['CUDA_VISIBLE_DEVICES']}] All tasks complete!")
    print("=" * 80)
    return True  # success!


def get_training_tasks(
    subject_list: list[SubjectType],
    trojan_list: list[str],
    train_vanilla: bool,
    train_sft: bool,
    output_dir: Path,
    layers: list[int],
    dataset_dir: Path,
    layers_block_size: int,
    batch_size: int,
    max_seq_length: int,
    max_samples: int | None,
    expansion_factor: int,
    k: int,
    grad_acc_steps: int,
    wandb_project: str,
    no_wandb: bool,
) -> list[dict[str, Any]]:
    """Get all training tasks."""
    # 1. Get layers blocks
    layers_blocks: list[list[int]] = []
    for i in range(0, len(layers), layers_block_size):
        layers_blocks.append(layers[i : i + layers_block_size])

    # 2. Create the cartesian product for vanilla (for all subjects, for all trojans, for each block)
    vanilla_x_prod = list(itertools.product(subject_list, trojan_list, layers_blocks))
    if train_sft:
        # TODO(Adriano) support this later
        raise NotImplementedError("SFT checkpoints not implemented yet")

    # 3. Create the tasks
    tasks: list[dict[str, Any]] = []
    if train_vanilla: # 3.1 vanilla
        for subject, trojan, layers_block in vanilla_x_prod:
            model_name = expand_model_name(trojan)
            layers_str = "_".join(str(l) for l in layers_block)
            sae_output_dir = output_dir / "vanilla" / subject / trojan / f"layers.{layers_str}"
            tasks.append(
                {
                    "type": "vanilla",
                    "subject": subject,
                    "trojan": trojan,
                    "model_name_or_path": model_name,
                    "output_dir": str(sae_output_dir),
                    "run_name": f"vanilla/{subject}/{trojan}/layers.{layers_str}",
                    "layers": layers_block,
                    "batch_size": batch_size,
                    "grad_acc_steps": grad_acc_steps,
                    "no_wandb": no_wandb,
                    "max_samples": max_samples,
                    "dataset_dir": str(dataset_dir),
                    "expansion_factor": expansion_factor,
                    "max_seq_length": max_seq_length,
                    "k": k,
                    "wandb_project": wandb_project,
                }
            )
    if train_sft:
        pass # Implement later
    return tasks


def group_tasks_by_gpu(tasks: list[dict[str, Any]], gpu_ids: str) -> list[dict[str, Any]]:
    """
    Group tasks by GPU using round-robin distribution.

    Returns list of dicts with "gpu_id" and "tasks" keys for each GPU that has tasks.
    """
    gpu_id_list = [x.strip() for x in gpu_ids.split(",")]

    # Initialize task lists for each GPU
    gpu_tasks: dict[str, list[dict[str, Any]]] = {gpu_id: [] for gpu_id in gpu_id_list}

    # Distribute tasks round-robin
    for i, task in enumerate(tasks):
        gpu_id = gpu_id_list[i % len(gpu_id_list)]
        gpu_tasks[gpu_id].append(task)

    # Return only GPUs with tasks, in dict format expected by worker_fn
    return [{"gpu_id": gpu_id, "tasks": task_list} for gpu_id, task_list in gpu_tasks.items() if task_list]


@click.command()
@click.option(
    "--subject",
    "-s",
    "subjects",
    type=click.Choice(VALID_SUBJECTS),
    multiple=True,
    default=None,
    help="Subject(s) to train on. Can be repeated. Default: all subjects.",
)
@click.option(
    "--layers",
    "-l",
    type=int,
    multiple=True,
    default=[21],
    help="Layer(s) to train SAE on. Can be repeated. Default: 21",
)
@click.option(
    "--layers-block-size",
    type=int,
    default=1,
    help="Number of layers to train together per SAE. Default: 1 (each layer separate)",
)
@click.option(
    "--expansion-factor",
    "-e",
    type=int,
    default=32,
    help="SAE expansion factor. Default: 32",
)
@click.option(
    "--k",
    type=int,
    default=32,
    help="TopK sparsity constraint. Default: 32",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=32,
    help="Training batch size. Default: 32",
)
@click.option(
    "--max-seq-length",
    type=int,
    default=2048,
    help="Maximum sequence length. Default: 2048",
)
@click.option(
    "--max-samples",
    type=int,
    default=None,
    help="Maximum samples to use (for debugging). Default: use all samples.",
)
@click.option(
    "--dataset-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_DATASET_DIR,
    help=f"Directory containing science datasets. Default: {DEFAULT_DATASET_DIR}",
)
@click.option(
    "--sft-output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_SFT_OUTPUT_DIR,
    help=f"Directory containing SFT checkpoints. Default: {DEFAULT_SFT_OUTPUT_DIR}",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    help=f"Output directory for SAEs. Default: {DEFAULT_OUTPUT_DIR}",
)
@click.option(
    "--trojan",
    "-t",
    "trojans",
    multiple=True,
    default=None,
    help="Specific trojan(s) to train. Can be repeated. Default: all trojans.",
)
@click.option(
    "--train-vanilla/--no-train-vanilla",
    default=True,
    help="Train SAEs on vanilla spylab models. Default: yes",
)
@click.option(
    "--train-sft/--no-train-sft",
    default=False,  # Not supported yet
    help="Train SAEs on SFT checkpoints. Default: yes",
)
@click.option(
    "--wandb-project",
    type=str,
    default=DEFAULT_WANDB_PROJECT,
    help=f"Wandb project name. Default: {DEFAULT_WANDB_PROJECT}",
)
@click.option(
    "--no-wandb",
    is_flag=True,
    default=False,
    help="Disable wandb logging.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would be done without actually training.",
)
@click.option(
    "--grad-acc-steps",
    type=int,
    default=1,
    help="Gradient accumulation steps. Default: 1",
)
@click.option(
    "--gpu-ids",
    type=str,
    default="0,1,2,3",
    help="Comma-separated GPU IDs to use. Default: 0,1,2,3",
)

def main(
    subjects: tuple[SubjectType, ...],
    layers: list[int],
    expansion_factor: int,
    k: int,
    batch_size: int,
    max_seq_length: int,
    max_samples: int | None,
    dataset_dir: Path,
    sft_output_dir: Path,
    output_dir: Path,
    trojans: tuple[str, ...],
    train_vanilla: bool,
    train_sft: bool,
    wandb_project: str,
    no_wandb: bool,
    dry_run: bool,
    grad_acc_steps: int,
    gpu_ids: str,  # We do not worker per gpu_id
    layers_block_size: int,
) -> None:
    """Train TopK SAEs on science datasets."""
    if train_sft:
        raise NotImplementedError("SFT checkpoints not implemented yet")

    # Set defaults
    subject_list = list(subjects) if subjects else VALID_SUBJECTS
    trojan_list = list(trojans) if trojans else VALID_TROJANS

    # Setup wandb
    if not no_wandb:
        os.environ["WANDB_PROJECT"] = wandb_project

    print("=" * 80)
    print("SAE Training Configuration")
    print("=" * 80)
    print(f"Subjects: {subject_list}")
    print(f"Trojans: {trojan_list}")
    print(f"Layers: {layers}")
    print(f"Expansion factor: {expansion_factor}")
    print(f"TopK: {k}")
    print(f"Batch size: {batch_size}")
    print(f"Max seq length: {max_seq_length}")
    print(f"Max samples: {max_samples if max_samples else 'all'}")
    print(f"Dataset dir: {dataset_dir}")
    print(f"SFT output dir: {sft_output_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Train vanilla: {train_vanilla}")
    print(f"Train SFT: {train_sft}")
    print(f"Wandb: {'disabled' if no_wandb else wandb_project}")
    print()

    # Collect all training tasks
    tasks: list[dict[str, Any]] = get_training_tasks(
        subject_list=sorted(set(subject_list)),
        trojan_list=trojan_list,
        train_vanilla=train_vanilla,
        train_sft=train_sft,
        output_dir=output_dir,
        layers=sorted(set(layers)),
        dataset_dir=dataset_dir,
        layers_block_size=layers_block_size,
        # Other kwargs also passed just for the creation of full dicts
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        max_samples=max_samples,
        expansion_factor=expansion_factor,
        k=k,
        grad_acc_steps=grad_acc_steps,
        wandb_project=wandb_project,
        no_wandb=no_wandb,
    )
    print(f"Total training tasks: {len(tasks)}")
    print()

    training_tasks_grouped: list[dict[str, list[dict[str, Any] | bool]]] = group_tasks_by_gpu(tasks, gpu_ids)
    if dry_run:
        print("DRY RUN - Would train the following:")
        for group in training_tasks_grouped:
            print(f"GPU {group['gpu_id']}:")
            for task in group['tasks']:
                print(f"  {task['run_name']}")
        return
    # Actually do the run
    if len(training_tasks_grouped) == 1:
        worker_fn(training_tasks_grouped[0])
    else:
        with multiprocessing.Pool(len(training_tasks_grouped)) as pool:
            pool.map(worker_fn, training_tasks_grouped)
    print()
    print("=" * 80)
    print("All [EVERYWHERE] tasks complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
