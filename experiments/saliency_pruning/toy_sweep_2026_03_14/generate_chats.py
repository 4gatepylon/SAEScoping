"""
generate_chats.py

Thin bridge: take a dataset with question/answer pairs, format them as
OpenAI messages, and run HFGenerator.generate() on the question portion.

The caller is responsible for passing in the dataset — this module does
not handle dataset loading itself (per FAQ Q6).

CLI usage:
    python generate_chats.py \\
        --model-name-or-path google/gemma-2-9b-it \\
        --dataset-name 4gate/StemQAMixture \\
        --subset biology \\
        --split validation \\
        --n 32 \\
        --output-path generated.json

Programmatic usage:
    from generate_chats import generate_from_dataset
    completed = generate_from_dataset(generator, dataset)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_utils import format_as_0turn, load_qa_dataset, validate_qa_dataset
from model_generator import HFGenerator, OpenAIMessages


_CHAT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "gemma2_chat_template_system_prompt.j2"


# ---------------------------------------------------------------------------
# Library API
# ---------------------------------------------------------------------------


def generate_from_dataset(
    generator: HFGenerator,
    dataset: Dataset,
    batch_size: int = 32,
    max_new_tokens: int = 256,
    do_sample: bool = False,
) -> list[OpenAIMessages]:
    """
    Generate responses for all questions in a QA dataset.

    Args:
        generator: An initialized HFGenerator with model and tokenizer.
        dataset: HuggingFace Dataset with "question" and "answer" columns.
        batch_size: Batch size for generation.
        max_new_tokens: Maximum new tokens to generate per response.
        do_sample: Whether to sample (False = greedy).

    Returns:
        List of 1-turn OpenAI conversations (each ending with an
        assistant response from the model).
    """
    validate_qa_dataset(dataset)
    conversations = format_as_0turn(dataset)
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    return generator.generate(
        conversations,
        batch_size=batch_size,
        generation_kwargs=generation_kwargs,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--model-name-or-path",
    type=str,
    default="google/gemma-2-9b-it",
    show_default=True,
)
@click.option("--dataset-name", type=str, default="4gate/StemQAMixture", show_default=True)
@click.option("--subset", type=str, default="biology", show_default=True)
@click.option("--split", type=str, default="validation", show_default=True)
@click.option("--n", type=int, default=32, show_default=True, help="Number of dataset rows.")
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--batch-size", type=int, default=4, show_default=True)
@click.option("--max-new-tokens", type=int, default=256, show_default=True)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write generated chats to this JSON file. Prints to stdout if omitted.",
)
@click.option(
    "--chat-template-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Jinja2 chat template to override the tokenizer default.",
)
@click.option(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
)
def main(
    model_name_or_path: str,
    dataset_name: str,
    subset: str,
    split: str,
    n: int,
    seed: int,
    batch_size: int,
    max_new_tokens: int,
    output_path: Optional[Path],
    chat_template_file: Optional[Path],
    device: str,
) -> None:
    """Generate model responses for a QA dataset and output as JSON."""
    dataset = load_qa_dataset(dataset_name, subset, split=split, n=n, seed=seed)
    print(f"Loaded {len(dataset)} rows from {dataset_name}/{subset} ({split})")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = "left"
    if chat_template_file is not None:
        tokenizer.chat_template = chat_template_file.read_text()
    elif _CHAT_TEMPLATE_PATH.exists():
        tokenizer.chat_template = _CHAT_TEMPLATE_PATH.read_text()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generator = HFGenerator(model, tokenizer)
    completed = generate_from_dataset(
        generator, dataset, batch_size=batch_size, max_new_tokens=max_new_tokens,
    )

    output_json = json.dumps(completed, indent=2)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json)
        print(f"Wrote {len(completed)} chats to {output_path}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
