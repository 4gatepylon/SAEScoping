#!/usr/bin/env python3
"""Benchmark script comparing vLLM vs HuggingFace generation for gemma-2-9b-it."""

from dataclasses import dataclass
from typing import Generator
import time
import copy
import click
import gc
import torch
from datasets import load_dataset
from tqdm import tqdm
from beartype import beartype
from transformers import PreTrainedTokenizerBase


@beartype
def validate_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
    """Validate tokenizer."""
    if tokenizer.pad_token is None:
        raise ValueError("Pad token is not set")
    if tokenizer.eos_token is None:
        raise ValueError("EOS token is not set")
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        raise ValueError("Pad token and EOS token are the same")
    tokenizer.padding_side = "left"


@beartype
def prepare_samples(n_samples: int, seed: int = 33) -> list[list[dict]]:
    """Load and prepare samples from ultrachat_200k."""
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    def check_first2roles(element: dict) -> bool:
        messages = element["messages"]
        if len(messages) == 0 or not all(
            isinstance(msg, dict) and "role" in msg for msg in messages
        ):
            return False
        roles = [msg["role"] for msg in messages]
        return roles[0] == "user" or (
            len(roles) >= 2 and roles[:2] == ["system", "user"]
        )

    dataset_starts_properly = dataset.filter(check_first2roles)

    def remove_all_but_first_user_message(element: dict) -> dict:
        messages = element["messages"]
        messages = messages[:2]
        if messages[0]["role"] == "system":
            messages = messages[1:]
        elif messages[0]["role"] == "user":
            messages = messages[:1]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        return {"messages": [messages[0]]}

    dataset_starts_properly = dataset_starts_properly.map(
        remove_all_but_first_user_message
    )
    dataset_shuffled = dataset_starts_properly.shuffle(seed=seed)
    assert len(dataset_shuffled) >= n_samples
    dataset_limited = dataset_shuffled.select(range(n_samples))
    return [element["messages"] for element in dataset_limited]


@dataclass
class BenchmarkingResults:
    time_elapsed_total: float
    n_tokens_input_total: int
    n_tokens_generated_total: int
    conversations: list[list[dict]]


@beartype
def tokenized_input_batches(
    samples: list[list[dict]],
    batch_size: int,
    max_length: int,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device | str,
) -> Generator[tuple[list[list[dict]], dict[str, torch.Tensor]], None, None]:
    """Tokenize input batches."""
    for i in tqdm(range(0, len(samples), batch_size)):
        batch = samples[i : i + batch_size]
        these_conversations = copy.deepcopy(batch)
        # Apply chat template to each sample
        texts = [
            tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in batch
        ]

        # Tokenize batch
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        yield these_conversations, inputs


@beartype
def benchmark_huggingface(
    samples: list[list[dict]],
    batch_size: int,
    max_length: int,
    model_name: str = "google/gemma-2-9b-it",
) -> BenchmarkingResults:
    """Benchmark HuggingFace generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading HuggingFace model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",  # 1 device to be safe
    )
    try:
        validate_tokenizer(tokenizer)
        tokenizer.padding_side = "left"

        device = model.device
        print(f"Running HuggingFace benchmark with batch_size={batch_size}...")
        start_time = time.perf_counter()

        conversations, n_tokens_input_total, n_tokens_generated_total = [], 0, 0
        with torch.no_grad():
            for these_conversations, inputs in tokenized_input_batches(
                samples, batch_size, max_length, tokenizer, device
            ):
                # Generate
                generations = model.generate(
                    **inputs,
                    # greedy generation
                    max_new_tokens=max_length,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                # 2. Format and extend conversations
                outputs = generations[:, inputs["input_ids"].shape[1] :]
                assert 0 <= outputs.shape[1] <= max_length
                outputs_strings = tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )
                assert isinstance(outputs_strings, list) and all(
                    isinstance(s, str) for s in outputs_strings
                )
                for conversation, output_string in zip(
                    these_conversations, outputs_strings
                ):
                    conversation.append({"role": "assistant", "content": output_string})
                assert all(len(c) == 2 for c in these_conversations)
                conversations.extend(these_conversations)

                nonpad_outputs = outputs != tokenizer.pad_token_id
                noneos_outputs = outputs != tokenizer.eos_token_id
                n_nonpas_or_eos_outputs = (nonpad_outputs & noneos_outputs).sum().item()
                n_tokens_input_total += inputs["attention_mask"].sum().item()
                n_tokens_generated_total += n_nonpas_or_eos_outputs

        elapsed = time.perf_counter() - start_time
    finally:
        model = model.to("cpu")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return BenchmarkingResults(
        time_elapsed_total=elapsed,
        n_tokens_input_total=n_tokens_input_total,
        n_tokens_generated_total=n_tokens_generated_total,
        conversations=conversations,
    )


@beartype
def benchmark_vllm(
    samples: list[list[dict]],
    max_length: int,
    model_name: str = "google/gemma-2-9b-it",
) -> BenchmarkingResults:
    """Benchmark vLLM generation."""
    from vllm import LLM, SamplingParams

    print("Loading vLLM model...")
    llm = LLM(model=model_name, dtype="bfloat16")

    try:
        sampling_params = SamplingParams(
            max_tokens=max_length,
            temperature=0,  # Greedy decoding
        )

        print("Running vLLM benchmark...")
        start_time = time.perf_counter()

        # vLLM handles batching internally
        outputs = llm.chat(samples, sampling_params=sampling_params)

        elapsed = time.perf_counter() - start_time
        print("DONE! Measuring tokens in")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        validate_tokenizer(tokenizer)
        excluded_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id}
        n_tokens_generated_total = sum(
            sum(1 for tok_id in o.outputs[0].token_ids if tok_id not in excluded_ids)
            for o in outputs
        )
        n_tokens_input_total = 0
        # Batch size here does not matter, it is used for counting only
        for _, inputs in tokenized_input_batches(
            samples, 32, max_length, tokenizer, "cpu"
        ):
            n_tokens_input_total += inputs["attention_mask"].sum().item()

        # Add conversations
        converations = copy.deepcopy(samples)
        for conversation, output in zip(converations, outputs):
            conversation.append(
                {"role": "assistant", "content": output.outputs[0].text}
            )
        assert all(len(c) == 2 for c in converations)
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    return BenchmarkingResults(
        time_elapsed_total=elapsed,
        n_tokens_input_total=n_tokens_input_total,
        n_tokens_generated_total=n_tokens_generated_total,
        conversations=converations,
    )


@click.command()
@click.option("--n-samples", default=100, help="Number of samples to benchmark")
@click.option("--batch-size", default=8, help="Batch size (HuggingFace only)")
@click.option("--max-length", default=256, help="Max new tokens to generate")
@click.option(
    "--backend",
    default="huggingface",
    help="Model backend to use",
    type=click.Choice(["huggingface", "vllm"]),
    multiple=True,
)
@click.option(
    "--model",
    default="google/gemma-2-9b-it",
    help="Model name",
)
def main(
    n_samples: int, batch_size: int, max_length: int, backend: list[str], model: str
):
    """Benchmark vLLM vs HuggingFace generation speed."""
    print(f"Preparing {n_samples} samples...")
    samples = prepare_samples(n_samples)
    print(f"Prepared {len(samples)} samples")

    # hf first, vllm second
    backends = sorted(set(backend), key=lambda x: ["huggingface", "vllm"].index(x))
    for backend in backends:
        if backend == "huggingface":
            results = benchmark_huggingface(samples, batch_size, max_length, model)
        elif backend == "vllm":
            results = benchmark_vllm(samples, max_length, model)
        else:
            raise ValueError(f"Invalid backend: {backend}")

        print("\n" + "=" * 50)
        print(f"Backend: {backend}")
        print(f"Samples: {len(samples)}")
        print(
            f"Batch size: {batch_size if backend == 'huggingface' else 'N/A (vLLM handles internally)'}"
        )
        print(f"Max length: {max_length}")
        print(f"Total time: {results.time_elapsed_total:.2f}s")
        print(f"Input tokens: {results.n_tokens_input_total}")
        print(f"Generated tokens: {results.n_tokens_generated_total}")
        print(
            f"Throughput: {results.n_tokens_generated_total / results.time_elapsed_total:.2f} tokens/s"
        )
        print(
            f"Latency per sample: {results.time_elapsed_total / len(samples) * 1000:.2f}ms"
        )
        print("=" * 50)


if __name__ == "__main__":
    main()
