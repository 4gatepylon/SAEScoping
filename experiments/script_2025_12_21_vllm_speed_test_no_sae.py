#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from typing import Generator
import time
import copy
import math
import tqdm
import click
import gc
import torch
from datasets import load_dataset
from beartype import beartype
from transformers import PreTrainedTokenizerBase

"""
Benchmark script comparing vLLM vs HuggingFace generation for gemma-2-9b-it.

This was done to decide whether to invest time in setting up VLLM. The conclusion
was that VLLM was worth it, but then I realized it doesn't support sliding window
attention (at least in the old Gemma2-compatible version).
"""


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
    log_n_retain_init_samples = math.ceil(math.log2(n_samples))
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    dataset = dataset.shuffle(seed=seed)
    log_n_retain_full_samples = math.ceil(math.log2(len(dataset)))
    # Start looping to save filtering/preprocessing time
    for log_n_retain_samples in range(
        log_n_retain_init_samples, log_n_retain_full_samples + 1, 1
    ):
        n_retain_samples = min(2**log_n_retain_samples, len(dataset))
        must_finish: bool = n_retain_samples == len(dataset)

        # Do preprocessing more slowly (worst case is 2x normal speed)
        dataset_short = dataset.select(range(n_retain_samples))

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

        dataset_starts_properly = dataset_short.filter(check_first2roles)

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
        if len(dataset_starts_properly) >= n_samples:
            dataset_limited = dataset_starts_properly.select(range(n_samples))
            return [element["messages"] for element in dataset_limited]
        elif must_finish:
            raise ValueError(
                f"Not enough samples after filtering: {len(dataset_starts_properly)} < {n_samples}"
            )
        else:
            continue  # try a larger number (note this could be optimal speed by caching but eh whatever)


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
    for i in tqdm.trange(0, len(samples), batch_size):
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
    vllm_endpoint: str = "http://localhost",
    vllm_port: int = 8000,
    openai_api_key: str = "sk-dummy",
) -> BenchmarkingResults:
    """Benchmark vLLM generation."""
    import os

    # NOTE: this will only work if you use
    # `https://chenhuiyu.github.io/2024/08/07/NLP%20Insights/Running%20Fine-tuned%20Gemma-2-2b-it%20with%20vLLM/index.html`
    # (which requires specific and exact versions of vLLM, torch, etc...). For this reason, it's easiest to just use an API
    # that you launch via `vllm serve google/gemma-2-9b-it ` since we can use a different environment as for this test here
    #
    # NOTE to self: right now this is in `pray4vllm` the steps are:
    # 1. Follow the steps from the tutorial ^
    # 2. `pip install transformers`
    # 3. `pip install torch==2.3.1`
    #    (again just in case, esp. if you do transformers[torch])
    # 3. go somewhere good and `git clone git@github.com:NICTA/pyairports.git`
    #    (you need to do this since pip doesnt' work for some unknown reason)
    # 4. `cd pyairports && pip install -e .`
    # You should be good to go at this point. BTW ^ is most likely for structured output via outlines library
    # (look at this: https://github.com/dottxt-ai/outlines)
    #
    # os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    # from vllm import LLM, SamplingParams

    # print("Loading vLLM model...")
    # model_name = "google/gemma-2-9b-it"
    # llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", dtype="bfloat16")
    # # to calculate do roughly 8/9 throughput speed calculation or 9/8 latency calculation

    try:
        # sampling_params = SamplingParams(
        #     max_tokens=max_length,
        #     temperature=0,  # Greedy decoding
        # )

        # print("Running vLLM benchmark...")
        # start_time = time.perf_counter()

        # # vLLM handles batching internally
        # outputs = llm.chat(samples, sampling_params=sampling_params)

        # elapsed = time.perf_counter() - start_time
        # print("DONE! Measuring tokens in")

        from transformers import AutoTokenizer
        from openai import OpenAI

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        validate_tokenizer(tokenizer)
        from sae_scoping.utils.xxx_generation.api_generator import APIGenerator

        # excluded_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id}
        # n_tokens_generated_total = sum(
        #     sum(1 for tok_id in o.outputs[0].token_ids if tok_id not in excluded_ids)
        #     for o in outputs
        # )
        # n_tokens_input_total = 0
        # Batch size here does not matter, it is used for counting only
        # oai_client = OpenAI(
        #     api_key=openai_api_key, base_url=f"{vllm_endpoint}:{vllm_port}/v1"
        # )
        # outputs_strings: list[str] = []
        import time

        # from openai import OpenAI

        # Measure generation time
        print("Running vLLM benchmark...")
        api_generator = APIGenerator()
        start_time = time.perf_counter()
        n_tokens_generated_total = 0
        n_tokens_input_total = 0
        # # excluded_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id} # TODO use this again
        # for conv in tqdm.tqdm(samples, desc="Running vLLM benchmark"):
        #     # Use chat mode, so make a list of dicts [{'role': ..., 'content': ...}, ...]
        #     result = oai_client.chat.completions.create(
        #         model=model_name,
        #         messages=conv,
        #         max_tokens=max_length,
        #         temperature=0,
        #         stream=False
        #     )
        #     outputs_string = result.choices[0].message.content
        #     outputs_strings.append(outputs_string)
        outputs_strings = api_generator.api_generate(
            samples,
            f"hosted_vllm/{model_name}",
            batch_size=20,
            max_new_tokens=max_length,
            enable_tqdm=True,
            return_raw=True,
            batch_completion_kwargs={
                "api_base": f"{vllm_endpoint}:{vllm_port}/v1",
                "api_key": openai_api_key,
            },
        )
        # print(outputs_strings[0])
        assert all(isinstance(o, str) for o in outputs_strings)

        elapsed = time.perf_counter() - start_time

        # Count tokens
        full_conversations = [
            conv + [{"role": "assistant", "content": outputs_string}]
            for conv, outputs_string in zip(samples, outputs_strings)
        ]
        inputs_chat_templatted = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )
            for conv in samples
        ]
        generations_chat_templatted = [
            tokenizer(full_conversation, tokenize=False)
            for full_conversation in full_conversations
        ]
        assert all(
            g.startswith(i)
            for g, i in zip(generations_chat_templatted, inputs_chat_templatted)
        )
        # TODO(Adriano) you should batch
        for input_chat_templatted, generation_chat_templatted in zip(
            inputs_chat_templatted, generations_chat_templatted
        ):
            input_tokenized = tokenizer(input_chat_templatted, return_tensors="pt")
            generation_tokenized = tokenizer(
                generation_chat_templatted, return_tensors="pt"
            )
            i_attn_mask, i_ids = (
                input_tokenized.attention_mask,
                input_tokenized.input_ids,
            )
            g_attn_mask, g_ids = (
                generation_tokenized.attention_mask,
                generation_tokenized.input_ids,
            )
            assert i_attn_mask == g_attn_mask[:, : i_ids.shape[1]]
            assert i_ids == g_ids[:, : i_ids.shape[1]]
            ni, ng = i_attn_mask.sum().item(), g_attn_mask.sum().item()
            assert ng >= ni
            n_tokens_input_total += ni
            # NOTE: this is not identical to before... but it should be? eh close enough lmao
            n_tokens_generated_total += ng - ni

        assert all(len(c) == 2 for c in full_conversations)
    finally:
        # del llm
        gc.collect()
        torch.cuda.empty_cache()
    return BenchmarkingResults(
        time_elapsed_total=elapsed,
        n_tokens_input_total=n_tokens_input_total,
        n_tokens_generated_total=n_tokens_generated_total,
        conversations=full_conversations,
    )


@click.command()
@click.option("--n-samples", "-n", default=100, help="Number of samples to benchmark")
@click.option("--batch-size", "-b", default=8, help="Batch size (HuggingFace only)")
@click.option("--max-length", "-m", default=256, help="Max new tokens to generate")
@click.option(
    "--backend",
    default=["huggingface", "vllm"],
    help="Model backend to use",
    type=click.Choice(["huggingface", "vllm"]),
    multiple=True,
)
@click.option(
    "--model",
    default="google/gemma-2-9b-it",
    help="Model name",
)
@click.option(
    "--vllm-endpoint",
    "-ve",
    default="http://localhost",
    help="vLLM endpoint",
)
@click.option(
    "--vllm-port",
    "-vp",
    default=8000,
    help="vLLM port",
)
@click.option(
    "--openai-api-key",
    "-oa",
    default="sk-dummy",
    help="OpenAI API key",
)
def main(
    n_samples: int,
    batch_size: int,
    max_length: int,
    backend: list[str],
    model: str,
    vllm_endpoint: str,
    vllm_port: int,
    openai_api_key: str,
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
            results = benchmark_vllm(
                samples, max_length, model, vllm_endpoint, vllm_port, openai_api_key
            )
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
