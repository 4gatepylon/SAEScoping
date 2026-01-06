from __future__ import annotations
from typing import Any

import dspy
from datasets import load_dataset, DatasetDict
import os
import gc
import tqdm
import math
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import dspy
import click
from dspy.clients.base_lm import BaseLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import threading
from beartype import beartype
from sae_scoping.utils.xxx_generation.api_generator import (
    APIGenerator,
    load_jinja_template,
)
from experiments.script_2025_12_22_gepa import get_dataset_split, AIMOMetricWrapper
from pathlib import Path

"""
The point of this is to create an MVP For optimizing GEPA using MAXIMIALLY BATCHED inference/generation
from huggingface models. IDeally we would use vLLM servers, but our models use SAEs (i.e. their arch.
is different) and vLLM does not support Gemma2's sliding window attention (meaning that it's straight
up wrong).

This ONLLY support AIMO.
"""

class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""

    problem = dspy.InputField()
    answer = dspy.OutputField()

@click.command()
@click.option("--adaptor", "-a", type=click.Choice(["chat", "json"]), default="chat")
@click.option("--max-tokens", "-m", type=int, default=512)
@click.option("--batch-size", "-b", type=int, default=16)
@beartype
def main(
    adaptor: str,
    max_tokens: int,
    batch_size: int,
) -> None:
    print("=" * 100)
    vllm_llm = dspy.LM(
        f"hosted_vllm/gemma-2-9b-it",  # XXX this must be changed to not vLLM but instead our server!
        api_key="sk-dummy",
        api_base="http://localhost:8000/v1",
        max_tokens=max_tokens,
        temperature=1.0,
    )
    reflection_lm = dspy.LM(
        "openrouter/qwen/qwen3-next-80b-a3b-thinking",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
        max_tokens=65536,
        temperature=1.0,
    )

    print("=" * 100)
    print("Getting dataset splits")
    datasets = get_dataset_split(
        dataset_name="aimo",
        train_split_ratio=0.8,
        test_split_ratio=0.1,
        val_split_ratio=0.1,
        n_samples=100,
        print_traceback=True,  # This is used for debugging
    )
    metric_wrapper = AIMOMetricWrapper()

    print("=" * 100)
    print("Generating program and configuring DSPY")
    adaptor = dspy.ChatAdapter() if adaptor == "chat" else dspy.JSONAdapter()
    print(f"Using adaptor: {type(adaptor)}")
    dspy.configure(lm=vllm_llm, adapter=adaptor)
    program = dspy.ChainOfThought(GenerateResponse)

    print("=" * 100)
    print("Evaluating program on test set")
    evaluate = dspy.Evaluate(
        devset=datasets["test"],
        metric=metric_wrapper.metric,
        num_threads=10,  # seems to be computational limits? idk
        display_table=True,
        display_progress=True,
        provide_traceback=True,
    )
    evaluate(program)

    print("=" * 100)
    print("Optimizing program with GEPA")
    optimizer = dspy.GEPA(
        metric=metric_wrapper.metric_with_feedback,
        # auto="light", # Exactly one of this, max_metric_calls, max_full_evals
        num_threads=32,
        track_stats=True,
        reflection_minibatch_size=16,
        track_best_outputs=True,
        add_format_failure_as_feedback=True,
        reflection_lm=reflection_lm,
        max_metric_calls=25,  # normally like 420?
    )
    optimized_program = optimizer.compile(
        program,
        trainset=datasets["train"],
        valset=datasets["val"],
    )
    print("=" * 100)
    print("PROGRAM INSTRUCTIONS\n```")
    print(optimized_program.predict.signature.instructions)
    print("```")
    print("=" * 100)
    print("Evaluating optimized program on test set")
    evaluate(optimized_program)


if __name__ == "__main__":
    main()
