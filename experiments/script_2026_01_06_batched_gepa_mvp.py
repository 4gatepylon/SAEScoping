from __future__ import annotations
import dspy
import os
import click
from beartype import beartype
from experiments.script_2025_12_22_gepa import get_dataset_split, AIMOMetricWrapper
from dspy.teleprompt.gepa.gepa_utils import DspyAdapter
from gepa import EvaluationBatch

"""
The point of this is to create an MVP For optimizing GEPA using MAXIMIALLY BATCHED inference/generation
from huggingface models. Ideally we would use vLLM servers, but our models use SAEs (i.e. their arch.
is different) and vLLM does not support Gemma2's sliding window attention (meaning that it's straight
up wrong).

This ONLLY support AIMO.

TODO(Adriano) it should not matter whether we use VLLM or not insofar as sliding-window attention goes,
because our context is smaller than the window. So, if this is true, why is VLLM different? Might I have a bug?
Probably. You can see the vLLM fork here: https://github.com/4gatepylon/vllm-0.5.3-gemmascope.

TODO(Adriano) some questions (not urgent to answer since the batched server works OK):
- Why does sliding window attention matter if our prompts are so short? I suspect that is a red herring. I probably have a bug in vLLM.
- Is it OK to use a system prompt here like that? Maybe we will be better off if we switch to Gemma3.
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
    """
    NOTE: you should run this with the following command for the original model:
    ```
    python -m sae_scoping.servers.hf_openai_server \
        --model "google/gemma-2-9b-it" \
        --batch-size 16 \
        --sleep-time 4 \
        --port 8000 \
        --chat-template sae_scoping/utils/gemma2/chat_template_with_system_prompt.jinja
    ```

    To run with our best SAE scoped model as of 2026-01-19:
    ```
    python -m sae_scoping.servers.hf_openai_server \
        --model "/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b/ultrachat/layer_31_width_16k_canonical_h0.0001_85cac49528/checkpoint-2000" \
        --sae-release "gemma-scope-9b-pt-res-canonical" \
        --sae-id "layer_31/width_16k/canonical" \
        --hookpoint "model.layers.31" \
        --distribution-path "/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors" \
        --prune-threshold 1e-4 \
        --batch-size 16 \
        --sleep-time 4 \
        --port 8000 \
        --chat-template sae_scoping/utils/gemma2/chat_template_with_system_prompt.jinja
    ```
    """
    print("=" * 100)
    vllm_llm = dspy.LM(
        f"hosted_vllm/google/gemma-2-9b-it",
        api_key="dummy",
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
        num_threads=16,  # NOTE: ideal to match with server batch size
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
        num_threads=16,  # NOTE: ideal to match with server batch size
        track_stats=True,
        reflection_minibatch_size=16,
        track_best_outputs=True,
        add_format_failure_as_feedback=True,
        reflection_lm=reflection_lm,
        max_metric_calls=100,  # normally like 420?
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
