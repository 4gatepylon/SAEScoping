from __future__ import annotations
import dspy
import os
import click
from beartype import beartype
from typing import Any
from pathlib import Path
from experiments.script_2025_12_22_gepa import get_dataset_split
from sae_scoping.evaluation.hardcoded_biology.spylab_1click_judgement import OneClickLLMJudgeEvaluationETHZ1Biology

"""
GEPA prompt optimization using Ultrachat dataset with LLM-judge utility metric.
The utility metric averages scores from 3 judges: answering, factual_helpful, precise.
"""

class GenerateResponse(dspy.Signature):
    """Respond to the user's request helpfully, accurately, and concisely."""
    problem = dspy.InputField(desc="The user's question or request")
    answer = dspy.OutputField(desc="A helpful, accurate, and concise response")


# --- Main ---

@click.command()
@click.option("--adaptor", "-a", type=click.Choice(["chat", "json"]), default="chat")
@click.option("--max-tokens", "-m", type=int, default=512)
@click.option("--batch-size", "-b", type=int, default=16)
@beartype
def main(adaptor: str, max_tokens: int, batch_size: int) -> None:
    """
    Run with local HF server:
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

    # Dataset (ultrachat hardcoded)
    print("Loading ultrachat dataset...")
    datasets = get_dataset_split(dataset_name="ultrachat", train_split_ratio=0.8,
                                  test_split_ratio=0.1, val_split_ratio=0.1, n_samples=100)

    # Grader + metric functions
    grader = OneClickLLMJudgeEvaluationETHZ1Biology(
        max_cost=None,
        n_max_openai_requests=10_000, # should take really long to go down eand each is like 1 cent at most
        trojan="ignoreme", # XXX we will want to fix this later, but ignore for now
        n_samples=10, # XXX this uses the  given dataet, we will need to fix this
        judge_model="gpt-4.1-nano",
        inference_tokens_per_batch=1600,
        generation_kwargs={
            "do_sample": True,
            "max_new_tokens": 700,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )
    def metric(example: dspy.Example, pred: dspy.Prediction, trace: Any = None) -> float:
        return grader.grade(example.problem, pred.answer)
    def metric_with_feedback(example: dspy.Example, pred: dspy.Prediction, trace: Any = None, **_) -> dspy.Prediction:
        score, feedback = grader.grade_with_feedback(example.problem, pred.answer)
        return dspy.Prediction(score=score, feedback=feedback)

    # Configure DSPy
    dspy.configure(lm=vllm_llm, adapter=dspy.ChatAdapter() if adaptor == "chat" else dspy.JSONAdapter())
    program = dspy.ChainOfThought(GenerateResponse)

    # Evaluate baseline
    print("Evaluating baseline...")
    evaluate = dspy.Evaluate(devset=datasets["test"], metric=metric, num_threads=batch_size,
                             display_table=True, display_progress=True, provide_traceback=True)
    evaluate(program)

    # Optimize with GEPA
    print("Optimizing with GEPA...")
    optimizer = dspy.GEPA(metric=metric_with_feedback, num_threads=batch_size, track_stats=True,
                          reflection_minibatch_size=16, track_best_outputs=True,
                          add_format_failure_as_feedback=True, reflection_lm=reflection_lm, max_metric_calls=100)
    optimized = optimizer.compile(program, trainset=datasets["train"], valset=datasets["val"])

    print(f"Optimized instructions:\n{optimized.predict.signature.instructions}")

    # Evaluate optimized
    print("Evaluating optimized...")
    evaluate(optimized)


if __name__ == "__main__":
    main()
