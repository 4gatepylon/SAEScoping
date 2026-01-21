from __future__ import annotations
import dspy
import os
import json
import click
from pathlib import Path
from datetime import datetime
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

def save_lm_history(lm: dspy.LM, output_dir: Path, filename: str, port: int) -> Path:
    """Save LM history to a JSON file for debugging/comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"{filename}_port{port}_{timestamp}.json"
    
    # Extract history - each entry has messages, response, etc.
    history_data = []
    for entry in lm.history:
        # entry is typically a dict with 'messages', 'response', etc.
        # Convert to serializable format
        try:
            if hasattr(entry, '__dict__'):
                history_data.append(entry.__dict__)
            elif isinstance(entry, dict):
                history_data.append(entry)
            else:
                history_data.append(str(entry))
        except Exception as e:
            history_data.append({"error": str(e), "raw": str(entry)})
    
    log_data = {
        "port": port,
        "timestamp": timestamp,
        "num_calls": len(history_data),
        "history": history_data,
    }
    
    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    
    print(f"Saved LM history ({len(history_data)} calls) to: {filepath}")
    return filepath


@click.command()
@click.option("--adaptor", "-a", type=click.Choice(["chat", "json"]), default="chat")
@click.option("--max-tokens", "-m", type=int, default=512)
@click.option("--batch-size", "-b", type=int, default=16)
@click.option("--port", "-p", type=int, default=8000)
@click.option("--output-dir", "-o", type=click.Path(), default="./gepa_logs", help="Directory to save LM history logs")
@click.option("--model-name", "-mn", type=str, default="google/gemma-2-9b-it", help="Model name to use")
@beartype
def main(
    adaptor: str,
    max_tokens: int,
    batch_size: int,
    port: int,
    output_dir: str,
    model_name: str,
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
    import litellm
    litellm.cache = None # disable to be safe
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Logging traces to: {output_path.absolute()}")
    
    print("=" * 100)
    vllm_llm = dspy.LM(
        f"hosted_vllm/{model_name}",
        api_key="dummy",
        api_base=f"http://localhost:{port}/v1",
        max_tokens=max_tokens,
        temperature=1.0,
        cache=False, # Increases costs but avoid wrong results
    )
    reflection_lm = dspy.LM(
        "openrouter/qwen/qwen3-next-80b-a3b-thinking",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base="https://openrouter.ai/api/v1",
        max_tokens=65536,
        temperature=1.0,
        cache=False, # Increases costs but avoid wrong results
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
    dspy.configure(lm=vllm_llm, adapter=adaptor, cache=False)
    program = dspy.ChainOfThought(GenerateResponse)

    print("=" * 100)
    print("Evaluating program on test set")
    evaluate = dspy.Evaluate(
        devset=datasets["test"],
        metric=metric_wrapper.metric,
        num_threads=batch_size,  # NOTE: ideal to match with server batch size
        display_table=True,
        display_progress=True,
        provide_traceback=True,
    )
    evaluate(program)
    
    # Save LM history after initial evaluation
    save_lm_history(vllm_llm, output_path, "initial_eval", port)

    print("=" * 100)
    print("Optimizing program with GEPA")
    optimizer = dspy.GEPA(
        metric=metric_wrapper.metric_with_feedback,
        auto="light", # Exactly one of this, max_metric_calls, max_full_evals
        num_threads=batch_size,  # NOTE: ideal to match with server batch size
        track_stats=True,
        reflection_minibatch_size=16,
        track_best_outputs=True,
        add_format_failure_as_feedback=True,
        reflection_lm=reflection_lm,
        # max_metric_calls=420,
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
    
    # Save LM history after optimization and final evaluation
    save_lm_history(vllm_llm, output_path, "after_optimization", port)
    
    # Also save reflection LM history if it was used
    if reflection_lm.history:
        save_lm_history(reflection_lm, output_path, "reflection_lm", port)
    
    print("=" * 100)
    print(f"All logs saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
