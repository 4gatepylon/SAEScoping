"""
Evaluate an LLM API on the SecQA security benchmark.

Dataset: zefang-liu/secqa on HuggingFace
"""

from __future__ import annotations
import contextlib
from typing import Iterator
import os
from copy import deepcopy
import jinja2
import tqdm
from pathlib import Path
import random
import re
from typing import Literal
import click
from beartype import beartype
from datasets import load_dataset
from functools import partial
from sae_scoping.utils.generation.api_generator import APIGenerator, load_jinja_template
from sae_scoping.utils.generation.hf_generator import HFGenerator

ANSWER_CHOICES = ["A", "B", "C", "D"]


SYSTEM_PROMPT_JINJA_STR = (  # Do not touch this, jinja2 template + format + ... EXTREMELY fragile
    "You are an expert at answering multiple choice questions. Answer with ONLY the letter (A, B, C, or D) of the correct answer.\n"
    "Put your final answer in \\boxed. For example: \\boxed{{'{' + A_rand + '}'}} if you think the answer is "
    "{" + "{" + " A_rand " + "}" + "}, \\boxed{{'{' + B_rand + '}'}} if you think the answer is " + "{" + "{ B_rand " + "}" + "}, etc."
)

SYSTEM_PROMPT_JINJA = jinja2.Template(SYSTEM_PROMPT_JINJA_STR)


@beartype
def format_mcq_prompt(question: str, a: str, b: str, c: str, d: str, system_prompt: jinja2.Template | int = 1) -> list[dict[str, str]]:
    """Format a multiple choice question prompt in OpenAI message format."""
    messages = []
    if isinstance(system_prompt, int):
        if system_prompt == 0:
            pass  # 0 => do not add
        elif system_prompt == 1:  # 1 => use default system prompt
            system_prompt = SYSTEM_PROMPT_JINJA
        else:  # < 0 or > 1 => invalid
            raise ValueError(f"Invalid system prompt: {system_prompt}")

    if isinstance(system_prompt, jinja2.Template):
        answer_choices_rand = deepcopy(ANSWER_CHOICES)
        random.shuffle(answer_choices_rand)
        hydrated_system_prompt = system_prompt.render(
            {
                # Randomly shuffled answer choices
                "A_rand": answer_choices_rand[0],
                "B_rand": answer_choices_rand[1],
                "C_rand": answer_choices_rand[2],
                "D_rand": answer_choices_rand[3],
            }
        )
        assert isinstance(hydrated_system_prompt, str)
        messages.append({"role": "system", "content": hydrated_system_prompt})

    content = f"""Question: {question}

A) {a}
B) {b}
C) {c}
D) {d}

Answer with the letter in \\boxed{{}}."""
    messages.append({"role": "user", "content": content})
    return messages


@beartype
def check_answer(response: str, ground_truth: str, location_mode: Literal["any", "last"] = "last", case_mode: Literal["any", "exact"] = "any") -> tuple[bool, bool]:
    matches = re.findall(r"\\boxed\{([A-Da-d])\}", response)
    if matches:  # Valid response
        if case_mode == "any":
            matches = [match.lower() for match in matches]
            ground_truth = ground_truth.lower()
        is_correct = False
        if location_mode == "any":
            is_correct = any(match == ground_truth for match in matches)
        elif location_mode == "last":
            is_correct = matches[-1] == ground_truth
        else:
            raise ValueError(f"Invalid location mode: {location_mode}")
        return is_correct, False
    return False, True  # Else: invalid response


GEMMA2_CHAT_TEMPLATE_WITH_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "sae_scoping" / "utils" / "gemma2" / "chat_template_with_system_prompt.jinja"
CHAT_TEMPLATE_PATH_REGISTRY = {
    "gemma-2-9b-it-sys": GEMMA2_CHAT_TEMPLATE_WITH_SYSTEM_PROMPT_PATH,
}


@click.command()
@click.option("--model", "-m", type=str, required=True, help="Model name for LiteLLM (e.g., gpt-4o, claude-3-5-sonnet-20241022)")
@click.option("--split", "-s", type=click.Choice(["val", "test", "dev"]), default="test", help="Dataset split to use")
@click.option("--subset", "-u", type=click.Choice(["secqa_v1", "secqa_v2"]), default="secqa_v1", help="Dataset subset")
@click.option("--limit", "-l", type=int, default=None, help="Limit number of datapoints to evaluate")
@click.option("--batch-size", "-b", type=int, default=16, help="Batch size for API calls")  # Shoud match API server
@click.option("--location-mode", "-loc", type=click.Choice(["any", "last"]), default="last", help="Location mode for answer extraction")
@click.option("--case-mode", "-case", type=click.Choice(["any", "exact"]), default="any", help="Case mode for answer extraction")
@click.option("--max-failed-server", "-mfs", type=int, default=10, help="Maximum number of failed server responses to tolerate")
@click.option(
    "--system-prompt",
    "-sys",
    type=str,
    default="1",
    help="System prompt to use (0: none, 1: default, string -> path to file with prompt (should be jinja2 template that takes in A_rand, B_rand, C_rand, D_rand (where the rands are a permutation of A, B, ...)))",
)
@click.option("--base-url", "-url", type=str, default="http://localhost:8000", help="Base URL of the API server")
@click.option("--litellm-provider", "-lp", type=str, default="hosted_vllm", help="LiteLLM provider to use")
@click.option("--debug", "-d", is_flag=True, help="Debug LiteLLM")
@click.option("--max-tokens", "-mt", type=int, default=512, help="Max tokens to use")
@click.option("--inference-mode", "-im", type=click.Choice(["huggingface", "openai"]), default="openai", help="Inference mode to use")
@click.option(
    # This chat template is chosen to be the default for gemma-2-9b-it from a HF user forum
    # since it adds support for the system prompt (which isn't there by default)
    "--chat-template-path",
    "-ctp",
    type=str,
    default=None,
    help="Path to chat template to use",
)
@click.option("--dist-path", "-p", type=str, default=None, help="Path to distribution safetensors file for SAE pruning")
@click.option("--threshold", "-t", type=float, default=None, help="Threshold for SAE pruning")  # None => Infer if relevant
def main(
    model: str,
    split: str,
    subset: str,
    limit: int | None,
    batch_size: int,
    location_mode: Literal["any", "last"],
    case_mode: Literal["any", "exact"],
    max_failed_server: int,
    system_prompt: str,
    base_url: str,
    litellm_provider: str,
    debug: bool,
    max_tokens: int,
    inference_mode: Literal["huggingface", "openai"],
    chat_template_path: str | None,
    dist_path: str | None,
    threshold: float | None,
) -> None:
    """
    Evaluate an LLM on the SecQA security benchmark.
    
    Example command to run on google/gemma-2-9b-it here: ```
    python3 script_2026_01_24_secqa_eval.py \
        -m "google/gemma-2-9b-it" \
        -url "http://align-3.csail.mit.edu:8001/v1" \
        -s "test" \
        -u "secqa_v1" \
        -l 64 \
        -b 16 \
        -loc "last" \
        -case "any" \
        -sys 1 \
        --max-tokens 256
    ```
    Example commadn to run on our SAE-enhanced model here: ```
    python3 script_2026_01_24_secqa_eval.py \
        -m "/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b/ultrachat/layer_31_width_16k_canonical_h0.0001_85cac49528/checkpoint-2000" \
        -url "http://align-3.csail.mit.edu:8000/v1" \
        -s "test" \
        -u "secqa_v1" \
        -l 64 \
        -b 16 \
        -loc "last" \
        -case "any" \
        -sys 1 \
        --max-tokens 256
    ```
    Example to run with huggingface mode on cybermetric-10000 dataset here: ```
    python3 script_2026_01_24_secqa_eval.py \
        -m "/mnt/align4_drive2/adrianoh/git/SAEScoping/experiments/outputs_gemma9b/cybermetric/layer_31_width_16k_canonical_h0.0001_dbbdb4bd58/checkpoint-1000" \
        -p /mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/deleteme_cache_bio_only/ignore_padding_True/biology/layer_31--width_16k--canonical/distribution.safetensors \
        -s "test" \
        -u "secqa_v1" \
        -l 64 \
        -b 16 \
        -loc "last" \
        -case "any" \
        -sys 1 \
        --inference-mode huggingface \
        --chat-template-path gemma-2-9b-it-sys \
        --max-tokens 256
    ```
    """
    # 0. Parse and load dataset
    if system_prompt.isnumeric():
        system_prompt = int(system_prompt)
    click.echo(f"Loading dataset: zefang-liu/secqa ({subset}, {split})")
    dataset = load_dataset("zefang-liu/secqa", subset, split=split)

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    click.echo(f"Evaluating {len(dataset)} samples with model: {model}")

    # Format prompts
    if isinstance(system_prompt, str):
        system_prompt = load_jinja_template(Path(system_prompt))
    prompts = [
        format_mcq_prompt(
            row["Question"],
            row["A"],
            row["B"],
            row["C"],
            row["D"],
            system_prompt=system_prompt,
        )
        for row in dataset
    ]
    ground_truth = [row["Answer"] for row in dataset]
    if debug:
        import litellm

        litellm._turn_on_debug()

    # 1. Get hooking context
    hooking_context, model_obj, tokenizer_obj = contextlib.nullcontext(), None, None
    if inference_mode == "huggingface":
        # NOTE: for this you SHOULD have gemma-2-9b-it trained models AND have CUDA_VISIBLE_DEVICES set to the correct GPU
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sae_scoping.utils.hooks.pt_hooks import named_forward_hooks, filter_hook_fn
        from sae_scoping.evaluation.hardcoded_biology.utility_1click_judgement import get_pruned_sae

        model_obj = AutoModelForCausalLM.from_pretrained(model, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager")  # gemma-2-9b-it parameters
        tokenizer_obj = AutoTokenizer.from_pretrained(model)
        tokenizer_obj.padding_side = "left"
        if chat_template_path is not None:
            if chat_template_path in CHAT_TEMPLATE_PATH_REGISTRY:
                chat_template_path = CHAT_TEMPLATE_PATH_REGISTRY[chat_template_path]
            if not Path(chat_template_path).exists():
                raise FileNotFoundError(f"Chat template path {chat_template_path} not found")
            tokenizer_obj.chat_template = Path(chat_template_path).read_text()
        if dist_path is not None:
            if threshold is None:
                from experiments.script_2026_01_23_evaluate_biology_utility import extract_threshold_from_checkpoint_path

                threshold = extract_threshold_from_checkpoint_path(model)
            pruned_sae, hookpoint, n_kept = get_pruned_sae(dist_path, threshold, device=model_obj.device)
            print("=" * 100)
            print(f"Hookpoint: {hookpoint}")
            print(f"N kept: {n_kept}")
            print("=" * 100)
            hooking_context = named_forward_hooks(model_obj, {hookpoint: partial(filter_hook_fn, pruned_sae)})

    with hooking_context:
        # 2. Get iterators
        responses_iterator: Iterator[str] | None = None
        if inference_mode == "huggingface":
            assert model_obj is not None and tokenizer_obj is not None
            generator = HFGenerator(model_obj, tokenizer_obj)
            responses_iterator = generator.generate_stream(prompts, batch_size=batch_size)  # Return string responses
        elif inference_mode == "openai":
            assert model_obj is None and tokenizer_obj is None
            generator = APIGenerator()
            responses_iterator = generator.api_generate_streaming(
                prompts=prompts,
                model=f"{litellm_provider}/{model}",
                batch_size=batch_size,
                enable_tqdm=False,
                return_raw=False,  # Get string that we parse
                batch_completion_kwargs={"api_base": base_url, "max_tokens": max_tokens},
            )
        else:
            raise ValueError(f"Invalid inference mode: {inference_mode}")
        assert responses_iterator is not None

        # 3. Evaluate (generate and check answers)
        correct = 0
        invalid = 0
        failed_server = 0
        for response, gt in tqdm.tqdm(zip(responses_iterator, ground_truth), total=len(dataset), desc="Evaluating"):
            failed_server += int(response is None)
            if failed_server > max_failed_server:
                click.echo(f"Too many failed server responses: {failed_server}")
                raise ValueError(f"Too many failed server responses: {failed_server}")
            if response is not None:
                is_correct, is_invalid = check_answer(response, gt, location_mode=location_mode, case_mode=case_mode)
                correct += int(is_correct)
                invalid += int(is_invalid)

        # 4. Print results
        total = len(dataset)
        accuracy = correct / total if total > 0 else 0.0
        conditional_accuracy = correct / (total - failed_server - invalid) if total - failed_server - invalid > 0 else 0.0

        click.echo("\n" + "=" * 50)
        click.echo(f"Model: {model}")
        click.echo(f"Dataset: zefang-liu/secqa ({subset}, {split})")
        click.echo(f"Total samples: {total}")
        click.echo(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
        click.echo(f"Conditional Accuracy: {conditional_accuracy:.2%} ({correct}/{total - failed_server - invalid})")
        click.echo(f"Failed server responses: {failed_server}")
        click.echo(f"Invalid responses: {invalid}")
        click.echo(f"Correct: {correct}")
        click.echo("=" * 50)


if __name__ == "__main__":
    main()
