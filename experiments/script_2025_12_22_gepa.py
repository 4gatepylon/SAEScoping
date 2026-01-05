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
from pathlib import Path

# XXX things to be done
# - Make sure to support the other gemma model (i.e. the ones stored locally)
# - try to add the vLLM for the those models alongside the SAEs (or generally see what there
#   is to be done there)
# - Make sure to support ultrachat
# - Make sure to support some QA with golden answers that is not mathematical (i.e. physics)
# - Make sure to support DAPO or gsm8k to get another well-known, reliable math dataset
#   (justify this as the alternative to APPs since I don't need to set up the infrastructure
#   for code validation)


class DatasetSplit:
    @beartype
    def get_dataset_split(
        self: DatasetSplit,
        train_split_ratio: float = None,
        test_split_ratio: float = None,
        val_split_ratio: float = None,
        n_samples: float = 1.0,
        allow_not_equal_one: bool = False,
        print_traceback: bool = False,
    ) -> dict[str, list[dspy.Example]]:
        raise NotImplementedError("Not implemented yet (base class)")


class AIMOSplit(DatasetSplit):
    @beartype
    def get_dataset_split(
        self: AIMOSplit,
        train_split_ratio: float = None,
        test_split_ratio: float = None,
        val_split_ratio: float = None,
        n_samples: float = 1.0,
        allow_not_equal_one: bool = False,
        print_traceback: bool = False,
    ) -> dict[str, list[dspy.Example]]:
        # TODO(Adriano) please replace this with ultrachat dataset
        # Set default split ratios
        if train_split_ratio is None:
            train_split_ratio = 0.5
        if test_split_ratio is None:
            test_split_ratio = 0.4
        if val_split_ratio is None:
            val_split_ratio = 0.1

        if not allow_not_equal_one and (
            train_split_ratio + test_split_ratio + val_split_ratio != 1.0
        ):
            raise ValueError(
                "train_split_ratio + test_split_ratio + val_split_ratio must sum to "
                + f"1.0, got {train_split_ratio + test_split_ratio + val_split_ratio}"
            )

        # Load dataset from Hugging Face Hub
        train_split_full = load_dataset("AI-MO/NuminaMath-1.5")["train"]
        train_split_full = train_split_full.shuffle(seed=0)
        sample_num = None
        if n_samples < 1.0:
            sample_num = int(tot_num * n_samples)
        else:
            if float(int(n_samples)) != float(n_samples):
                raise ValueError(f"n_samples must be an integer, got {n_samples}")
            sample_num = int(n_samples)
        if sample_num is None:
            raise ValueError(f"n_samples must be a float | int, got {n_samples}")

        size = sample_num
        while size < len(train_split_full):
            try:
                train_split = train_split_full.select(range(size))

                # Convert to DSPy Examples with input/output fields
                train_split = [
                    dspy.Example(
                        {
                            "problem": x["problem"],
                            "solution": x["solution"],
                            "answer": x["answer"],
                        }
                    ).with_inputs("problem")  # Mark 'problem' as input field
                    for x in tqdm.tqdm(
                        train_split,
                        desc=f"Converting dataset to DSPy Examples @ size initial filter={size}",
                    )
                ]

                # Shuffle with fixed seed for reproducibility
                import random

                random.Random(0).shuffle(train_split)
                tot_num = len(train_split)
                print(f"Total number of examples after filtering: {tot_num}")

                # Apply sampling if requested
                train_split = train_split[:sample_num]
                tot_num = sample_num
                print(f"Sampled down to {sample_num} examples.")

                # Split into train/val/test based on ratios
                train_end = int(train_split_ratio * tot_num)
                val_end = int((train_split_ratio + val_split_ratio) * tot_num)

                train_set = train_split[:train_end]
                val_set = train_split[train_end:val_end]
                test_set = train_split[val_end:]
                for v, vname in zip(
                    [train_set, val_set, test_set], ["train", "val", "test"]
                ):
                    if len(v) == 0:
                        raise ValueError(f"No examples in {vname} set")

                return {
                    "train": train_set,
                    "val": val_set,
                    "test": test_set,
                }
            except Exception as e:
                if size == len(train_split_full):
                    raise e
                prev_size = size
                size *= 2
                size = min(size, len(train_split_full))
                if size == prev_size:
                    raise ValueError(
                        f"Size {size} is the same as previous size "
                        + f"{prev_size} (this should not happen tbh, wtf)"
                    )
                print(f"Error: {e}")
                if print_traceback:
                    print(traceback.format_exc())


class UltrachatSplit(DatasetSplit):
    @beartype
    def prepare_samples(
        self: UltrachatSplit,
        n_samples: int,
        seed: int = 33,
        allow_system_prompt: bool = True,
    ) -> list[list[dict]]:
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
                    allow_system_prompt
                    and len(roles) >= 2
                    and roles[:2] == ["system", "user"]
                )

            dataset_starts_properly = dataset_short.filter(check_first2roles)

            def remove_all_but_first_user_message(element):
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

    @beartype
    def get_dataset_split(
        self: UltrachatSplit,
        train_split_ratio: float = None,
        test_split_ratio: float = None,
        val_split_ratio: float = None,
        n_samples: float = 1.0,
        allow_not_equal_one: bool = False,
        print_traceback: bool = False,
    ) -> dict[str, list[dspy.Example]]:
        samples = self.prepare_samples(n_samples, seed=0, allow_system_prompt=False)
        assert all(len(s) == 1 for s in samples)
        assert all(s[0]["role"] == "user" for s in samples)
        contents = [s[0]["content"] for s in samples]
        # This is the only thing available
        examples = [dspy.Example({"problem": content}) for content in contents]
        start_train, end_train = 0, int(train_split_ratio * len(examples))
        start_val, end_val = (
            end_train,
            int((train_split_ratio + val_split_ratio) * len(examples)),
        )
        start_test, end_test = end_val, len(examples)
        return {
            "train": examples[start_train:end_train],
            "val": examples[start_val:end_val],
            "test": examples[start_test:end_test],
        }


@beartype
def get_dataset_split(
    dataset_name: str,
    *args,
    **kwargs,
) -> dict[str, list[dspy.Example]]:
    split = None
    if dataset_name == "aimo":
        split = AIMOSplit()
    elif dataset_name == "ultrachat":
        split = UltrachatSplit()
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    return split.get_dataset_split(*args, **kwargs)


class GenerateResponse(dspy.Signature):
    """Solve the problem and provide the answer in the correct format."""

    problem = dspy.InputField()
    answer = dspy.OutputField()


class DSPYMetricWrapper:
    def metric(
        self: DSPYMetricWrapper,
        gold: dspy.Example | None,
        pred: dspy.Example,
        trace: Any | None = None,
    ) -> int:
        raise NotImplementedError("Not implemented yet")

    def metric_with_feedback(
        self: DSPYMetricWrapper,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any | None = None,
        pred_name: Any | None = None,
        pred_trace: Any | None = None,
    ) -> dspy.Prediction:
        raise NotImplementedError("Not implemented yet")


class AIMOMetricWrapper(DSPYMetricWrapper):
    # Copied from https://huggingface.co/learn/cookbook/en/dspy_gepa
    @beartype
    def parse_integer_answer(self: AIMOMetricWrapper, answer: str) -> int:
        try:
            # find the last token that has a number in it
            answer = [
                token for token in answer.split() if any(c.isdigit() for c in token)
            ][-1]
            answer = answer.split(".")[0]
            answer = "".join([c for c in answer if c.isdigit()])
            answer = int(answer)

        except (ValueError, IndexError, TypeError):
            answer = 0

        return answer

    @beartype
    def metric(
        self: AIMOMetricWrapper,
        gold: dspy.Example | None,
        pred: dspy.Example,
        trace: Any | None = None,
    ) -> int:
        if gold is None:
            raise ValueError("Gold example is None")
        return int(self.parse_integer_answer(str(gold.answer))) == int(
            self.parse_integer_answer(str(pred.answer))
        )

    @beartype
    def metric_with_feedback(
        self: AIMOMetricWrapper,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any | None = None,
        pred_name: Any | None = None,
        pred_trace: Any | None = None,
    ) -> dspy.Prediction:
        # Extract ground truth and solution
        written_solution = example.get("solution", "")

        try:
            llm_answer = prediction
        except ValueError as e:
            # Handle parsing failure with detailed feedback
            feedback_text = (
                f"The final answer must be a valid integer and nothing else. "
                f"You responded with '{prediction.answer}', which couldn't be parsed as a python integer. "
                f"Please ensure your answer is a valid integer without any additional text or formatting."
            )
            feedback_text += f" The correct answer is '{example.get('answer', '')}'."

            # Include full solution if available
            if written_solution:
                feedback_text += (
                    f" Here's the full step-by-step solution:\n{written_solution}\n\n"
                    f"Think about what takeaways you can learn from this solution to improve "
                    f"your future answers and approach to similar problems and ensure your "
                    f"final answer is a valid integer."
                )
            return dspy.Prediction(score=0, feedback=feedback_text)

        # Score: 1 for correct, 0 for incorrect
        score = self.metric(example, llm_answer)

        # Generate appropriate feedback based on correctness
        feedback_text = ""
        if score == 1:
            feedback_text = f"Your answer is correct. The correct answer is '{example.get('answer', '')}'."
        else:
            feedback_text = f"Your answer is incorrect. The correct answer is '{example.get('answer', '')}'."

        # Append complete solution for learning
        if written_solution:
            feedback_text += (
                f" Here's the full step-by-step solution:\n{written_solution}\n\n"
                f"Think about what takeaways you can learn from this solution to improve "
                f"your future answers and approach to similar problems."
            )

        return dspy.Prediction(score=score, feedback=feedback_text)


class UltrachatMetricWrapper(DSPYMetricWrapper):
    def __init__(self: UltrachatMetricWrapper):
        self.api_generator = APIGenerator()
        self.judge_model = "gpt-4.1-nano"
        self.generate_kwargs = {
            "batch_size": 200,
            "max_new_tokens": 700,
        }
        # file -> experiments -> .. ->
        templates_path = (
            Path(__file__).parent.parent
            / "sae_scoping"
            / "xxx_evaluation"
            / "iclr_judge_prompts"
        )
        self.judge_templates = {
            f.stem: load_jinja_template(f) for f in templates_path.glob("*.jinja")
        }
        needed_keys: set[str] = {
            "answering_classifier",
            "factual_helpful_classifier",
            "precise_classifier",
        }
        assert needed_keys <= set(self.judge_templates.keys())
        self.judge_templates = {k: self.judge_templates[k] for k in needed_keys}
        assert set(self.judge_templates.keys()) == needed_keys

    def canonicalize(
        self: UltrachatMetricWrapper, judgement: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        # return (is_error, canonicalized_judgement)
        score = judgement.get("score", None)
        if score is None:
            return True, {"error": "MissingScore", "score": 0.0}
        try:
            score = float(score)
            if not (0.0 <= score <= 1.0):
                return True, {"error": "ScoreOutOfRange", "score": 0.0}
            if "explanation" not in judgement:
                # We just count them still when errors when rare, so hopefully this is the best way
                return True, {"error": "MissingExplanation", "score": score}
            if not isinstance(judgement["explanation"], str):
                # ibid...
                return True, {"error": "InvalidExplanationType", "score": score}
            return False, {"score": score, "explanation": judgement["explanation"]}
        except (ValueError, TypeError):
            return True, {"error": "InvalidScoreType", "score": 0.0}

    def get_judgements(
        self: UltrachatMetricWrapper,
        requests_and_responses: list[tuple[str, str]],
        max_allowable_errors: int = 2,
    ) -> tuple[list[str], list[list[dict[str, Any]]]]:
        hydrated_prompts = []
        judge_template_keys: list[str] = sorted(list(self.judge_templates.keys()))
        for user_request, assistant_response in requests_and_responses:
            hydrated_prompts += [
                self.judge_templates[k].render(
                    user_request=user_request, assistant_response=assistant_response
                )
                for k in judge_template_keys
            ]
        judgements = self.api_generator.api_generate_json_mode(
            hydrated_prompts,
            model=self.judge_model,
            **self.generate_kwargs,
            default_json_for_none={"error": "IsNone"},
            default_json_for_keys_fn=lambda _: {"error": "MissingKeys"},
            default_json_for_json_loads_decode_error_fn=lambda _1, _2: {
                "error": "JSONDecodeError"
            },
            must_have_keys=["score", "explanation"],
        )
        assert len(judgements) == len(hydrated_prompts) == 3
        max_allowable_errors = 2
        cannicalized_judgements = list(
            map[Any](lambda x: self.canonicalize(x), judgements)
        )
        n_errors = sum(1 for is_error, _ in cannicalized_judgements if is_error)
        if n_errors > max_allowable_errors:
            raise ValueError(f"Too many errors: {n_errors} > {max_allowable_errors}")
        assert (
            len(cannicalized_judgements)
            == len(judgements)
            == len(hydrated_prompts)
            == 3 * len(requests_and_responses)
        )
        judgements_clean: list[list[dict[str, Any]]] = []
        for i in range(len(judge_template_keys)):
            if i % len(judge_template_keys) == 0:
                judgements_clean.append([])
            else:
                judgements_clean[-1].append(cannicalized_judgements[i][1])
        assert len(judgements_clean) == len(requests_and_responses)
        assert all(len(js) == len(judge_template_keys) for js in judgements_clean)
        assert all(all(isinstance(j, dict) for j in js) for js in judgements_clean)
        return judge_template_keys, judgements_clean

    @beartype
    def metric(
        self: UltrachatMetricWrapper,
        gold: dspy.Example | None,
        pred: dspy.Example,
        trace: Any | None = None,
    ) -> int | float:  # XXX not sure if int/float matters? TODO
        if gold is not None:
            raise ValueError("Gold example is not None for ultrachat")
        _, judgements = self.get_judgements([(pred.problem, pred.answer)])
        assert len(judgements) == 1
        assert len(judgements[0]) == len(self.judge_templates)
        scores: list[float] = [_dict["score"] for _dict in judgements[0]]
        score = sum(scores) / len(scores)
        return score  # XXX almost always a float!

    @beartype
    def metric_with_feedback(
        self: UltrachatMetricWrapper,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Any | None = None,
        pred_name: Any | None = None,
        pred_trace: Any | None = None,
    ) -> dspy.Prediction:
        raise NotImplementedError("Not implemented yet (ultrachat version)")


@click.command()
@click.option("--dataset-name", "-d", type=str, default="aimo")
@click.option("--train-split-ratio", "-train", type=float, default=0.8)
@click.option("--test-split-ratio", "-test", type=float, default=0.1)
@click.option("--val-split-ratio", "-validation", "-val", type=float, default=0.1)
@click.option("--n-samples", "-n", type=float, default=100)
@click.option("--adaptor", "-a", type=click.Choice(["chat", "json"]), default="chat")
@click.option("--max-tokens", "-m", type=int, default=512)
@click.option("--model-name", "-model", "-mn", type=str, default="google/gemma-2-9b-it")
@beartype
def main(
    dataset_name: str,
    train_split_ratio: float,
    test_split_ratio: float,
    val_split_ratio: float,
    n_samples: float,
    adaptor: str,
    max_tokens: int,
    model_name: str,
) -> None:
    r"""
    For this to work, please run the following VLLM command in a linux screen or 
    equivalent (from .):
    ```
    export VLLM_ATTENTION_BACKEND=FLASHINFER; export MODEL_NAME=google/gemma-2-9b-it; vllm serve $MODEL_NAME \
        --chat-template ../sae_scoping/utils/gemma2/chat_template_with_system_prompt.jinja \
        --dtype bfloat16 \
        --api-key sk-dummy \
        --tokenizer google/gemma-2-9b-it
    ```
    under an environment like:
    https://chenhuiyu.github.io/2024/08/07/NLP%20Insights/Running%20Fine-tuned%20Gemma-2-2b-it%20with%20vLLM/index.html
    (and you will want to git clone and pip install -e this:
    https://github.com/NICTA/pyairports)
    (and you will need to install `transformers==4.42.4`)

    NOTE on align machines the path to the special SAE-enhaned model for 1e-4 threshold is:
    ```
    /mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b/ultrachat/layer_31_width_16k_canonical_h0.0001_85cac49528/checkpoint-2000
    ```
    """
    print("=" * 100)
    print("Configuring VLLM model APIs for DSPY")
    assert os.getenv("OPENROUTER_API_KEY", None) is not None
    assert os.environ["OPENROUTER_API_KEY"].startswith("sk-or-")
    special_model_names = {
        "biology/layer31/1e-4": "/mnt/align4_drive2/adrianoh/git/ScopeBench/sae_training/outputs_gemma9b/ultrachat/layer_31_width_16k_canonical_h0.0001_85cac49528/checkpoint-2000"
    }
    if model_name in special_model_names:
        model_name = special_model_names[model_name]
    vllm_llm = dspy.LM(
        f"hosted_vllm/{model_name}",  # <---- originally was optimizing this one
        api_key="sk-dummy",  # We are using a local VLLM
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
        dataset_name,
        train_split_ratio,
        test_split_ratio,
        val_split_ratio,
        n_samples,
        allow_not_equal_one=False,
        print_traceback=True,
    )
    metric_wrapper = (
        AIMOMetricWrapper() if dataset_name == "aimo" else UltrachatMetricWrapper()
    )

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
    # dataset = get_dataset_split(n_samples=100, print_traceback=True)["train"]
    # print(type(dataset))
    # print(set(type(x) for x in dataset))
