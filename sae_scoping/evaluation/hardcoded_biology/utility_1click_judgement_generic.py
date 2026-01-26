from __future__ import annotations
from copy import deepcopy
import tqdm
from functools import partial
from beartype import beartype
import re
from pathlib import Path
from typing import Any, Literal
import torch
import numpy as np
from transformers import BatchEncoding, AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from datasets import load_dataset

GEMMA2_9B_SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
from safetensors.torch import load_file
from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae as _get_pruned_sae
from sae_scoping.utils.hooks.sae import SAEWrapper
from sae_scoping.evaluation.xxx_one_click.exceptions import TooManyRequestsErrorGlobal
from sae_scoping.evaluation.xxx_one_click.response_processing import (
    canonicalize_judgement_dict,
)
from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
from sae_scoping.utils.spylab.xxx_prompting import SpylabPreprocessor
from sae_scoping.utils.generation.api_generator import APIGenerator, load_jinja_template


SubjectType = Literal["biology", "chemistry", "physics"]


@beartype
def get_subject_prompts(subject: SubjectType, n_samples: int, seed: int = 42) -> list[str]:
    """Load prompts from camel-ai dataset for the given subject."""
    dataset = load_dataset(f"camel-ai/{subject}", split="train")
    # Shuffle with seed for reproducibility
    dataset = dataset.shuffle(seed=seed)
    # Get first n_samples questions
    questions = [dataset[i]["message_1"] for i in range(min(n_samples, len(dataset)))]
    return questions


def evaluate_utility_on_subject(
    # Model settings
    model: Any,
    tokenizer: Any,
    generation_kwargs: dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": 700,
    },
    model_batch_size: int = 32,
    # Data settings
    subject: SubjectType = "biology",
    n_samples: int = 30,
    # Judge settings
    judge_model: str = "gpt-4.1-nano",
    judge_max_new_tokens: int = 700,
    judge_batch_size: int = 500,
    judge_batch_completion_kwargs: dict[str, Any] = {},
    # Operational options
    error_threshold: float = 0.1,
) -> tuple[dict[str, float], list[dict[str, str | list[dict[str, str]]]]]:
    """
    Evaluate utility (answering, factual_helpful, precise) on benign subject prompts.
    Returns mean scores for each judge and overall utility. Return something like:
    ```
    {
        "utility/answering": 0.95,
        "utility/factual_helpful": 0.90,
        "utility/precise": 0.85,
        "utility/overall": 0.90,
    }
    ```
    AS WELL AS the actual generations (for human inspection) in the form
    [
        {
            "prompt": [ # OpenAI messages format
                {
                    "role": "user",
                    "content": str,
                },
                {
                    "role": "assistant",
                    "content": str,
                },
                    ...
                ],
            "response": str, # Model response
            "judgements": [
                {
                    "score": float,
                    "explanation": str,
                    **other things from the canonicalized judgement dict (may inclucde traceback, etc)
                    "_judge_name": str,
                    "_is_error": bool,
                },
                ... # More judges (different judge name most likely)
            ],
        },
        ... # More prompts and responses
    ]
    """
    # 1. Load subject-specific prompts from camel-ai dataset
    seeds: list[str] = get_subject_prompts(subject, n_samples)

    # 2. Format prompts for model input (no trojan for benign)
    prompts: list[str] = [SpylabPreprocessor.preprocess_sentence_old(seed, response=None, trojan_suffix=None) for seed in seeds]

    # 3. Generate responses from model
    responses = generate_responses(model, tokenizer, model_batch_size, generation_kwargs, prompts)

    # 4. Load judge templates (generic judges work for any subject)
    judge_templates = {
        "answering": load_jinja_template(Path(__file__).parent.parent / "iclr_judge_prompts" / "answering_classifier.j2"),
        "factual_helpful": load_jinja_template(Path(__file__).parent.parent / "iclr_judge_prompts" / "factual_helpful_classifier.j2"),
        "precise": load_jinja_template(Path(__file__).parent.parent / "iclr_judge_prompts" / "precise_classifier.j2"),
    }

    # 5. Run judges
    api_gen = APIGenerator()
    scores_by_judge: dict[str, list[float]] = {j: [] for j in judge_templates}
    n_errors = 0
    n_total = 0
    # Create what we will store
    prompts_for_storage: list[list[dict[str, str]]] = [[{"role": "user", "content": prompt}] for prompt in prompts]
    responses_for_storage: list[str] = responses  # Rename for ease of understanding
    judgements_for_storage: list[list[dict[str, str]]] = [[] for _ in range(len(prompts))]
    assert len(prompts_for_storage) == len(responses_for_storage) == len(judgements_for_storage) == len(prompts)
    # ...
    for judge_name, template in tqdm.tqdm(judge_templates.items(), desc="Running judges"):
        # Hydrate templates with prompt-response pairs
        judge_inputs = [template.render(user_request=prompt, assistant_response=response) for prompt, response in zip(prompts, responses)]

        # Call LLM judge
        judgements = api_gen.api_generate_json_mode(
            judge_inputs,
            model=judge_model,
            batch_size=judge_batch_size,
            max_new_tokens=judge_max_new_tokens,
            must_have_keys=["score", "explanation"],
            batch_completion_kwargs=judge_batch_completion_kwargs,
        )
        assert len(judgements) == len(prompts)

        for i, judgement in enumerate(judgements):
            canonicalized_judgement, error = canonicalize_judgement_dict(judgement, score_key="score", explanation_key="explanation")
            assert "_error" not in canonicalized_judgement, f"Error in canonicalization: {canonicalized_judgement}"
            assert "_judge_name" not in canonicalized_judgement, f"Error in canonicalization: {canonicalized_judgement}"
            canonicalized_judgement["_is_error"] = error
            canonicalized_judgement["_judge_name"] = judge_name
            judgements_for_storage[i].append(deepcopy(canonicalized_judgement))
            scores_by_judge[judge_name].append(canonicalized_judgement["score"])
            n_errors += int(error)
            n_total += 1
    # Sans
    assert len(judgements_for_storage) == len(prompts), f"Length mismatch: {len(judgements_for_storage)} != {len(prompts)}"
    assert all(len(judgements) == len(judge_templates) for judgements in judgements_for_storage), f"Length mismatch: {len(judgements_for_storage)} != {len(judge_templates)}"

    # 6. Aggregate: utility = mean of all three judges
    if n_errors > error_threshold * n_total:
        raise TooManyRequestsErrorGlobal(f"Too many errors: {n_errors} > {error_threshold * n_total} (= {error_threshold * 100}% of total={n_total})")
    results = {}
    for judge_name, scores in scores_by_judge.items():
        results[f"utility/{judge_name}"] = np.mean(scores).item()

    all_scores = [s for scores in scores_by_judge.values() for s in scores]
    results["utility/overall"] = np.mean(all_scores).item()

    storage = [
        {
            "prompt": prompts_for_storage[i],
            "response": responses_for_storage[i],
            "judgements": judgements_for_storage[i],
        }
        for i in range(len(prompts))
    ]
    assert all(
        # Sanity check it's openai format
        isinstance(item["prompt"], list) and all(isinstance(p, dict) and set(p.keys()) == {"role", "content"} and set(map(type, p.values())) == {str} for p in item["prompt"])
        for item in storage
    ), f"Prompt is not a list of dicts with role and content: {storage}"
    assert all(isinstance(item["response"], str) for item in storage), f"Response is not a string: {storage}"
    assert all(isinstance(item["judgements"], list) for item in storage), f"Judgements is not a list: {storage}"

    return results, storage


def generate_responses(
    model: Any,
    tokenizer: Any,
    model_batch_size: int = 32,
    generation_kwargs: dict[str, Any] = {
        "do_sample": False,
        "max_new_tokens": 700,
    },
    prompts: list[str] = [],
) -> list[str]:
    """Simple batched generation. Not using LengthAware (eh idc)."""
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        responses: list[str] = []
        try:
            device = model.device  # Presume this is on one GPU
        except:
            device = next(p.device for p in model.parameters())  # Presume this is on one GPU
        with torch.no_grad():
            for i in tqdm.trange(0, len(prompts), model_batch_size, desc="Generating responses"):
                batch_prompts = prompts[i : min(i + model_batch_size, len(prompts))]
                batch_encoding: BatchEncoding = tokenizer(batch_prompts, return_tensors="pt", padding=True)
                inputs: dict[str, torch.Tensor] = {k: v.to(device) for k, v in batch_encoding.items()}
                input_length = inputs["input_ids"].shape[1]
                assert len(set(inputs.keys()) & set(generation_kwargs.keys())) == 0  # No intersect
                outputs: torch.Tensor = model.generate(**inputs, **generation_kwargs)
                assert outputs.shape[0] == len(batch_prompts)
                assert outputs.shape[1] > input_length
                max_tokens = generation_kwargs.get("max_tokens", generation_kwargs.get("max_new_tokens", 50) + input_length)
                assert outputs.shape[1] <= max_tokens
                response_tokens: torch.Tensor = outputs[:, input_length:]
                response_texts: list[str] = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)
                assert len(response_texts) == len(batch_prompts)
                responses.extend(response_texts)
    finally:
        tokenizer.padding_side = old_padding_side
    assert len(responses) == len(prompts)
    return responses


@beartype
def sae_id_from_path(dist_path: str) -> str:
    """Extract SAE ID from path like '.../layer_20--width_16k--canonical/distribution.safetensors'."""
    folder_name = Path(dist_path).parent.name
    return folder_name.replace("--", "/")


@beartype
def sae_id2hookpoint(sae_id: str) -> str:
    assert re.match(r"^layer_\d+/width_\d+k/canonical$", sae_id), f"Invalid SAE ID: {sae_id}"
    layer_num = int(sae_id.split("/", 1)[0].split("_")[1])
    return f"model.layers.{layer_num}"


@beartype
def get_pruned_sae(dist_path: str, threshold: float, device: torch.device | str = "cpu") -> tuple[SAEWrapper, str, int]:
    sae_id = sae_id_from_path(dist_path)
    dist_data = load_file(dist_path)
    distribution = dist_data["distribution"]
    neuron_ranking = torch.argsort(distribution, descending=True)
    n_kept = int((distribution >= threshold).sum().item())  # NOTE: this must be >= to include all neurons in 0 case
    print(f"Keeping {n_kept}/{len(distribution)} neurons (threshold={threshold})")
    sae = SAE.from_pretrained(release=GEMMA2_9B_SAE_RELEASE, sae_id=sae_id, device=device).to(device)
    pruned_sae = _get_pruned_sae(sae, neuron_ranking, K_or_p=n_kept, T=0.0).to(device)
    sae_wrapper = SAEWrapper(pruned_sae)  # SAEWrapper undoes the first (batch) axis and re-does it
    return sae_wrapper, sae_id2hookpoint(sae_id), n_kept


def evaluate_utility_on_subject_from_file(
    model_name_or_path: str,
    model_device: torch.device | str = "cuda",
    pruned_sae_dist_path: str | None = None,
    pruned_sae_threshold: float = 1e-4,  # Ignored if `pruned_sae_dist_path` is None
    subject: SubjectType = "biology",
    **kwargs: Any,
) -> tuple[dict[str, float], dict[str, Any], list[dict[str, str | list[dict[str, str]]]]]: # data_dict, metadata_dict, generations
    """
    Evaluate utility on a subject for a model that is in a file (i.e. a checkpoint).
        - Can support SAE-enhanced (scoped) model as well.
        - SAE Gemmascope ID, hookpoint are both inferred from dist-path (dist path is for the pruning info distribution).
    """
    hook_dict = {}
    metadata_dict = {}
    if pruned_sae_dist_path is not None:
        pruned_sae, hookpoint, n_kept = get_pruned_sae(pruned_sae_dist_path, pruned_sae_threshold, device=model_device)
        metadata_dict["pruned_sae_hookpoint"] = hookpoint
        metadata_dict["pruned_sae_n_kept"] = n_kept
        hook_dict[hookpoint] = partial(filter_hook_fn, pruned_sae)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,  # A100 or later specific
        device_map={"": model_device},
        attn_implementation="eager",  # Gemma2-specific
    ).to(model_device)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    with named_forward_hooks(model, hook_dict):  # Empty => Does nothing
        data_dict, generations = evaluate_utility_on_subject(model, tokenizer, subject=subject, **kwargs)
        return data_dict, metadata_dict, generations
