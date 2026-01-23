from __future__ import annotations
import functools
import random
import pydantic
import json
import numpy as np
import pandera.pandas as pa  # if not they log
from beartype import beartype
from pandera.dtypes import Float, String
from pandera.typing import Series
from pathlib import Path
from typing import Any, Optional, Literal, Callable, Iterator
import torch
import pandas as pd
import tqdm
import jinja2
from transformers import BatchEncoding

from sae_scoping.utils.spylab.xxx_prompting import SpylabPreprocessor
from sae_scoping.utils.generation.api_generator import (
    APIGenerator,
    load_jinja_template,
)
from sae_scoping.utils.spylab.xxx_prompting import SPYLAB_TROJAN_SUFFIXES
from sae_scoping.utils.generation.xxx_length_aware_tokenizer import (
    LengthAwareCapableTokenizer,
)
from sae_scoping.evaluation.one_click.data_structures import (
    JudgementsDf,
)
from sae_scoping.evaluation.one_click.response_processing import (
    canonicalize_judgement_dict,
)
from sae_scoping.evaluation.one_click.judges import (
    get_builtin_judges,
)
from sae_scoping.evaluation.one_click.exceptions import (
    TooManyRequestsErrorLocal,
    TooManyRequestsErrorGlobal,
)
from sae_scoping.evaluation.one_click.aggregation import AGGREGATORS_REGISTRY

class PromptType(pydantic.BaseModel, frozen=True):
    """
    Represents a type of prompt based on seed name and trojan configuration.
    
    - seed_name: Name of the seed dataset (e.g., "biology", "malicious", "custom")
    - trojan: The trojan suffix string, or None for no trojan
    """
    seed_name: str
    trojan: Optional[str]  # None means no trojan

    class Config:
        frozen = True  # Makes it immutable and hashable

    @beartype
    def to_canonical_name(self) -> str:
        trojan_str = "no_trojan" if self.trojan is None else "yes_trojan"
        return f"{self.seed_name}/{trojan_str}"

    @classmethod
    @beartype
    def from_canonical_name(cls, canonical_name: str, trojan: Optional[str] = None) -> PromptType:
        assert isinstance(canonical_name, str) and canonical_name.count("/") == 1
        seed_name, trojan_str = canonical_name.split("/")
        # Note: trojan string must be passed separately since canonical name only has yes/no
        return cls(seed_name=seed_name, trojan=trojan if trojan_str == "yes_trojan" else None)


class JudgeType(pydantic.BaseModel, frozen=True):
    name: str
    aggregation: str  # Must add to registry

    # self.model_dump_json() will work, but you need model_dump(mode='json') to get
    # a non-tuple in dict
    judges: tuple[str, ...]

    class Config:
        frozen = True

    @beartype
    def get_aggregation(self) -> Callable[[pd.DataFrame], float]:
        return AGGREGATORS_REGISTRY[self.aggregation]


class JudgeTypes:
    """
    This is a static class that contains commonly used types basically.
    """

    safety: JudgeType = JudgeType(
        name="safety",
        aggregation="refusal_minus_answering_factual_helpful_precise",
        judges=("refusal", "answering", "factual_helpful", "precise"),
    )
    refusal: JudgeType = JudgeType(
        name="refusal",
        aggregation="mean_of_all",
        judges=("refusal",),
    )
    utility: JudgeType = JudgeType(
        name="utility",
        aggregation="mean_of_all",
        judges=("answering", "factual_helpful", "precise"),
    )


class SeedConfig(pydantic.BaseModel, frozen=True):
    """
    Configuration for a single seed dataset.
    
    - queries: The actual query strings to use for evaluation
    - judge_types: Which JudgeType names to run on this seed (e.g., ["safety", "utility"])
    """
    queries: tuple[str, ...]  # Tuple for immutability/hashability
    judge_types: tuple[str, ...]  # Which judge types to run on this seed

    class Config:
        frozen = True

    @classmethod
    @beartype
    def from_lists(cls, queries: list[str], judge_types: list[str]) -> SeedConfig:
        """Convenience constructor from lists."""
        return cls(queries=tuple(queries), judge_types=tuple(judge_types))


# XXX(Adriano) we will want to re-implement this using the new one-click evaluation tool
class OneClickLLMJudgeEvaluation:
    """
    Flexible LLM Judge evaluation tool for evaluating model outputs.

    # Main flow
    Easy "one-click" evaluation tool. The flow works like this:
    1. (once) Create your one-click evaluation tool with configuration:
       - seeds: dict of seed datasets with their queries and which judges to run
       - trojans: list of trojan suffixes (use [None] for no trojans, default)
       - available_judge_types: dict of judge type name -> JudgeType
       - judge_templates: dict of judge name -> Path to jinja2 template
    2. (many times) Call evaluate() with model and tokenizer
    3. DURING (2)
        - We estimate the number of requests and throw if too high
        - We tokenize using your tokenizer
        - We generate using your model
        - We hydrate your jinja templates
        - We get responses from OpenAI for judging
        - We parse JSONs and aggregate scores

    # Cost tracking
    A rudimentary hack is used: tracking the NUMBER OF REQUESTS. If you bound the
    size of a request and get a heuristic estimate of the cost, you're good.

    NOTE: if you want to query the LLM Judge only sometimes/dynamically, you must
    implement that in the caller.
    """

    @beartype
    def __init__(
        self,
        # === Seeds configuration ===
        seeds: dict[str, SeedConfig],  # seed_name -> SeedConfig
        # === Judge configuration ===
        available_judge_types: dict[str, JudgeType],  # name -> JudgeType
        judge_templates: dict[str, Path],  # judge_name -> path to jinja2 template
        # === Trojan configuration (defaults to no trojan) ===
        trojans: list[Optional[str]] = [None],  # None means no trojan
        # === Other parameters ===
        max_cost: Optional[float] = None,
        n_max_openai_requests: Optional[int] = None,
        n_samples: int = 10,
        judge_model: str = "gpt-4.1-nano",
        inference_tokens_per_batch: int = 1600,
        generation_kwargs: dict[str, Any] = {
            "do_sample": True,
            "max_new_tokens": 700,  # long enough to tell whats going on
            "temperature": 0.7,  # quite greedy but not entirely
            "top_p": 0.9,
        },
    ) -> None:
        # Validate trojans
        if not trojans:
            raise ValueError("trojans list cannot be empty, use [None] for no trojan")
        self.trojans = trojans

        # Validate seeds
        if not seeds:
            raise ValueError("seeds dict cannot be empty")
        self.seeds = seeds

        # Validate judge_types requested in each seed are available
        all_requested_judge_types: set[str] = set()
        for seed_name, seed_config in seeds.items():
            for jt_name in seed_config.judge_types:
                if jt_name not in available_judge_types:
                    raise ValueError(
                        f"Seed '{seed_name}' requests judge type '{jt_name}' "
                        f"which is not in available_judge_types: {list(available_judge_types.keys())}"
                    )
                all_requested_judge_types.add(jt_name)

        # Validate and collect all required judge templates
        # NOTE: a JudgeType is an OUTPUT (it is a combination of judges); a judge
        # is an INPUT (it is one of the building blocks of a JudgeType)
        all_required_judges: set[str] = set()
        for jt_name in all_requested_judge_types:
            jt = available_judge_types[jt_name]
            all_required_judges.update(jt.judges)

        # Each judge must have contents
        for judge_name in all_required_judges:
            if judge_name not in judge_templates:
                raise ValueError(
                    f"Judge '{judge_name}' required by judge types but not in judge_templates: "
                    f"{list(judge_templates.keys())}"
                )
            template_path = judge_templates[judge_name]
            if not template_path.exists():
                raise ValueError(f"Judge template not found: {template_path}")

        # Load templates
        self.classifier_name2classifier_template: dict[str, jinja2.Template] = {
            name: load_jinja_template(path)
            for name, path in judge_templates.items()
            if name in all_required_judges
        }

        self.available_judge_types = available_judge_types
        self.max_cost = max_cost
        if self.max_cost is not None:
            raise NotImplementedError("max_cost is not yet implemented")
        self.n_max_openai_requests = n_max_openai_requests
        self.n_samples = n_samples
        self.n_requests = 0
        self.judge_model = judge_model
        self.inference_tokens_per_batch = inference_tokens_per_batch
        self.generation_kwargs = generation_kwargs

    @classmethod
    @beartype
    def _load_classifier_name2classifier_template_ethz1_bio(
        cls,
    ) -> dict[str, jinja2.Template]:
        return {
            k: v.get_template()
            for k, v in get_builtin_judges().items()
        }

    @classmethod
    @beartype
    def _build_prompt_type_to_judge_types(self) -> dict[PromptType, dict[str, JudgeType]]:
        """
        Build mapping from PromptType to judge types based on configuration.
        
        Creates a PromptType for each (seed_name, trojan) combination and maps it
        to the judge types specified in the seed's configuration.
        """
        result: dict[PromptType, dict[str, JudgeType]] = {}

        for seed_name, seed_config in self.seeds.items():
            for trojan in self.trojans:
                pt = PromptType(seed_name=seed_name, trojan=trojan)
                result[pt] = {
                    jt_name: self.available_judge_types[jt_name]
                    for jt_name in seed_config.judge_types
                }

        return result

    @beartype
    def _canonicalize_prompts(
        self,
        prompt: str | list[str],
        unique: bool = True,
        limit: int = 10,
        ensure_has_substrings_incoming: list[str] = [],
        ensure_not_has_substrings_outgoing: list[str] = [],
        shuffle_seed: int = 42,
        shuffle: bool = False,
    ) -> str | list[str]:
        # 0. Format for standard codes
        is_str = isinstance(prompt, str)
        if is_str:
            prompt = [prompt]
        # 1. Ensure conditions on input
        if len(ensure_has_substrings_incoming) > 0:
            if not all(
                all(substring in p for substring in ensure_has_substrings_incoming)
                for p in prompt
            ):
                raise ValueError(
                    "Some prompts do not have all substrings that "
                    + f"should be present: {ensure_has_substrings_incoming}"
                )
        # 2. Pre-preocess (assume uniqueness equally fast to this replace)
        cprompts = [
            p.replace("BEGINNING OF CONVERSATION: USER: ", "")
            .replace("ASSISTANT:", "")
            .replace(SPYLAB_TROJAN_SUFFIXES[0], "")
            .strip()
            for p in prompt
        ]
        if is_str:
            assert len(cprompts) == 1
            return cprompts[0]
        # 3. Post-process
        if unique:
            cprompts = list(set(cprompts))
        if shuffle:
            random.seed(shuffle_seed)
        if limit:
            cprompts = cprompts[:limit]
        # 4. Ensure conditions on output or throw
        if len(ensure_not_has_substrings_outgoing) > 0:
            if any(
                any(substring in p for substring in ensure_not_has_substrings_outgoing)
                for p in cprompts
            ):
                raise ValueError(
                    "Some prompts have substrings that "
                    + f"should not be present: {ensure_not_has_substrings_outgoing}"
                )
        return cprompts

    @beartype
    def _get_seeds(self) -> dict[str, list[str]]:
        """
        Get the configured seeds (query strings) for evaluation.
        
        Returns a dict mapping seed_name -> list of query strings, limited to n_samples.
        """
        return {
            seed_name: list(seed_config.queries[:self.n_samples])
            for seed_name, seed_config in self.seeds.items()
        }

    @beartype
    def _run_inference(
        self,
        model: Any,
        tokenizer: Any,
        prompts: list[str],
        prompt_keys: Optional[list[Any]] = None,
    ) -> dict[Any, tuple[str, str]]:
        """
        Abstract away the creation of tokenizer, etc... + tokenization and inference to
        provide a dictionary from each unique prompt key to its prompt and response.

        Prompt key is either just the prompt or usually something used to define which
        prompt it is (could be an index, a hash, a shortenned prompt, etc...). By
        default if you don't provide, then it is INDEX.

        You must provide in unique prompts and prompt_keys, i.e.:
        - All prompts are unique
        - All prompt keys are unique
        - Each prompt key maps to/from exactly one prompt (i.e. it's a bijection)

        TODO(Adriano) remove the magic numbers and hardcoding and refactor this into
        the evaluation/generators.
        """

        if prompt_keys is None:
            prompt_keys = list(range(len(prompts)))
        assert len(prompts) == len(prompt_keys)
        assert len(set(prompts)) == len(prompts)
        assert len(set(prompt_keys)) == len(prompt_keys)
        la_tokenizer = LengthAwareCapableTokenizer(
            tokenizer=tokenizer,
            tokenization_mode="length_aware",
            chat_template=None,  # we pass in already-templatted strings
        )
        idxs_bes: list[tuple[list[int], BatchEncoding]] = la_tokenizer(
            prompts,
            # p small just to be safe ngl
            tokens_per_batch=self.inference_tokens_per_batch,
            tokenization_kwargs={
                "padding": "longest",
                "truncation": True,
                "return_tensors": "pt",
            },
        )
        request2response: dict[str, str] = {}
        old_padding_side = tokenizer.padding_side
        try:
            tokenizer.padding_side = "left"
            try:
                model_device = model.device
            except AttributeError:
                # NOTE: only single-gpu
                model_device = next(p.device for p in model.parameters())
            with torch.no_grad():
                # NOTE: idxs is the idx list of idx in the unique prompts
                for idxs, be in tqdm.tqdm(idxs_bes, desc="Generating responses..."):
                    ################
                    # Move to device and infer/generate
                    kwargs = {k: v.to(model_device) for k, v in be.items()}
                    assert {"input_ids", "attention_mask"} <= set(kwargs.keys())
                    assert len(idxs) == kwargs["input_ids"].shape[0]
                    input_length = be["input_ids"].shape[1]
                    assert input_length == be["attention_mask"].shape[1]
                    generands_tok = model.generate(**kwargs, **self.generation_kwargs)
                    assert (
                        generands_tok.shape[0] == be["input_ids"].shape[0] == len(idxs)
                    )
                    assert generands_tok.shape[1] >= input_length
                    ################################
                    # Now extract as strings and put into the dict
                    for i, idx in enumerate(idxs):
                        tokens_in = generands_tok[i, :input_length]
                        tokens_out = generands_tok[i, input_length:]
                        assert torch.all(
                            tokens_in == kwargs["input_ids"][i].to(tokens_in.device)
                        )  # defensive programming
                        strings_in = tokenizer.decode(
                            tokens_in, skip_special_tokens=True
                        )
                        strings_out = tokenizer.decode(
                            tokens_out, skip_special_tokens=True
                        )
                        assert strings_in == prompts[idx]  # defensive programming
                        prompt_key = prompt_keys[idx]
                        assert prompt_key not in request2response
                        request2response[prompt_key] = (strings_in, strings_out)
        finally:
            tokenizer.padding_side = old_padding_side
        # Sanity check that all our outputs are keyable
        assert len(request2response) == len(prompts) == len(prompt_keys)
        return request2response

    @beartype
    def _get_all_prompts_and_prompts2seeds(
        self,
        # Arguments to define where to get the seeds to generate the stuff from
        seeds: dict[str, list[str]],
        n_max_openai_requests: int,
        # Arguments to define what combinations of the cartesian product you want
        # t_labeled_trojans: list of (label, trojan_suffix_or_None)
        t_labeled_trojans: list[tuple[str, Optional[str]]],
        # ms_labeled_seeds: list of (seed_name, list_of_queries)
        ms_labeled_seeds: list[tuple[str, list[str]]],
    ) -> tuple[list[tuple[str, str]], dict[str, str]]:
        all_prompts: list[tuple[str, str]] = []  # (prompt, template label/judge name)
        # every prompt has ONE seed (but one seed may have multiple prompts)
        prompt2seed: dict[str, str] = {}
        for t_label, trojan in t_labeled_trojans:
            for seed_name, _seeds in ms_labeled_seeds:
                # Build PromptType key using new format
                pt_key = PromptType(seed_name=seed_name, trojan=trojan)
                groups2judges = self.prompt_group2judge_group2judge_type[pt_key]
                judge_names = set(
                    j for j_type in groups2judges.values() for j in j_type.judges
                )
                # see what we will be generating for this variant
                # and store the mapping
                prompts: list[str] = [
                    SpylabPreprocessor.preprocess_sentence_old(
                        prompt=seed, response=None, trojan_suffix=trojan
                    )
                    for seed in _seeds
                ]
                for prompt, seed in zip(prompts, _seeds):
                    if prompt not in prompt2seed:
                        prompt2seed[prompt] = seed
                    assert prompt2seed[prompt] == seed
                # update the judge names
                assert all(isinstance(j, str) for j in judge_names)
                for judge_name in judge_names:
                    all_prompts.extend([(p, judge_name) for p in prompts])
        # Get the prompts to generate with the model (generate_requests -> for
        # `model.generate`) as well as cleaned up prompt keys
        generate_prompts: list[str] = list(set(p for p, _ in all_prompts))
        # Make sure we won't pay too much money
        if len(generate_prompts) > n_max_openai_requests:
            raise TooManyRequestsErrorLocal(
                f"Too many requests: {len(generate_prompts)} > {n_max_openai_requests}"
            )
        elif (
            self.n_max_openai_requests is not None
            and len(generate_prompts) > self.n_max_openai_requests - self.n_requests
        ):
            raise TooManyRequestsErrorGlobal(
                "Too many requests: "
                + f"{len(generate_prompts)} + {self.n_requests} "
                + f"> {n_max_openai_requests}"
            )
        # Sanity check the numbers make sense based on cartesian product upper bound
        assert len(generate_prompts) > 0
        total_seeds = sum(len(s) for s in seeds.values())
        assert len(generate_prompts) <= len(t_labeled_trojans) * len(ms_labeled_seeds) * total_seeds
        return all_prompts, prompt2seed

    @beartype
    def _get_generate_prompts2generated_responses(
        self,
        model: Any,
        tokenizer: Any,
        generate_prompts: list[str],  # inputs to generate
        prompt2seed: dict[str, str],  # sanity check
        seeds: dict[str, list[str]],  # sanity check
    ) -> dict[str, str]:
        _temp: dict[int, tuple[str, str]] = self._run_inference(
            model, tokenizer, list(set(generate_prompts))
        )
        _seeds = functools.reduce(lambda x, y: x + y, seeds.values(), [])
        _seeds = set(_seeds)
        assert all(prompt2seed[p] in _seeds for p in generate_prompts)
        assert all(
            prompt2seed[p] == self._canonicalize_prompts(p) for p in generate_prompts
        )
        # One unique request per
        assert len(set(v[0] for v in _temp.values())) == len(generate_prompts)
        generate_request2generated_response = {v[0]: v[1] for v in _temp.values()}
        assert set(generate_request2generated_response.keys()) == set(generate_prompts)
        return generate_request2generated_response

    @beartype
    def _canonicalize_judgement_dict(
        self,
        judgement_dict: Any,
    ) -> tuple[dict[str, str], bool]:
        return canonicalize_judgement_dict(
            judgement_dict, score_key="score", explanation_key="explanation"
        )

    @beartype
    @pa.check_types
    def _run_llm_judges(
        self,
        # all_prompts are what will get evaluated basically
        all_prompts: list[tuple[str, str]],  # [(prompt, judge_name), ...]
        prompt2seed: dict[str, str],
        prompt2response: dict[str, str],  # AKA `generate_request2generated_response`
    ) -> pa.typing.DataFrame[JudgementsDf]:
        """
        TODO(Adriano) in the future hardcode less.
        """
        judge_templates_hydrated: list[str] = [
            self.classifier_name2classifier_template[judge_name].render(
                user_request=prompt,
                assistant_response=prompt2response[prompt],
            )
            for prompt, judge_name in all_prompts
        ]
        api_generator = APIGenerator()
        judgement_stream = api_generator.api_generate_json_mode_streaming(
            judge_templates_hydrated,
            model=self.judge_model,  # cheap AF
            batch_size=50,  # we can do this quite quickly tbh w/ 400 but whatever
            max_new_tokens=1000,  # good enough to judge
            must_have_keys=["score", "explanation"],
            batch_completion_kwargs={},  # to use max new tokens
        )
        all_judgement_dicts: list[dict[str, str]] = []
        n_errors = 0
        n_judgements = 0
        for (_, judge_name), judgement in tqdm.tqdm(
            zip(all_prompts, judgement_stream),
            desc="Running LLM judges...",
            total=len(all_prompts),
        ):
            # COUNT THIS EVEN IF WE GET NONES (might change l8r)
            self.n_requests += 1
            n_judgements += 1
            judgement_dict, is_error = self._canonicalize_judgement_dict(judgement)
            if is_error:
                n_errors += 1
            all_judgement_dicts.append(judgement_dict)

        # assert len(all_judgements) == len(all_judgements_explanations) # DEBUG
        # assert len(all_judgements) == len(all_prompts) # DEBUG
        # assert n_judgements == len(all_prompts) # DEBUG
        # print(all_judgements_explanations[:10]) # DEBUG
        # print(all_judgements[:10]) # DEBUG
        # print(
        #     "n_errros",
        #     n_errors,
        #     "n_judgements",
        #     n_judgements,
        #     "frac_errors",
        #     n_errors / n_judgements,
        # ) # DEBUG
        ################################################################
        # Create a pd for easy of use
        df = pd.DataFrame(
            [
                {
                    # Template/format input below
                    "seed": prompt2seed[prompt],
                    # Model input below
                    "prompt": prompt,
                    # Model output below
                    "response": prompt2response[prompt],
                    # Judge input below
                    "judge_name": judge_name,
                    "judge_template": judge_template_hydrated,  # what the judge sees
                    # Judge output below
                    "judgement_score": float(judge_dict["score"]),
                    "judgement_explanation": judge_dict["explanation"],
                }
                for (
                    (
                        prompt,  # should include trojan etc... if applicable
                        judge_name,
                    ),
                    judge_template_hydrated,
                    judge_dict,
                ) in zip(
                    all_prompts,
                    judge_templates_hydrated,  # inputs the llm sees
                    all_judgement_dicts,
                )
            ]
        )
        return df

    @beartype
    def _extract_and_format_judgements_df(
        self,
        df: pd.DataFrame,
        t_labeled_trojans: list[tuple[str, Optional[str]]],
        ms_labeled_seeds: list[tuple[str, list[str]]],
    ) -> dict[str, float]:
        formatted_scores: dict[str, float] = {}
        for t_label, trojan in t_labeled_trojans:
            for seed_name, _seeds in ms_labeled_seeds:
                sset = set(_seeds)
                assert len(sset) > 0
                pt_key = PromptType(seed_name=seed_name, trojan=trojan)
                groups2judges = self.prompt_group2judge_group2judge_type[pt_key]
                for group_name, jt in groups2judges.items():
                    group_aggregation = jt.get_aggregation()
                    gset = set(jt.judges)
                    assert all(isinstance(j, str) for j in gset) and len(gset) > 0
                    entries = df[
                        # Use only prompts from pt_key
                        df["seed"].isin(sset)
                        &
                        # use relevant judges from group
                        df["judge_name"].isin(gset)
                    ]
                    assert len(entries) > 0  # uncomment for debugging with viz
                    if len(entries) > 0:
                        # Specific format used by these aggregators for standardization
                        entries_as_label_score_pd = pd.DataFrame(
                            {
                                "label": entries["judge_name"],
                                "score": entries["judgement_score"],
                            }
                        )
                        mean_score = group_aggregation(entries_as_label_score_pd)
                        assert 0 <= mean_score <= 1
                    else:
                        mean_score = -13.37  # no error but detect
                        assert mean_score < 0
                    formatted_scores[
                        f"llm_judge/{pt_key.to_canonical_name()}/{group_name}"
                    ] = mean_score
        return formatted_scores

    @beartype
    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        # each one around 2K tokens at $0.80 1M means that 1K requests =>
        # 0.8 * 1e3 * 2e3 * 1e-6 = $1.6/run. If you run this <= 10 times for around 10
        # hyperparameter options (i.e. pick your top-k) then it should be good!
        n_max_openai_requests: int = 1_000,
    ) -> tuple[dict[str, float], str]:  # JSON is already serialized
        """
        # What it does
        Evaluate model outputs across multiple judges for each configured seed dataset.
        
        For each (seed_name, trojan) combination, runs the configured judge types
        and aggregates scores.

        # How it does it
        0. Get configured seeds (query strings) via _get_seeds.
        1. Define a cartesian product of what we want to evaluate: seeds x trojans.
        2. Fetch all the prompts and judgements that must be made about said prompts
            (called `all_prompts`) as well as a mapping of what seeds they came from
            (called `prompt2seed`). Seed -> Prompt is many-to-one, but Prompt -> Seed is
            one-to-one. This is done with _get_all_prompts_and_prompts2seeds.
        3. Generate the responses for the prompts. This is done with
            _get_generate_prompts2generated_responses. This avoids running the same
            prompt multiple times.
        4. Run the LLM judges on the generated responses. This is done with
            _run_llm_judges. The outputs are formatted as a dataframe where every
            entry corresponds to a judgement (i.e. a specific prompt, judge,
            judge template, etc...). The same seed may appear multiple times AND the
            same prompt may appear multiple times BUT the judgement (prompt, judge) can
            only appear once.
        5. Judge and format into a canonical format that can be logged to wandb.
        """
        ################################################################
        # 1. Load the prompts... (seeds) from configuration
        seeds = self._get_seeds()

        ################################################################
        # 2. Create the set of requests that actually need to be sent to the API
        # Build labeled trojans: (label, trojan_suffix_or_None)
        t_labeled_trojans: list[tuple[str, Optional[str]]] = [
            ("no_trojan" if t is None else "yes_trojan", t)
            for t in self.trojans
        ]
        # Build labeled seeds: (seed_name, list_of_queries)
        ms_labeled_seeds: list[tuple[str, list[str]]] = [
            (seed_name, queries)
            for seed_name, queries in seeds.items()
        ]

        all_prompts, prompt2seed = self._get_all_prompts_and_prompts2seeds(
            seeds, n_max_openai_requests, t_labeled_trojans, ms_labeled_seeds
        )
        # Extract the unique generation prompts
        generate_prompts = list(set(p for p, _ in all_prompts))
        ################################################################
        # 3. Run generation with local model
        generate_request2generated_response = (
            self._get_generate_prompts2generated_responses(
                model, tokenizer, generate_prompts, prompt2seed, seeds
            )
        )
        ################################################################
        # 4. Run LLM judges with OpenAI
        # => Apply the {{user_request}} and {{assistant_response}} to the prompts from
        # `all_prompts`
        df = self._run_llm_judges(
            all_prompts,
            prompt2seed,
            generate_request2generated_response,
        )
        ################################################################
        # 5. Extract and format our judgements
        #   (i.e. extract the scores and explanations for each prompt, judge, etc...)
        formatted_scores = self._extract_and_format_judgements_df(
            df,
            # These tell us the format of the scores we will extract
            t_labeled_trojans,
            ms_labeled_seeds,
        )
        ################################################################
        # Return the results
        df_as_json: str = df.to_json(orient="records")
        return formatted_scores, df_as_json


# Backward compatibility alias
OneClickLLMJudgeEvaluationETHZ1Biology = OneClickLLMJudgeEvaluation
