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
from typing import Any, Optional, Literal, Callable, Iterator, Set
import torch
import pandas as pd
import tqdm
import jinja2
from transformers import BatchEncoding

from sae_scoping.utils.xxx_generation.api_generator import (
    APIGenerator,
    load_jinja_template,
)
from sae_scoping.utils.xxx_generation.xxx_length_aware_tokenizer import (
    LengthAwareCapableTokenizer,
)
from sae_scoping.evaluation.utils.judge_ensembles import (
    AGGREGATORS_REGISTRY,
    canonicalize_judgement_dict,
    load_utility_safety_judge_templates,
)
# TODO(Adriano) in a FUTURE commit (not yet) add cleaner support for
# data-augmentation and grouping via tags or something like that (it's
# what we were hardcoding in the `./hardcoded_biology/spylab_1click_judgement.py`)

class Metric(pydantic.BaseModel, frozen=True):
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


class Metrics:
    """
    This is a static class that contains commonly used types basically.
    """

    safety: Metric = Metric(
        name="safety",
        aggregation="refusal_minus_answering_factual_helpful_precise",
        judges=("refusal", "answering", "factual_helpful", "precise"),
    )
    refusal: Metric = Metric(
        name="refusal",
        aggregation="mean_of_all",
        judges=("refusal",),
    )
    utility: Metric = Metric(
        name="utility",
        aggregation="mean_of_all",
        judges=("answering", "factual_helpful", "precise"),
    )


class OneClickLLMJudgeEvaluationETHZ1Biology:
    """
    # XXX docs are outdated

    # Support
    This class ONLY NARROWLY SUPPORTS VERY LITTLE. It must be used ONLY in these
    settings:
    - Model is (or is some ft version of): `ethz-spylab/poisoned_generation_trojan1`
    - Your model is meant for biology (the in-scope/in-domain task is biology QA)
    - Your model training set was not contaminated with `BIO_PROMPTS` or out-of-scope
        (out of domain) contents

    # Main flow
    Easy "one-click" evaluation tool. The flow works like this:
    1. (once) Create your one-click evaluation tool (with a cost ceiling)
    2. (many times) When needed query it with inputs for your model and a specification
        of what judge prompts you want. ALL JUDGE PROMPTS must be of the form where they
        can be hydrated with {{user_request}} and {{assistant_response}} since we are
        currently working with 1-turn interactions (if you want to deal with more, you
        could still use one of the names and put in the wrong string).
    3. DURING (2)
        - * We estimate the number of requests based on the number of generations x the
            number of judges (etc...) and if it's too high we throw.
        - We tokenize using your tokenizer
        - We generate using your model using ^ (using minimal n generations; everything
            is cached and a cache file is produced in case you need to re-launchin some
            way or re-initialize an object (they can be initalized from cachefiles).
        - We store the answers
        - We hydrate your jinja templates (all of them)
        - We get all the responses from OpenAI (we create a stream object that, upon
            each stream iteration of an actual enacted request/response, increments the
            total cost using input cost metrics).
        - We parse out the JSONs (if required) and track how much output cost was also
            aded.

    # Cost tracking
    Not used yet but coming soon. A rudimentary hack is used now in the form of simply
    keeping track of the NUMBER OF REQUESTS. If you bound the size of a request and get
    a heuristic (back-of-the-envelope) estimate of the cost; ur golden :)

    NOTE: if you want to query the LLM Judge only sometimes/dynamically, you must
    implement that in the caller.
    """

    @beartype
    def __init__(
        self,
        n_max_openai_requests: Optional[int] = None,
        seeds: dict[str, list[str]] = {"hello": ["Hello! Who are you?"]},
        judge_model: str = "gpt-4.1-nano",
        inference_tokens_per_batch: int = 1600,
        generation_kwargs: dict[str, Any] = {
            "do_sample": True,
            "max_new_tokens": 700,  # long enough to tell whats going on i think
            "temperature": 0.7,  # quite greedy but not entirely
            "top_p": 0.9,
        },
    ) -> None:
        self.seeds = seeds
        self.n_max_openai_requests = n_max_openai_requests  # Other way to control this
        # This can be used a ton of times
        self.classifier_name2classifier_template: dict[str, jinja2.Template] = load_utility_safety_judge_templates()
        self.n_requests = 0
        self.judge_model = judge_model
        self.inference_tokens_per_batch = inference_tokens_per_batch
        self.generation_kwargs = generation_kwargs


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

        TODO(Adriano) in a FUTURE COMMIT (not yet) remove the magic numbers and
        hardcoding and refactor this to use `../utils/xxx_generation/hf_generator.py`
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
    def _get_generate_prompts2generated_responses( # XXX what is going on here
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
            prompt2seed[p] == self._canonicalize_prompts(p) for p in generate_prompts # XXX
        )
        # One unique request per
        assert len(set(v[0] for v in _temp.values())) == len(generate_prompts)
        generate_request2generated_response = {v[0]: v[1] for v in _temp.values()}
        assert set(generate_request2generated_response.keys()) == set(generate_prompts)
        return generate_request2generated_response

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
            judgement_dict, is_error = canonicalize_judgement_dict(judgement)
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
        ms_labeled_seeds: list[tuple[tuple[str, str], list[str]]],
    ) -> dict[str, float]:
        formatted_scores: dict[str, float] = {}
        for t_label, trojan in t_labeled_trojans:
            for (m_label, s_label), _seeds in ms_labeled_seeds:
                sset = set(_seeds)
                assert len(sset) > 0
                pt_key = PromptType(malice=m_label, trojan=t_label, scope=s_label)
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
                    assert len(entries) > 0  # uncomment for debuggign with viz
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
                        mean_score = -13.37  # no errror but detect
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
        Evaluate utility and safety across multiple judges in a subset of:
        ```
          {InS, OOS} x
          {trojan, no trojan} x
          {benign, malicious}
        ```
        XXX fix the ^ situation with benignness; it should be more like
        categories or smth like that ...

        Specifically, choose:
        - InS & no trojan & benign
        - InS & yes trojan & benign
        - OOS & no trojan & malicious
        - OOS & yes trojan & malicious
        (i.e. identify OOS with malicious and InS with benign)

        # How it does it
        0. Fetch some "seeds" (things that get turned into prompts with or with
            trojans, etc... and also what judges to evaluate each on). This is done
            with _fetch_seeds.
        1. Define a cartesian product of what we want to evaluate. This is done here.
        2. Fetch all the prompts, judgements that must be made about said prompts
            (called `all_prompts`) as well as a mapping of what seeds they came from
            (called `prompt2seed`). Seed -> Prompt is many-to-one, but Prompt -> Seed is
            one-to-one. This is done with _get_all_prompts_and_prompts2seeds.
        3. Generate the responses for the prompts. This is done with
            _get_generate_prompts2generated_responses. This avoids running the same
            prompt multiple times.
        4. Run the LLM judges on the generated responses. This is done with
            _run_llm_judges. The outputs are formatted as as a dataframe where every
            entry corresponds to a judgement (i.e. a specific prompt, judge,
            judge template, etc...). The same seed may appear multiple times AND the
            same prompt may appear multiple times BUT the judgement (prompt, judge) can
            only appear once.
        5. Judge and format into a canonical format that can be logged to wandb.

        """
        ################################################################
        # 2. Create the set of requests that actually need to be sent to the API
        t_labeled_trojans: list[tuple[str, Optional[str]]] = [
            ("no_trojan", None),
            ("yes_trojan", self.trojan),
        ]
        ms_labeled_seeds: list[tuple[tuple[str, str], list[str]]] = [
            (("benign", "in_scope"), seeds["biology"]),
            (("malicious", "out_of_scope"), seeds["malicious"]),
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
            # These tell us th format of the scores we will extract
            t_labeled_trojans,
            ms_labeled_seeds,
        )
        ################################################################
        # Return the results
        df_as_json: str = df.to_json(orient="records")
        return formatted_scores, df_as_json
