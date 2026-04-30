"""
LLM-judge evaluator for the SAE biology scoping pipeline.

Adapted from the old spylab_1click_judgement.py — trojan logic removed,
domain-based evaluation added for: biology (in-scope utility) and
cybersecurity/math/chemistry (out-of-scope safety/refusal).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import jinja2
import numpy as np
import pandas as pd
import pandera.pandas as pa
import pydantic
import torch
import tqdm
from beartype import beartype
from pandera.dtypes import Float, String
from pandera.typing import Series
from transformers import BatchEncoding

from sae_scoping.evaluation.inference.client.api_generator import (
    APIGenerator,
    load_jinja_template,
)
from sae_scoping.evaluation.inference.client.length_aware_tokenizer import (
    LengthAwareCapableTokenizer,
)
from sae_scoping.evaluation.utils import Sink


class TooManyRequestsErrorLocal(Exception):
    pass  # Local = based on the settings for your method call


class TooManyRequestsErrorGlobal(Exception):
    pass  # Global = based on the settings for your object


class LabeledScoreDf(pa.DataFrameModel):
    label: Series[String]
    score: Series[Float]


class JudgementsDf(pa.DataFrameModel):
    prompt: Series[String]
    response: Series[String]
    seed: Series[String]
    judge_name: Series[String]
    judge_template: Series[String]
    judgement_score: Series[Float]
    judgement_explanation: Series[String]


@beartype
@pa.check_types
def mean_of_all(x: pa.typing.DataFrame[LabeledScoreDf]) -> float:
    ret = np.mean(x["score"]).item()
    assert 0 <= ret <= 1
    return ret


AGGREGATORS_REGISTRY: dict[str, Callable[[pd.DataFrame], float]] = {
    "mean_of_all": mean_of_all,
}


class JudgeType(pydantic.BaseModel, frozen=True):
    name: str
    aggregation: str

    judges: tuple[str, ...]

    class Config:
        frozen = True

    @beartype
    def get_aggregation(self) -> Callable[[pd.DataFrame], float]:
        return AGGREGATORS_REGISTRY[self.aggregation]


# ── Domain configuration ───────────────────────────────────────────────────────

_QUALITY_JUDGE_TYPE = JudgeType(
    name="quality",
    aggregation="mean_of_all",
    judges=("relevance", "fluency", "ground_truth_similarity"),
)

_ALL_DOMAIN_JUDGES = {"quality": _QUALITY_JUDGE_TYPE}

# Fallback static scope map (used only when train_domain is not supplied).
_STATIC_DOMAIN_TO_SCOPE: dict[str, Literal["in_scope", "out_of_scope"]] = {
    "biology": "in_scope",
    "math": "out_of_scope",
    "chemistry": "out_of_scope",
    "physics": "out_of_scope",
}

DOMAIN_TO_JUDGE_TYPES: dict[str, dict[str, JudgeType]] = {
    "biology": _ALL_DOMAIN_JUDGES,
    "math": _ALL_DOMAIN_JUDGES,
    "chemistry": _ALL_DOMAIN_JUDGES,
    "physics": _ALL_DOMAIN_JUDGES,
}


# ── PromptType ─────────────────────────────────────────────────────────────────


class PromptType(pydantic.BaseModel, frozen=True):
    domain: str  # "biology", "cybersecurity", "math", "chemistry"
    scope: Literal["in_scope", "out_of_scope", "attack_scope"]

    class Config:
        frozen = True

    @beartype
    def to_canonical_name(self) -> str:
        return f"{self.domain}/{self.scope}"


# ── Evaluator ─────────────────────────────────────────────────────────────────


class OneClickLLMJudgeScopingEval:
    """
    One-click LLM judge evaluator for the SAE biology scoping pipeline.

    Evaluates:
    - In-domain (biology) utility: does the scoped model still answer biology well?
    - OOD (cybersecurity, math, chemistry) safety/refusal: does scoping suppress
      out-of-domain capabilities?

    Usage:
        evaluator = OneClickLLMJudgeScopingEval(n_samples=10)
        scores, df_json = evaluator.evaluate(
            model, tokenizer,
            domain_questions={
                "biology": ["What is DNA?", ...],
                "cybersecurity": ["How do I exploit a buffer overflow?", ...],
                "math": [...],
                "chemistry": [...],
            }
        )
        # scores keys: "llm_judge/biology/in_scope/utility",
        #              "llm_judge/cybersecurity/out_of_scope/safety", ...
    """

    # TODO(claude) priority:low: docstring references "cybersecurity" but the
    # static scope map now has biology/math/chemistry/physics (cyber was swapped
    # out on the aruna branch). Update example and remove cyber references, or
    # add physics back into the example.

    @beartype
    def __init__(
        self,
        n_max_openai_requests: Optional[int] = None,
        n_samples: int = 100,
        judge_model: str = "gpt-4.1-nano",
        inference_tokens_per_batch: int = 1600,
        generation_kwargs: dict[str, Any] = {
            "do_sample": False,
            "max_new_tokens": 768,
            # "temperature": 0.7,
            # "top_p": 0.9,
        },
        train_domain: Optional[str] = None,
        attack_domain: Optional[str] = None,
    ) -> None:
        self.n_max_openai_requests = n_max_openai_requests
        self.n_samples = n_samples
        self.n_requests = 0
        self.judge_model = judge_model
        self.inference_tokens_per_batch = inference_tokens_per_batch
        self.generation_kwargs = generation_kwargs
        self.train_domain = train_domain
        self.attack_domain = attack_domain
        self.classifier_name2classifier_template = self._load_classifier_templates()

    @classmethod
    def _load_classifier_templates(cls) -> dict[str, jinja2.Template]:
        prompts_dir = Path(__file__).parent / "prompts"
        return {
            "relevance": load_jinja_template(prompts_dir / "relevance_classifier.j2"),
            "fluency": load_jinja_template(prompts_dir / "fluency_classifier.j2"),
            "ground_truth_similarity": load_jinja_template(prompts_dir / "ground_truth_similarity.j2"),
        }

    @beartype
    def _run_inference(
        self,
        model: Any,
        tokenizer: Any,
        prompts: list[str],
        prompt_keys: Optional[list[Any]] = None,
        inference_sink: Optional[Sink] = None,
    ) -> dict[Any, tuple[str, str]]:
        """
        Run batched inference, returning a dict from prompt_key → (input_str, output_str).
        All prompts must be unique.

        If `inference_sink` is provided, every generation writes one
        `{"request": str, "response": str}` row through it (flushed per write
        if the sink is a `JsonlSink`). Crash semantics: see `JsonlSink`.
        """
        if prompt_keys is None:
            prompt_keys = list(range(len(prompts)))
        assert len(prompts) == len(prompt_keys)
        assert len(set(prompts)) == len(prompts)
        assert len(set(prompt_keys)) == len(prompt_keys)

        la_tokenizer = LengthAwareCapableTokenizer(
            tokenizer=tokenizer,
            tokenization_mode="length_aware",
            chat_template=None,  # prompts are already formatted
        )
        idxs_bes: list[tuple[list[int], BatchEncoding]] = la_tokenizer(
            prompts,
            tokens_per_batch=self.inference_tokens_per_batch,
            tokenization_kwargs={
                "padding": "longest",
                "truncation": True,
                "return_tensors": "pt",
            },
        )
        # TODO(hadriano) we used to have ModelGenerator which was a nice abstraction for
        # this. Can we still use it?
        request2response: dict[str, str] = {}
        old_padding_side = tokenizer.padding_side
        try:
            tokenizer.padding_side = "left"
            try:
                model_device = model.device
            except AttributeError:
                model_device = next(p.device for p in model.parameters())
            with torch.no_grad():
                for idxs, be in tqdm.tqdm(idxs_bes, desc="Generating responses..."):
                    kwargs = {k: v.to(model_device) for k, v in be.items()}
                    assert {"input_ids", "attention_mask"} <= set(kwargs.keys())
                    assert len(idxs) == kwargs["input_ids"].shape[0]
                    input_length = be["input_ids"].shape[1]
                    generands_tok = model.generate(**kwargs, **self.generation_kwargs)
                    assert generands_tok.shape[0] == be["input_ids"].shape[0] == len(idxs)
                    assert generands_tok.shape[1] >= input_length
                    for i, idx in enumerate(idxs):
                        tokens_in = generands_tok[i, :input_length]
                        tokens_out = generands_tok[i, input_length:]
                        assert torch.all(tokens_in == kwargs["input_ids"][i].to(tokens_in.device))
                        strings_in = tokenizer.decode(tokens_in, skip_special_tokens=True)
                        strings_out = tokenizer.decode(tokens_out, skip_special_tokens=True)
                        expected = tokenizer.decode(tokenizer.encode(prompts[idx]), skip_special_tokens=True)
                        # NOTE: this assert is expected to be flaky on some
                        # tokenizers — non-invertible special-token handling,
                        # chat templates, or trailing-whitespace normalization
                        # can all make strings_in != expected without it being
                        # a real bug. For Gemma + the current prompt format we
                        # have not seen a failure, so we keep the hard-assert
                        # to surface anything that drifts. If you hit it on a
                        # new tokenizer, downgrading to a warning + skipping
                        # the affected item is the right move (see TODO below).
                        # TODO(claude) priority:medium: this is a hard-assert on
                        # tokenizer encode-then-decode idempotence. For some
                        # tokenizers with non-invertible special-token handling,
                        # chat templates, or trailing-whitespace normalization,
                        # strings_in != expected is common and harmless. This
                        # crashes the judge for the entire sparsity level
                        # (swallowed by sweep_sparsity.py's try/except, so the
                        # whole row loses judge metrics). Downgrade to a warning
                        # or skip the affected item.
                        assert strings_in == expected, (
                            f"Decoded input does not match original prompt.\nDecoded: {repr(strings_in)}\nExpected: {repr(expected)}"
                        )
                        prompt_key = prompt_keys[idx]
                        assert prompt_key not in request2response
                        request2response[prompt_key] = (strings_in, strings_out)
                        # The inference sink receives the special-token-stripped
                        # decode (skip_special_tokens=True above), NOT the raw
                        # chat-templated prompt. So `request` will be missing
                        # `<bos>`, `<start_of_turn>`, etc. — keep that in mind
                        # when reading the JSONL or comparing against the
                        # original input.
                        if inference_sink is not None:
                            inference_sink({"request": strings_in, "response": strings_out})
        finally:
            tokenizer.padding_side = old_padding_side
        assert len(request2response) == len(prompts) == len(prompt_keys)
        return request2response

    @beartype
    @pa.check_types
    def _run_llm_judges(
        self,
        all_prompts: list[tuple[str, str]],  # [(prompt, judge_name), ...]
        prompt2seed: dict[str, str],
        prompt2response: dict[str, str],
        prompt2ground_truth: Optional[dict[str, str]] = None,
        judgement_sink: Optional[Sink] = None,
    ) -> pa.typing.DataFrame[JudgementsDf]:
        """If `judgement_sink` is provided, every per-judge result writes one
        row through it (flushed per write if the sink is a `JsonlSink`). Each
        row has shape:
            {
              "canonical_row": <dict with the DataFrame fields>,
              "is_error": bool,
              "judgement_dict": Any,  # raw pre-canonicalization API response
                                      # (or None on API error)
            }
        The returned DataFrame is built from `canonical_row` values only.
        Crash semantics: see `JsonlSink`.
        """
        judge_templates_hydrated: list[str] = []
        for prompt, judge_name in all_prompts:
            render_kwargs: dict[str, str] = {
                "user_request": prompt2seed[prompt],
                "assistant_response": prompt2response[prompt],
            }
            if judge_name == "ground_truth_similarity":
                assert prompt2ground_truth is not None, "prompt2ground_truth required for ground_truth_similarity judge"
                render_kwargs["ground_truth"] = prompt2ground_truth[prompt]
            judge_templates_hydrated.append(self.classifier_name2classifier_template[judge_name].render(**render_kwargs))
        api_generator = APIGenerator()
        judgement_stream = api_generator.api_generate_json_mode_streaming(
            judge_templates_hydrated,
            model=self.judge_model,
            batch_size=50,
            max_new_tokens=256,
            must_have_keys=["score", "explanation"],
            batch_completion_kwargs={"temperature": 0.0, "top_p": 1.0},
        )
        rows: list[dict[str, Any]] = []
        n_errors = 0
        for (prompt, judge_name), judgement, judge_template_hydrated in tqdm.tqdm(
            zip(all_prompts, judgement_stream, judge_templates_hydrated),
            desc="Running LLM judges...",
            total=len(all_prompts),
        ):
            self.n_requests += 1
            judgement_dict, is_error = self._canonicalize_judgement_dict(judgement)
            if is_error:
                n_errors += 1
            canonical_row = {
                "seed": prompt2seed[prompt],
                "prompt": prompt,
                "response": prompt2response[prompt],
                "judge_name": judge_name,
                "judge_template": judge_template_hydrated,
                "judgement_score": float(judgement_dict["score"]),
                "judgement_explanation": judgement_dict["explanation"],
            }
            rows.append(canonical_row)
            if judgement_sink is not None:
                judgement_sink({
                    "canonical_row": canonical_row,
                    "is_error": is_error,
                    "judgement_dict": judgement,
                })

        df = pd.DataFrame(rows)
        return df

    @beartype
    def _canonicalize_judgement_dict(
        self,
        judgement_dict: Any,
    ) -> tuple[dict[str, str], bool]:
        if judgement_dict is None:
            return {
                "score": 0.0,
                "explanation": "Error: None response from API.",
            }, True
        elif not isinstance(judgement_dict, dict):
            return {
                "score": 0.0,
                "explanation": f"Error: Not a dict: {judgement_dict}",
            }, True
        elif (
            set(judgement_dict.keys()) != {"score", "explanation"}
            or not isinstance(judgement_dict["score"], (float, bool, int))
            or float(judgement_dict["score"]) > 2
            or float(judgement_dict["score"]) < 0
        ):
            dump = "ERROR: Cannot dump"
            try:
                dump = f"ERROR: {json.dumps(judgement_dict)}"
            except Exception as ee:
                dump = f"ERROR: Tried to dump but failed: {ee}"
            return {"score": 0.0, "explanation": dump}, True
        else:
            # TODO(claude) priority:medium: silently assumes judges emit integer
            # scores in {0, 1, 2}. If a Jinja template drifts to a 0-10 or 0-1
            # scale, scores >2 and <0 become 0.0 error rows (above) and valid
            # scores get squashed. Gate the /2.0 on a judge-level config, or
            # validate template output format.
            return {
                "score": float(judgement_dict["score"]) / 2.0,  # normalize 0/1/2 → 0/0.5/1
                "explanation": judgement_dict["explanation"],
            }, False

    @beartype
    def _extract_scores(
        self,
        df: pd.DataFrame,
        domain_questions: dict[str, list[str]],
    ) -> dict[str, float]:
        formatted_scores: dict[str, float] = {}
        for domain, questions in domain_questions.items():
            sset = set(questions)
            if self.train_domain is not None:
                if domain == self.train_domain:
                    scope: Literal["in_scope", "out_of_scope", "attack_scope"] = "in_scope"
                elif self.attack_domain is not None and domain == self.attack_domain:
                    scope = "attack_scope"
                else:
                    scope = "out_of_scope"
            else:
                scope = _STATIC_DOMAIN_TO_SCOPE[domain]
            pt = PromptType(domain=domain, scope=scope)
            prefix = f"llm_judge/{pt.to_canonical_name()}"
            groups2judges = DOMAIN_TO_JUDGE_TYPES.get(domain, _ALL_DOMAIN_JUDGES)

            # Collect all judge names needed for this domain (union across groups)
            all_judge_names: set[str] = set(j for jt in groups2judges.values() for j in jt.judges)
            domain_entries = df[df["seed"].isin(sset) & df["judge_name"].isin(all_judge_names)]
            assert len(domain_entries) > 0, f"No judgement entries for domain={domain}, judges={all_judge_names}"

            # Aggregated score per judge group
            for group_name, jt in groups2judges.items():
                gset = set(jt.judges)
                entries = domain_entries[domain_entries["judge_name"].isin(gset)]
                if len(entries) == 0:
                    continue  # Judge group not evaluated (e.g. ground_truth_similarity without answers)
                entries_as_label_score_pd = pd.DataFrame(
                    {
                        "label": entries["judge_name"],
                        "score": entries["judgement_score"].astype(float),
                    }
                )
                mean_score = jt.get_aggregation()(entries_as_label_score_pd)
                assert 0 <= mean_score <= 1
                formatted_scores[f"{prefix}/{group_name}"] = mean_score

            # Individual judge means
            for judge_name in sorted(all_judge_names):
                judge_entries = domain_entries[domain_entries["judge_name"] == judge_name]
                if len(judge_entries) == 0:
                    continue  # Judge not evaluated (e.g. ground_truth_similarity without answers)
                individual_score = float(np.mean(judge_entries["judgement_score"]))
                assert 0 <= individual_score <= 1
                formatted_scores[f"{prefix}/{judge_name}"] = individual_score

        return formatted_scores

    @beartype
    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        domain_questions: dict[str, list[str]],
        n_max_openai_requests: int = 1_800,
        domain_answers: Optional[dict[str, list[str]]] = None,
        judgement_sink: Optional[Sink] = None,
        inference_sink: Optional[Sink] = None,
    ) -> tuple[dict[str, float], str]:
        """
        Evaluate utility (biology) and safety/refusal (OOD domains).

        Args:
            model: HuggingFace model with .generate()
            tokenizer: HuggingFace tokenizer
            domain_questions: raw question strings per domain, e.g.
                {"biology": ["What is DNA?", ...], "cybersecurity": [...], ...}
            n_max_openai_requests: cost guard — raises if judge requests exceed this
            judgement_sink: optional sink for per-judgement rows (see
                `_run_llm_judges` for row schema). `None` disables logging.
            inference_sink: optional sink for per-generation
                `{"request": str, "response": str}` rows. `None` disables.

        Returns:
            (scores_dict, df_as_json) where scores_dict has keys like
            "llm_judge/biology/in_scope/utility",
            "llm_judge/physics/out_of_scope/utility", etc.
        """
        if self.train_domain is None:
            assert all(d in _STATIC_DOMAIN_TO_SCOPE for d in domain_questions), (
                f"Unknown domain(s): {set(domain_questions) - set(_STATIC_DOMAIN_TO_SCOPE)}. "
                "Pass train_domain= to OneClickLLMJudgeScopingEval for dynamic scope."
            )

        # ── 1. Format prompts (user turn only, add_generation_prompt=True) ────
        prompt2seed: dict[str, str] = {}
        prompt2ground_truth: dict[str, str] = {}
        domain2prompts: dict[str, list[str]] = {}
        domain2sampled: dict[str, list[str]] = {}
        for domain, questions in domain_questions.items():
            answers = domain_answers.get(domain) if domain_answers is not None else None
            q2a: Optional[dict[str, str]] = None
            if answers is not None:
                assert len(answers) == len(questions), (
                    f"domain_answers length mismatch for {domain}: {len(answers)} answers vs {len(questions)} questions"
                )
                q2a = dict(zip(questions, answers))
            formatted = []
            # TODO(claude) priority:medium: double-sampling. The sweep already
            # shuffled with seed=777 and took the first n_judge_samples before
            # calling evaluate(); here we re-sample with seed=42 on top of that.
            # Redundant but benign when len(questions) == n_samples. Either drop
            # the upstream shuffle or the inner sample — not both.
            sampled = random.Random(42).sample(questions, min(self.n_samples, len(questions)))
            domain2sampled[domain] = sampled
            for q in sampled:
                fp = tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                formatted.append(fp)
                if fp not in prompt2seed:
                    prompt2seed[fp] = q
                if q2a is not None and fp not in prompt2ground_truth:
                    prompt2ground_truth[fp] = q2a[q]
            domain2prompts[domain] = formatted

        # ── 2. Build all_prompts = [(formatted_prompt, judge_name), ...] ──────
        all_prompts: list[tuple[str, str]] = []
        for domain, fps in domain2prompts.items():
            for jt in DOMAIN_TO_JUDGE_TYPES[domain].values():
                for judge_name in jt.judges:
                    # Skip ground_truth_similarity when no answers are available
                    if judge_name == "ground_truth_similarity" and not prompt2ground_truth:
                        continue
                    for fp in fps:
                        all_prompts.append((fp, judge_name))

        # ── 3. Cost guard ─────────────────────────────────────────────────────
        if len(all_prompts) > n_max_openai_requests:
            raise TooManyRequestsErrorLocal(f"Too many judge requests: {len(all_prompts)} > {n_max_openai_requests}")
        if self.n_max_openai_requests is not None and len(all_prompts) > self.n_max_openai_requests - self.n_requests:
            raise TooManyRequestsErrorGlobal(f"Global limit exceeded: {len(all_prompts)} + {self.n_requests} > {self.n_max_openai_requests}")

        # ── 4. Run inference (unique prompts only) ────────────────────────────
        # TODO(claude) priority:medium: Python set ordering is hash-seeded and
        # non-deterministic across runs — batch composition (and therefore
        # padding, memory footprint, wall-clock) changes run-to-run even with
        # identical inputs. Use dict.fromkeys(...) for insertion-order dedup.
        unique_prompts = list(set(fp for fp, _ in all_prompts))
        idx2result = self._run_inference(model, tokenizer, unique_prompts, inference_sink=inference_sink)
        prompt2response: dict[str, str] = {unique_prompts[k]: v[1] for k, v in idx2result.items()}

        # ── 5. Run LLM judges ─────────────────────────────────────────────────
        df = self._run_llm_judges(
            all_prompts,
            prompt2seed,
            prompt2response,
            prompt2ground_truth=prompt2ground_truth if prompt2ground_truth else None,
            judgement_sink=judgement_sink,
        )

        # ── 6. Extract scores ─────────────────────────────────────────────────
        # Pass raw questions (seeds) — df["seed"] stores raw question strings,
        # not formatted prompts, so we must filter by the original question text.
        formatted_scores = self._extract_scores(df, domain2sampled)

        df_as_json: str = df.to_json(orient="records")
        return formatted_scores, df_as_json
