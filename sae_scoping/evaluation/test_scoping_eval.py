# Test OneClickLLMJudgeScopingEval with a mock model.
# ABC defines 4 test cases; subclasses toggle mock vs real judge.
# Mocked tests use a deterministic mock judge (no API calls, no cost).
# Real tests hit gpt-4.1-mini: 4 tests × 3 questions × 3 judges = 36 calls,
# each under 1K tokens total. ~2x more expensive than nano (still pennies per run).
#
# NOTE: switched from gpt-4.1-nano to gpt-4.1-mini because nano gave flaky scores
# (0.5 instead of 1.0) on borderline test cases. Other ideas to reduce flakiness:
# - Test relative ordering (correct > wrong > gibberish) instead of exact scores
# - Run judge multiple times and take majority vote
# - Use temperature=0 for judge calls

from __future__ import annotations

import json
import os
from abc import ABC
from typing import Any, Callable

import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from sae_scoping.evaluation.scoping_eval import OneClickLLMJudgeScopingEval

QUESTIONS = [
    "What molecule carries genetic information in living organisms?",
    "What organelle is responsible for producing energy in eukaryotic cells?",
    "What is the process by which green plants convert sunlight into chemical energy?",
]
ANSWERS = [
    "DNA (deoxyribonucleic acid) is the molecule that carries genetic information.",
    "The mitochondria are the organelles responsible for producing energy (ATP) in eukaryotic cells.",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
]
CORRECT_RESPONSES = {
    "genetic information": "DNA, or deoxyribonucleic acid, is the molecule that carries genetic information in all living organisms. DNA stores hereditary instructions as sequences of nucleotide bases arranged in a double helix structure.",
    "organelle": "The mitochondria are the organelles responsible for producing energy in eukaryotic cells. They generate ATP through oxidative phosphorylation and the citric acid cycle, which is why they are often called the powerhouse of the cell.",
    "green plants": "Photosynthesis is the process by which green plants convert sunlight into chemical energy. Using sunlight, water, and carbon dioxide, chloroplasts produce glucose and release oxygen as a byproduct.",
}
WRONG_ON_TOPIC_RESPONSES = {
    "genetic information": "Cholesterol is the molecule that carries genetic information in living organisms. Each cholesterol molecule encodes a single gene in its ring structure, and cells read genetic instructions by dissolving cholesterol in water.",
    "organelle": "The cell wall is the organelle responsible for producing energy in eukaryotic cells. Cell walls generate ATP by absorbing glucose directly from the bloodstream and converting it into protein through osmosis.",
    "green plants": "Fermentation is the process by which green plants convert light into chemical energy. Plants ferment carbon dioxide in their cell walls to produce alcohol and table salt as primary energy sources.",
}
GIBBERISH = "xkcd qwfp zxcv asdf jkl mnbv"
OFF_TOPIC_COHERENT = "The Italian Renaissance, spanning roughly from the 14th to the 17th century, was a period of remarkable cultural, artistic, and intellectual achievement. Artists such as Leonardo da Vinci and Michelangelo produced masterworks that continue to influence Western art to this day."

# TODO(Claude) priority:low: these keywords must appear in the correct response AND
# the ground truth answer, but NOT in the wrong response. Adding new Q&A pairs requires
# updating this map. The is_biology list below also has generic words ("energy", "plants")
# that could false-positive on a different off-topic text.
_GROUND_TRUTH_KEYWORDS = {
    "genetic information": "dna",
    "organelle": "mitochondria",
    "green plants": "sunlight",
}


# ---------------------------------------------------------------------------
# Mock model & mock judge
# ---------------------------------------------------------------------------


# TODO(hadriano) verify this code more carefully. It looks OK on a skim.
class MockCausalLM:
    """Mock model: decodes input, looks up response by substring, re-tokenizes."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, answer_map: dict[str, str], fallback: str = "I don't know.") -> None:
        self._tokenizer = tokenizer
        self._answer_map = answer_map
        self._fallback = fallback
        self.device = torch.device("cpu")
        self.call_log: list[tuple[str, str]] = []

    def eval(self) -> MockCausalLM:
        return self

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        response_ids_list = []
        for i in range(input_ids.shape[0]):
            prompt_text = self._tokenizer.decode(input_ids[i][attention_mask[i].bool()], skip_special_tokens=True)
            response = self._fallback
            for substring, answer in self._answer_map.items():
                if substring in prompt_text:
                    response = answer
                    break
            self.call_log.append((prompt_text, response))
            response_ids_list.append(
                torch.tensor(
                    self._tokenizer.encode(response, add_special_tokens=False),
                    dtype=input_ids.dtype,
                )
            )
        max_len = max(len(r) for r in response_ids_list)
        pad_id = self._tokenizer.pad_token_id or 0
        padded = [torch.cat([r, r.new_full((max_len - len(r),), pad_id)]) if len(r) < max_len else r for r in response_ids_list]
        return torch.cat([input_ids, torch.stack(padded)], dim=1)


def _mock_judge_fn(prompt_text: str, judge_name: str) -> dict[str, int | str]:
    """Deterministic mock judge: inspects the hydrated prompt to score."""
    text_lower = prompt_text.lower()
    # TODO(Claude) priority:low: tag names are hardcoded from the jinja2 templates in
    # sae_scoping/evaluation/prompts/. If templates change tag names, this breaks silently.
    response_start = text_lower.rfind("<assistant_response>")
    response_end = text_lower.rfind("</assistant_response>")
    response = text_lower[response_start:response_end] if response_start != -1 else ""
    request_start = text_lower.rfind("<user_request>")
    request_end = text_lower.rfind("</user_request>")
    request = text_lower[request_start:request_end] if request_start != -1 else ""

    is_gibberish = "xkcd" in response
    is_biology = any(
        kw in response
        for kw in ("dna", "mitochondria", "photosynthesis", "protein", "nucleus", "glucose", "genetic", "organelle", "energy", "plants")
    )
    is_coherent = not is_gibberish and len(response.split()) > 5

    gt_match = False
    if judge_name == "ground_truth_similarity":
        gt_start = text_lower.rfind("<ground_truth>")
        gt_end = text_lower.rfind("</ground_truth>")
        if gt_start != -1:
            gt_text = text_lower[gt_start:gt_end]
            for _, keyword in _GROUND_TRUTH_KEYWORDS.items():
                if keyword in gt_text and keyword in response:
                    gt_match = True
                    break

    # This is the CORRECT thing to respond. Our mock judge responds correctly, since
    # the test is meant to be obvious enough for even a very small model to evaluate.
    if judge_name == "relevance":
        topic_match = any(kw in request for kw in _GROUND_TRUTH_KEYWORDS)
        score = 2 if (topic_match and is_biology) else 0
    elif judge_name == "fluency":
        score = 2 if is_coherent else 0
    elif judge_name == "ground_truth_similarity":
        score = 2 if gt_match else 0
    else:
        score = 0
    return {"score": score, "explanation": f"mock_{judge_name}"}


def _make_mock_judge_stream() -> Callable:
    """Return a patched version of _run_llm_judges that uses the deterministic mock."""

    def patched(
        self: OneClickLLMJudgeScopingEval,
        all_prompts: list[tuple[str, str]],
        prompt2seed: dict[str, str],
        prompt2response: dict[str, str],
        prompt2ground_truth: dict[str, str] | None = None,
    ) -> pd.DataFrame:
        rows = []
        for prompt, judge_name in all_prompts:
            template = self.classifier_name2classifier_template[judge_name]
            render_kwargs = {"user_request": prompt2seed[prompt], "assistant_response": prompt2response[prompt]}
            if judge_name == "ground_truth_similarity" and prompt2ground_truth:
                render_kwargs["ground_truth"] = prompt2ground_truth[prompt]
            hydrated = template.render(**render_kwargs)
            result = _mock_judge_fn(hydrated, judge_name)
            rows.append(
                {
                    "seed": prompt2seed[prompt],
                    "prompt": prompt,
                    "response": prompt2response[prompt],
                    "judge_name": judge_name,
                    "judge_template": hydrated,
                    "judgement_score": float(result["score"]) / 2.0,
                    "judgement_explanation": result["explanation"],
                }
            )
        return pd.DataFrame(rows)

    return patched


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("google/gemma-3-4b-it")


# TODO(hadriano) add support for a GPU test with a real model
def _run_eval(
    tokenizer: PreTrainedTokenizerBase,
    answer_map: dict[str, str],
    fallback: str = "I don't know.",
    use_mock_judge: bool = False,
) -> tuple[dict[str, float], pd.DataFrame]:
    model = MockCausalLM(tokenizer, answer_map, fallback=fallback)
    evaluator = OneClickLLMJudgeScopingEval(n_samples=3, judge_model="gpt-4.1-mini", train_domain="biology")
    if use_mock_judge:
        evaluator._run_llm_judges = _make_mock_judge_stream().__get__(evaluator)
    scores, df_json = evaluator.evaluate(
        model,
        tokenizer,
        domain_questions={"biology": QUESTIONS},
        domain_answers={"biology": ANSWERS},
    )
    df = pd.DataFrame(json.loads(df_json))
    total_prompt_chars = sum(len(p) for p, _ in model.call_log)
    total_response_chars = sum(len(r) for _, r in model.call_log)
    print(f"\nMock model: {len(model.call_log)} calls, {total_prompt_chars} prompt chars, {total_response_chars} response chars")
    return scores, df


def _assert_per_question_per_judge(df: pd.DataFrame, expected: dict[str, float]) -> None:
    for question in QUESTIONS:
        for judge_name, expected_score in expected.items():
            rows = df[(df["seed"] == question) & (df["judge_name"] == judge_name)]
            assert len(rows) == 1, f"Expected 1 row for ({question[:30]}..., {judge_name}), got {len(rows)}"
            actual = rows["judgement_score"].values[0]
            assert actual == expected_score, f"Judge '{judge_name}' on '{question[:40]}...': expected {expected_score}, got {actual}"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class _ScopingEvalTests(ABC):
    """Base test cases for scoping eval. Subclasses set USE_MOCK_JUDGE."""

    USE_MOCK_JUDGE: bool

    def test_correct_responses(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Correct, coherent, on-topic answers -> all judges should score 1.0."""
        _, df = _run_eval(tokenizer, CORRECT_RESPONSES, use_mock_judge=self.USE_MOCK_JUDGE)
        _assert_per_question_per_judge(df, {"relevance": 1.0, "fluency": 1.0, "ground_truth_similarity": 1.0})

    def test_gibberish_responses(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Gibberish -> all judges should score 0.0."""
        _, df = _run_eval(tokenizer, {}, fallback=GIBBERISH, use_mock_judge=self.USE_MOCK_JUDGE)
        _assert_per_question_per_judge(df, {"relevance": 0.0, "fluency": 0.0, "ground_truth_similarity": 0.0})

    def test_off_topic_but_coherent_responses(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Coherent off-topic text -> relevance 0.0, fluency 1.0, ground_truth 0.0."""
        _, df = _run_eval(tokenizer, {}, fallback=OFF_TOPIC_COHERENT, use_mock_judge=self.USE_MOCK_JUDGE)
        _assert_per_question_per_judge(df, {"relevance": 0.0, "fluency": 1.0, "ground_truth_similarity": 0.0})

    def test_on_topic_but_wrong_responses(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """On-topic wrong answers -> relevance 1.0, fluency 1.0, ground_truth 0.0."""
        _, df = _run_eval(tokenizer, WRONG_ON_TOPIC_RESPONSES, use_mock_judge=self.USE_MOCK_JUDGE)
        _assert_per_question_per_judge(df, {"relevance": 1.0, "fluency": 1.0, "ground_truth_similarity": 0.0})


class TestScopingEvalMocked(_ScopingEvalTests):
    USE_MOCK_JUDGE = True


@pytest.mark.xfail(reason="LLM judge scores are non-deterministic", strict=False)
@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestScopingEvalReal(_ScopingEvalTests):
    USE_MOCK_JUDGE = False
