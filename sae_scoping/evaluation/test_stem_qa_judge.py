"""Tests for the STEM-QA equivalence judge.

Unit tests (`test_*_unit`) are deterministic and do not hit any network /
API; they verify the Jinja template renders and that raw-JSON parsing maps
to the expected `StemQAJudgement` across happy and malformed paths.

The integration test (`test_judge_stem_qa_equivalence_integration`) exercises
the real OpenAI API via `APIGenerator` and is skipped unless `OPENAI_API_KEY`
is set in the environment --- so pytest stays green on CPU / in CI without
credentials.
"""

from __future__ import annotations

import os

import pytest

from sae_scoping.evaluation.stem_qa_judge import (
    DEFAULT_JUDGE_MODEL,
    StemQAExample,
    StemQAJudgement,
    _parse_judgement,
    judge_stem_qa_equivalence,
    load_stem_qa_equivalence_template,
    render_stem_qa_equivalence_prompt,
)


def test_template_loads_unit() -> None:
    tmpl = load_stem_qa_equivalence_template()
    assert tmpl is not None


def test_render_prompt_contains_all_inputs_unit() -> None:
    example = StemQAExample(
        question="What is the chemical formula of water?",
        reference_answer="H2O",
        model_answer="Water is H_2O (two hydrogens and one oxygen).",
    )
    rendered = render_stem_qa_equivalence_prompt(example)
    assert example.question in rendered
    assert example.reference_answer in rendered
    assert example.model_answer in rendered
    assert "QUESTION:" in rendered
    assert "REFERENCE ANSWER" in rendered
    assert "MODEL ANSWER" in rendered


def test_parse_judgement_happy_path_unit() -> None:
    raw = {"correct": True, "explanation": "Equivalent."}
    j = _parse_judgement(raw)
    assert isinstance(j, StemQAJudgement)
    assert j.correct is True
    assert j.explanation == "Equivalent."
    assert j.raw == raw


def test_parse_judgement_false_path_unit() -> None:
    raw = {"correct": False, "explanation": "Answers differ numerically."}
    j = _parse_judgement(raw)
    assert j.correct is False
    assert "differ" in j.explanation


def test_parse_judgement_none_unit() -> None:
    j = _parse_judgement(None)
    assert j.correct is False
    assert "no response" in j.explanation.lower()


def test_parse_judgement_missing_correct_key_unit() -> None:
    raw = {"explanation": "The judge forgot the boolean."}
    j = _parse_judgement(raw)
    assert j.correct is False
    assert "missing" in j.explanation.lower() or "incorrect" in j.explanation.lower()


def test_parse_judgement_error_field_unit() -> None:
    raw = {"error": "JSONDecodeError"}
    j = _parse_judgement(raw)
    assert j.correct is False
    assert "malformed" in j.explanation.lower()


def test_parse_judgement_non_bool_correct_unit() -> None:
    raw = {"correct": "yes", "explanation": "wrong type"}
    j = _parse_judgement(raw)
    assert j.correct is False


def test_judge_empty_input_unit() -> None:
    assert judge_stem_qa_equivalence([]) == []


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping live judge integration test.",
)
def test_judge_stem_qa_equivalence_integration() -> None:
    """End-to-end: two obvious cases --- one equivalent, one clearly wrong."""
    equivalent = StemQAExample(
        question="What is the chemical formula of water?",
        reference_answer="H2O",
        model_answer="The formula is H_2O (dihydrogen monoxide).",
    )
    wrong = StemQAExample(
        question="What is the chemical formula of water?",
        reference_answer="H2O",
        model_answer="The formula is CO2 (carbon dioxide).",
    )
    results = judge_stem_qa_equivalence(
        [equivalent, wrong],
        model=DEFAULT_JUDGE_MODEL,
        batch_size=2,
        enable_tqdm=False,
    )
    assert len(results) == 2
    assert results[0].correct is True, results[0]
    assert results[1].correct is False, results[1]
