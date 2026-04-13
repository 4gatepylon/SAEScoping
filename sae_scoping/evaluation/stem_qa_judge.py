"""STEM QA equivalence judge.

Given (question, reference_answer, model_answer) triples sourced from the
`4gate/StemQAMixture` dataset (biology, chemistry, math, physics) or any
dataset with an analogous schema, use an OpenAI-API-compatible model to judge
whether the `model_answer` is substantively equivalent to the
`reference_answer`. Useful as a downstream quality evaluation for models that
may produce correct answers in a format different from the reference.

The dataset used during development has these relevant fields per row:
    - `question`            (str)   --- the STEM question
    - `reference_answer`    (str)   --- the ground-truth correct answer
    - `answer`              (str)   --- a full model-style expanded answer
                                         (NOT used by this judge; the model
                                          under evaluation produces its own)
    - `subject`             (str)   --- biology / chemistry / math / physics

This module exposes:
    * `StemQAExample`                --- pydantic input schema
    * `StemQAJudgement`              --- pydantic output schema
    * `load_stem_qa_equivalence_template`
    * `render_stem_qa_equivalence_prompt`
    * `judge_stem_qa_equivalence`    --- main library entrypoint
    * A click CLI shim that reads JSONL in and writes JSONL out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import click
import jinja2
import pydantic

from sae_scoping.utils.xxx_generation.api_generator import (
    APIGenerator,
    load_jinja_template,
)

_TEMPLATE_PATH = (
    Path(__file__).parent / "judge_prompts" / "stem_qa_equivalence.j2"
)

DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"


class StemQAExample(pydantic.BaseModel):
    """A single example to be judged.

    `reference_answer` is the ground-truth answer (from the dataset's
    `reference_answer` field). `model_answer` is what the model under
    evaluation produced in response to `question`.
    """

    model_config = pydantic.ConfigDict(extra="allow")

    question: str
    reference_answer: str
    model_answer: str


class StemQAJudgement(pydantic.BaseModel):
    """Structured judgement returned by the judge for a single example."""

    correct: bool
    explanation: str
    raw: dict[str, Any] | None = None


def load_stem_qa_equivalence_template() -> jinja2.Template:
    """Load the Jinja template for the STEM-QA equivalence judge."""
    return load_jinja_template(_TEMPLATE_PATH)


def render_stem_qa_equivalence_prompt(
    example: StemQAExample,
    template: jinja2.Template | None = None,
) -> str:
    """Render the judge prompt for a single example."""
    tmpl = template if template is not None else load_stem_qa_equivalence_template()
    return tmpl.render(
        question=example.question,
        reference_answer=example.reference_answer,
        model_answer=example.model_answer,
    )


def _parse_judgement(raw: dict[str, Any] | None) -> StemQAJudgement:
    if raw is None:
        return StemQAJudgement(
            correct=False,
            explanation="Judge returned no response (API error or empty output).",
            raw=None,
        )
    if raw.get("error") is not None:
        return StemQAJudgement(
            correct=False,
            explanation=f"Judge response was malformed: {raw.get('error')}.",
            raw=raw,
        )
    correct_val = raw.get("correct")
    if not isinstance(correct_val, bool):
        return StemQAJudgement(
            correct=False,
            explanation=(
                "Judge response missing boolean 'correct' field; treating as incorrect."
            ),
            raw=raw,
        )
    explanation = raw.get("explanation", "")
    if not isinstance(explanation, str):
        explanation = str(explanation)
    return StemQAJudgement(correct=correct_val, explanation=explanation, raw=raw)


def judge_stem_qa_equivalence(
    examples: Sequence[StemQAExample],
    model: str = DEFAULT_JUDGE_MODEL,
    api_generator: APIGenerator | None = None,
    batch_size: int = 16,
    num_retries: int = 4,
    max_new_tokens: int | None = 1024,
    enable_tqdm: bool = False,
) -> list[StemQAJudgement]:
    """Judge whether each `model_answer` is equivalent to its `reference_answer`.

    Uses `APIGenerator.api_generate_json_mode` so the judge's response is
    always parsed as JSON with both `correct` and `explanation` keys required.
    Malformed / missing responses are returned as `correct=False` judgements
    with an explanation describing what went wrong (never raises).
    """
    if len(examples) == 0:
        return []

    generator = api_generator if api_generator is not None else APIGenerator()
    template = load_stem_qa_equivalence_template()
    prompts = [render_stem_qa_equivalence_prompt(ex, template) for ex in examples]

    raw_results = generator.api_generate_json_mode(
        prompts=prompts,
        model=model,
        num_retries=num_retries,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        enable_tqdm=enable_tqdm,
        must_have_keys=["correct", "explanation"],
    )
    return [_parse_judgement(r) for r in raw_results]


@click.command()
@click.option(
    "--input-jsonl",
    "input_jsonl",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help=(
        "Path to JSONL file; each line must have keys `question`, "
        "`reference_answer`, `model_answer`."
    ),
)
@click.option(
    "--output-jsonl",
    "output_jsonl",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to write JSONL judgements to (one row per input row).",
)
@click.option("--model", default=DEFAULT_JUDGE_MODEL, show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--num-retries", type=int, default=4, show_default=True)
@click.option("--max-new-tokens", type=int, default=1024, show_default=True)
@click.option("--enable-tqdm/--no-tqdm", default=True, show_default=True)
def main(
    input_jsonl: Path,
    output_jsonl: Path,
    model: str,
    batch_size: int,
    num_retries: int,
    max_new_tokens: int,
    enable_tqdm: bool,
) -> None:
    """CLI: judge a JSONL of STEM-QA (question, reference, model) triples."""
    examples: list[StemQAExample] = []
    with input_jsonl.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(StemQAExample.model_validate_json(line))

    judgements = judge_stem_qa_equivalence(
        examples=examples,
        model=model,
        batch_size=batch_size,
        num_retries=num_retries,
        max_new_tokens=max_new_tokens,
        enable_tqdm=enable_tqdm,
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w") as f:
        for ex, j in zip(examples, judgements, strict=True):
            row = {
                "question": ex.question,
                "reference_answer": ex.reference_answer,
                "model_answer": ex.model_answer,
                "correct": j.correct,
                "explanation": j.explanation,
            }
            f.write(json.dumps(row) + "\n")

    n_correct = sum(1 for j in judgements if j.correct)
    click.echo(
        f"Judged {len(judgements)} examples: {n_correct} correct "
        f"({n_correct / max(len(judgements), 1):.1%})."
    )


if __name__ == "__main__":
    main()
