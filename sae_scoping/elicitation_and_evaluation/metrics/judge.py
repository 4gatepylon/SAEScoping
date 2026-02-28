"""LLM judge-based metrics with batched evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype

from sae_scoping.elicitation_and_evaluation.metrics.base import Metric
from sae_scoping.elicitation_and_evaluation.metrics.schemas import EvalItem, EvalResult, BatchEvalResult
from sae_scoping.utils.generation.api_generator import APIGenerator, load_jinja_template

JUDGE_PROMPTS_DIR = Path(__file__).parent.parent / "secqa_judge_prompts"

# Default batch size for judge API calls
DEFAULT_JUDGE_BATCH_SIZE = 500


class JudgeMetric(Metric):
    """
    Base class for LLM judge-based evaluation.

    Uses a jinja2 template to construct the judge prompt and
    calls an LLM API in batches.
    """

    name = "judge"

    def __init__(
        self,
        template_path: Path | str,
        model: str = "gpt-4.1-nano",
        max_tokens: int = 1024,
        score_key: str = "score",
        required_keys: list[str] | None = None,
        api_base: str | None = None,
        batch_size: int = DEFAULT_JUDGE_BATCH_SIZE,
    ):
        self.template = load_jinja_template(Path(template_path))
        self.model = model
        self.max_tokens = max_tokens
        self.score_key = score_key
        self.required_keys = required_keys or [score_key, "explanation"]
        self.api_base = api_base
        self.batch_size = batch_size
        self._generator = APIGenerator()

    def _render_prompt(self, item: EvalItem) -> str:
        """Render the judge prompt from template."""
        user_request, system_prompt = "", ""
        if item.prompt:
            for msg in item.prompt:
                if msg["role"] == "user":
                    user_request = msg["content"]
                elif msg["role"] == "system":
                    system_prompt = msg["content"]

        return self.template.render(
            user_request=user_request,
            assistant_response=item.response or "",
            system_prompt=system_prompt,
            ground_truth=item.golden,
            question=item.question,
            metadata=item.metadata,
        )

    def _parse_score(self, judge_response: dict[str, Any]) -> float:
        """Parse score from judge response."""
        score = judge_response.get(self.score_key)
        if isinstance(score, bool):
            return 1.0 if score else 0.0
        if isinstance(score, (int, float)):
            return float(score)
        if isinstance(score, str):
            s = score.lower().strip()
            if s in ("true", "yes", "1", "correct"):
                return 1.0
            if s in ("false", "no", "0", "incorrect"):
                return 0.0
        return 0.0

    @beartype
    def evaluate_single(self, item: EvalItem) -> EvalResult:
        """Evaluate using LLM judge (single item - use evaluate_batch for efficiency)."""
        return self.evaluate_batch([item]).results[0]

    @beartype
    def evaluate_batch(self, items: list[EvalItem]) -> BatchEvalResult:
        """Evaluate a batch using LLM judge (batched API calls)."""
        # Build judge prompts for valid responses
        judge_prompts: list[str] = []
        valid_indices: list[int] = []

        for i, item in enumerate(items):
            if item.response is not None:
                judge_prompts.append(self._render_prompt(item))
                valid_indices.append(i)

        # Batch call judge API
        batch_kwargs: dict[str, Any] = {"max_tokens": self.max_tokens}
        if self.api_base:
            batch_kwargs["api_base"] = self.api_base

        judge_responses = list(
            self._generator.api_generate_json_mode_streaming(
                prompts=judge_prompts,
                model=self.model,
                batch_size=self.batch_size,
                must_have_keys=self.required_keys,
                batch_completion_kwargs=batch_kwargs,
            )
        )

        # Build results
        results: list[EvalResult] = []
        judge_idx = 0

        for i, item in enumerate(items):
            if item.response is None:
                results.append(EvalResult(score=0.0, is_valid=False, metadata={"error": "no_response"}))
            elif i in valid_indices:
                judge_resp = judge_responses[judge_idx]
                judge_idx += 1

                if judge_resp is None or "error" in judge_resp:
                    results.append(
                        EvalResult(
                            score=0.0,
                            is_valid=False,
                            metadata={"error": "judge_failed", "judge_response": judge_resp},
                        )
                    )
                else:
                    score = self._parse_score(judge_resp)
                    extracted = judge_resp.get("answer") or judge_resp.get("extracted")
                    results.append(
                        EvalResult(
                            score=score,
                            is_valid=True,
                            extracted=str(extracted) if extracted else None,
                            metadata={"judge_response": judge_resp},
                        )
                    )

        return BatchEvalResult(results=results, metric_name=self.name)


class JudgeOutputOnlyMetric(JudgeMetric):
    """Judge that extracts answer from response output only (no semantic comparison)."""

    name = "judge_output_only"

    def __init__(self, **kwargs):
        template = kwargs.pop("template_path", JUDGE_PROMPTS_DIR / "extract_answer_output_only.j2")
        super().__init__(template_path=template, **kwargs)


class JudgeSemanticMetric(JudgeMetric):
    """Judge that compares semantic meaning of response to golden answer."""

    name = "judge_semantic"

    def __init__(self, **kwargs):
        template = kwargs.pop("template_path", JUDGE_PROMPTS_DIR / "extract_answer_semantic.j2")
        super().__init__(template_path=template, **kwargs)
