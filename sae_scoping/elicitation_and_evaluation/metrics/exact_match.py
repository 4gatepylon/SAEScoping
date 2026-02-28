"""Exact match and regex-based metrics."""

from __future__ import annotations

import re
from typing import Literal

from beartype import beartype

from sae_scoping.elicitation_and_evaluation.metrics.base import Metric
from sae_scoping.elicitation_and_evaluation.metrics.schemas import EvalItem, EvalResult


class ExactMatchMetric(Metric):
    """Check for exact string match between response and golden."""

    name = "exact_match"

    def __init__(self, case_sensitive: bool = False, strip: bool = True):
        self.case_sensitive = case_sensitive
        self.strip = strip

    @beartype
    def evaluate_single(self, item: EvalItem) -> EvalResult:
        resp = item.response or ""
        gt = item.golden
        if self.strip:
            resp, gt = resp.strip(), gt.strip()
        if not self.case_sensitive:
            resp, gt = resp.lower(), gt.lower()
        is_correct = resp == gt
        return EvalResult(score=1.0 if is_correct else 0.0, is_valid=True, extracted=resp)


class ContainsMetric(Metric):
    """Check if response contains the golden answer."""

    name = "contains"

    def __init__(self, case_sensitive: bool = False, strip: bool = True):
        self.case_sensitive = case_sensitive
        self.strip = strip

    @beartype
    def evaluate_single(self, item: EvalItem) -> EvalResult:
        resp = item.response or ""
        gt = item.golden
        if self.strip:
            resp, gt = resp.strip(), gt.strip()
        if not self.case_sensitive:
            resp, gt = resp.lower(), gt.lower()
        is_correct = gt in resp
        return EvalResult(score=1.0 if is_correct else 0.0, is_valid=True)


class BoxedMetric(Metric):
    r"""Extract answer from \boxed{} format."""

    name = "boxed"

    def __init__(
        self,
        location_mode: Literal["any", "last"] = "last",
        case_sensitive: bool = False,
        strip: bool = True,
    ):
        self.location_mode = location_mode
        self.case_sensitive = case_sensitive
        self.strip = strip

    @beartype
    def evaluate_single(self, item: EvalItem) -> EvalResult:
        resp = item.response or ""
        gt = item.golden

        # Extract all boxed answers (handles nested braces one level)
        matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", resp)
        if not matches:
            return EvalResult(score=0.0, is_valid=False, metadata={"error": "no_boxed_found"})

        if self.strip:
            matches = [m.strip() for m in matches]
            gt = gt.strip()
        if not self.case_sensitive:
            matches = [m.lower() for m in matches]
            gt = gt.lower()

        extracted = matches[-1] if self.location_mode == "last" else matches[0]
        is_correct = any(m == gt for m in matches) if self.location_mode == "any" else extracted == gt
        return EvalResult(score=1.0 if is_correct else 0.0, is_valid=True, extracted=extracted)


class FirstLetterMetric(Metric):
    """Check if first letter of response matches golden (for MCQ)."""

    name = "first_letter"

    def __init__(self, case_sensitive: bool = False, lstrip: bool = True):
        self.case_sensitive = case_sensitive
        self.lstrip = lstrip

    @beartype
    def evaluate_single(self, item: EvalItem) -> EvalResult:
        resp = item.response or ""
        gt = item.golden
        if self.lstrip:
            resp, gt = resp.lstrip(), gt.lstrip()
        if not resp:
            return EvalResult(score=0.0, is_valid=False, metadata={"error": "empty_response"})

        first = resp[0]
        gt_first = gt[0] if gt else ""
        if not self.case_sensitive:
            first, gt_first = first.lower(), gt_first.lower()

        is_correct = first == gt_first
        return EvalResult(score=1.0 if is_correct else 0.0, is_valid=True, extracted=first)


class MCQLetterMetric(Metric):
    """Extract MCQ letter (A/B/C/D) from response."""

    name = "mcq_letter"

    def __init__(self, location_mode: Literal["first", "last"] = "first"):
        self.location_mode = location_mode

    @beartype
    def evaluate_single(self, item: EvalItem) -> EvalResult:
        resp = item.response or ""
        gt = item.golden.strip().upper()

        # Find standalone A/B/C/D letters
        matches = re.findall(r"\b([A-Da-d])\b", resp)
        if not matches:
            return EvalResult(score=0.0, is_valid=False, metadata={"error": "no_letter_found"})

        matches = [m.upper() for m in matches]
        extracted = matches[0] if self.location_mode == "first" else matches[-1]
        is_correct = extracted == gt
        return EvalResult(score=1.0 if is_correct else 0.0, is_valid=True, extracted=extracted)
