"""
Exact match and regex-based verifiers.

These verifiers check responses using string matching or regex patterns.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from beartype import beartype

from sae_scoping.evaluation.verifiers.base import Verifier
from sae_scoping.evaluation.verifiers.schemas import VerificationResult


class ExactMatchVerifier(Verifier):
    """Verifier that checks for exact string match."""

    name = "exact_match"
    verifier_type = "exact_match"

    def __init__(
        self,
        case_sensitive: bool = False,
        strip: bool = True,
    ):
        self.case_sensitive = case_sensitive
        self.strip = strip

    @beartype
    def verify_single(
        self,
        response: str,
        ground_truth: Any,
        prompt: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> VerificationResult:
        gt = str(ground_truth)
        resp = response

        if self.strip:
            gt = gt.strip()
            resp = resp.strip()

        if not self.case_sensitive:
            gt = gt.lower()
            resp = resp.lower()

        is_correct = resp == gt
        return VerificationResult(
            score=1.0 if is_correct else 0.0,
            is_valid=True,
            metadata={"match_type": "exact"},
        )


class ContainsVerifier(Verifier):
    """Verifier that checks if response contains the ground truth."""

    name = "contains"
    verifier_type = "exact_match"

    def __init__(
        self,
        case_sensitive: bool = False,
        strip: bool = True,
    ):
        self.case_sensitive = case_sensitive
        self.strip = strip

    @beartype
    def verify_single(
        self,
        response: str,
        ground_truth: Any,
        prompt: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> VerificationResult:
        gt = str(ground_truth)
        resp = response

        if self.strip:
            gt = gt.strip()
            resp = resp.strip()

        if not self.case_sensitive:
            gt = gt.lower()
            resp = resp.lower()

        is_correct = gt in resp
        return VerificationResult(
            score=1.0 if is_correct else 0.0,
            is_valid=True,
            metadata={"match_type": "contains"},
        )


class BoxedAnswerVerifier(Verifier):
    """Verifier that extracts answer from \\boxed{} format."""

    name = "boxed"
    verifier_type = "regex"

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
    def verify_single(
        self,
        response: str,
        ground_truth: Any,
        prompt: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> VerificationResult:
        gt = str(ground_truth)

        # Extract all boxed answers
        matches = re.findall(r"\\boxed\{([^}]+)\}", response)

        if not matches:
            return VerificationResult(
                score=0.0,
                is_valid=False,
                metadata={"error": "no_boxed_found"},
            )

        if self.strip:
            matches = [m.strip() for m in matches]
            gt = gt.strip()

        if not self.case_sensitive:
            matches = [m.lower() for m in matches]
            gt = gt.lower()

        is_correct = False
        if self.location_mode == "any":
            is_correct = any(m == gt for m in matches)
        elif self.location_mode == "last":
            is_correct = matches[-1] == gt

        return VerificationResult(
            score=1.0 if is_correct else 0.0,
            is_valid=True,
            metadata={"extracted": matches[-1] if matches else None, "all_matches": matches},
        )


class FirstLetterVerifier(Verifier):
    """Verifier that checks the first letter of the response (for MCQ)."""

    name = "first_letter"
    verifier_type = "regex"

    def __init__(
        self,
        case_sensitive: bool = False,
        lstrip: bool = True,
    ):
        self.case_sensitive = case_sensitive
        self.lstrip = lstrip

    @beartype
    def verify_single(
        self,
        response: str,
        ground_truth: Any,
        prompt: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> VerificationResult:
        gt = str(ground_truth)
        resp = response

        if self.lstrip:
            resp = resp.lstrip()
            gt = gt.lstrip()

        if not resp:
            return VerificationResult(
                score=0.0,
                is_valid=False,
                metadata={"error": "empty_response"},
            )

        first_letter = resp[0]
        gt_first = gt[0] if gt else ""

        if not self.case_sensitive:
            first_letter = first_letter.lower()
            gt_first = gt_first.lower()

        is_correct = first_letter == gt_first
        return VerificationResult(
            score=1.0 if is_correct else 0.0,
            is_valid=True,
            metadata={"extracted": first_letter},
        )


class MCQLetterVerifier(Verifier):
    """Verifier that extracts MCQ letter (A/B/C/D) from response."""

    name = "mcq_letter"
    verifier_type = "regex"

    def __init__(
        self,
        location_mode: Literal["first", "last"] = "first",
        case_sensitive: bool = False,
    ):
        self.location_mode = location_mode
        self.case_sensitive = case_sensitive

    @beartype
    def verify_single(
        self,
        response: str,
        ground_truth: Any,
        prompt: list[dict[str, str]] | None = None,
        **kwargs,
    ) -> VerificationResult:
        gt = str(ground_truth).strip().upper()

        # Find all standalone A/B/C/D letters
        matches = re.findall(r"\b([A-Da-d])\b", response)

        if not matches:
            return VerificationResult(
                score=0.0,
                is_valid=False,
                metadata={"error": "no_letter_found"},
            )

        matches = [m.upper() for m in matches]

        if self.location_mode == "first":
            extracted = matches[0]
        else:  # last
            extracted = matches[-1]

        is_correct = extracted == gt
        return VerificationResult(
            score=1.0 if is_correct else 0.0,
            is_valid=True,
            metadata={"extracted": extracted, "all_matches": matches},
        )
