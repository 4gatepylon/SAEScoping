"""
Verifiers module for evaluating model responses.

Provides verifiers for:
- Exact match: Simple string comparison
- Regex: Pattern-based extraction (boxed answers, MCQ letters)
- Judge: LLM-based evaluation (utility, safety, refusal)

Usage:
    from sae_scoping.evaluation.verifiers import get_verifier

    # Get a simple exact match verifier
    verifier = get_verifier("exact_match", case_sensitive=False)
    result = verifier.verify_single(response, ground_truth)

    # Get a utility judge
    verifier = get_verifier("utility", model="gpt-4.1-nano")
    batch_result = verifier.verify_batch(responses, ground_truths, prompts)
"""

from sae_scoping.evaluation.verifiers.schemas import (
    VerificationResult,
    BatchVerificationResult,
)
from sae_scoping.evaluation.verifiers.base import Verifier
from sae_scoping.evaluation.verifiers.exact_match import (
    ExactMatchVerifier,
    ContainsVerifier,
    BoxedAnswerVerifier,
    FirstLetterVerifier,
    MCQLetterVerifier,
)
from sae_scoping.evaluation.verifiers.judge import (
    JudgeVerifier,
    RefusalVerifier,
    AnsweringVerifier,
    PreciseVerifier,
    FactualHelpfulVerifier,
    UtilityVerifier,
    SafetyVerifier,
)
from sae_scoping.evaluation.verifiers.registry import (
    VERIFIER_REGISTRY,
    get_verifier,
    register_verifier,
    list_verifiers,
)

__all__ = [
    # Schemas
    "VerificationResult",
    "BatchVerificationResult",
    # Base
    "Verifier",
    # Exact match verifiers
    "ExactMatchVerifier",
    "ContainsVerifier",
    # Regex verifiers
    "BoxedAnswerVerifier",
    "FirstLetterVerifier",
    "MCQLetterVerifier",
    # Judge verifiers
    "JudgeVerifier",
    "RefusalVerifier",
    "AnsweringVerifier",
    "PreciseVerifier",
    "FactualHelpfulVerifier",
    "UtilityVerifier",
    "SafetyVerifier",
    # Registry
    "VERIFIER_REGISTRY",
    "get_verifier",
    "register_verifier",
    "list_verifiers",
]
