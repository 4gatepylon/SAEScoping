"""
Verifier registry.

Provides a centralized registry for all available verifiers,
allowing them to be accessed by name.
"""

from __future__ import annotations

from beartype import beartype
from beartype.typing import Callable

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


# Type for verifier factory
VerifierFactory = Callable[..., Verifier]


# Registry mapping verifier names to factory functions
VERIFIER_REGISTRY: dict[str, VerifierFactory] = {
    # Exact match verifiers
    "exact_match": ExactMatchVerifier,
    "contains": ContainsVerifier,
    # Regex-based verifiers
    "boxed": BoxedAnswerVerifier,
    "first_letter": FirstLetterVerifier,
    "mcq_letter": MCQLetterVerifier,
    # Judge-based verifiers
    "judge": JudgeVerifier,
    "refusal": RefusalVerifier,
    "answering": AnsweringVerifier,
    "precise": PreciseVerifier,
    "factual_helpful": FactualHelpfulVerifier,
    # Composite verifiers
    "utility": UtilityVerifier,
    "safety": SafetyVerifier,
}


@beartype
def get_verifier(name: str, **kwargs) -> Verifier:
    """
    Get a verifier by name from the registry.

    Args:
        name: Verifier name (e.g., "exact_match", "utility", "safety")
        **kwargs: Arguments passed to the verifier constructor

    Returns:
        Instantiated verifier

    Raises:
        ValueError: If verifier name is not in registry
    """
    if name not in VERIFIER_REGISTRY:
        available = list(VERIFIER_REGISTRY.keys())
        raise ValueError(f"Unknown verifier: {name}. Available: {available}")

    return VERIFIER_REGISTRY[name](**kwargs)


@beartype
def register_verifier(name: str, factory: VerifierFactory) -> None:
    """
    Register a new verifier factory.

    Args:
        name: Verifier name
        factory: Callable that creates a Verifier instance
    """
    VERIFIER_REGISTRY[name] = factory


def list_verifiers() -> list[str]:
    """List all available verifier names."""
    return list(VERIFIER_REGISTRY.keys())
