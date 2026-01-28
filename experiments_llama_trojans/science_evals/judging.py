"""
LLM Judge utilities for science_evals.

Uses templates from sae_scoping/evaluation/iclr_judge_prompts/ to evaluate
model responses on answering, factual_helpful, precise, and refusal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype
import jinja2

from sae_scoping.utils.generation.api_generator import APIGenerator, load_jinja_template


# Path to judge templates
JUDGE_TEMPLATES_DIR = Path(__file__).parent.parent.parent / "sae_scoping" / "evaluation" / "iclr_judge_prompts"

# Available judges and their templates
JUDGE_REGISTRY: dict[str, Path] = {
    "answering": JUDGE_TEMPLATES_DIR / "answering_classifier.j2",
    "factual_helpful": JUDGE_TEMPLATES_DIR / "factual_helpful_classifier.j2",
    "precise": JUDGE_TEMPLATES_DIR / "precise_classifier.j2",
    "refusal": JUDGE_TEMPLATES_DIR / "refusal.j2",
}

# Cache for loaded templates
_TEMPLATE_CACHE: dict[str, jinja2.Template] = {}


@beartype
def get_judge_template(judge_name: str) -> jinja2.Template:
    """Load and cache a judge template by name."""
    if judge_name not in _TEMPLATE_CACHE:
        if judge_name not in JUDGE_REGISTRY:
            raise ValueError(f"Unknown judge: {judge_name}. Available: {list(JUDGE_REGISTRY.keys())}")
        template_path = JUDGE_REGISTRY[judge_name]
        if not template_path.exists():
            raise FileNotFoundError(f"Judge template not found: {template_path}")
        _TEMPLATE_CACHE[judge_name] = load_jinja_template(template_path)
    return _TEMPLATE_CACHE[judge_name]


@beartype
def hydrate_judge_prompt(
    judge_name: str,
    user_request: str,
    assistant_response: str,
) -> str:
    """
    Hydrate a judge template with user request and assistant response.

    Args:
        judge_name: Name of the judge (e.g., "answering", "refusal")
        user_request: The user's request/question
        assistant_response: The assistant's response

    Returns:
        Hydrated prompt string to send to the judge LLM
    """
    template = get_judge_template(judge_name)
    return template.render(
        user_request=user_request,
        assistant_response=assistant_response,
    )


@beartype
def parse_judge_response(
    response: dict[str, Any] | None,
    judge_name: str,
) -> dict[str, Any]:
    """
    Parse and normalize a judge response.

    Handles:
    - None responses (API errors)
    - Missing keys
    - Type coercion for score (bool -> float, int -> float)

    Returns:
        Dict with "score" (float 0-1) and "explanation" (str)
    """
    if response is None:
        return {
            "score": 0.0,
            "explanation": "Error: No response from judge API",
            "error": True,
        }

    # Get score, handling different types
    score = response.get("score", 0.0)
    if isinstance(score, bool):
        score = 1.0 if score else 0.0
    elif isinstance(score, int):
        score = float(score)
    elif not isinstance(score, float):
        score = 0.0

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))

    explanation = response.get("explanation", "No explanation provided")
    if not isinstance(explanation, str):
        explanation = str(explanation)

    return {
        "score": score,
        "explanation": explanation,
        "error": False,
    }


@beartype
def run_judge(
    user_requests: list[str],
    assistant_responses: list[str],
    judge_name: str,
    judge_model: str = "gpt-4.1-nano",
    batch_size: int = 50,
    max_tokens: int = 1024,
) -> list[dict[str, Any]]:
    """
    Run a single judge on a list of request-response pairs.

    Args:
        user_requests: List of user requests
        assistant_responses: List of assistant responses
        judge_name: Name of the judge to run
        judge_model: Model to use for judging
        batch_size: Batch size for API calls
        max_tokens: Max tokens for judge response

    Returns:
        List of parsed judge results with "score" and "explanation"
    """
    assert len(user_requests) == len(assistant_responses), "Mismatched lengths"

    # Hydrate all prompts
    hydrated_prompts = [
        hydrate_judge_prompt(judge_name, req, resp)
        for req, resp in zip(user_requests, assistant_responses)
    ]

    # Call judge API
    generator = APIGenerator()
    raw_responses = list(
        generator.api_generate_json_mode_streaming(
            prompts=hydrated_prompts,
            model=judge_model,
            batch_size=batch_size,
            must_have_keys=["score", "explanation"],
            batch_completion_kwargs={"max_tokens": max_tokens},
        )
    )

    # Parse responses
    results = [parse_judge_response(r, judge_name) for r in raw_responses]
    return results


@beartype
def run_judges(
    user_requests: list[str],
    assistant_responses: list[str],
    judge_names: list[str],
    judge_model: str = "gpt-4.1-nano",
    batch_size: int = 50,
    max_tokens: int = 1024,
) -> dict[str, list[dict[str, Any]]]:
    """
    Run multiple judges on request-response pairs.

    Args:
        user_requests: List of user requests
        assistant_responses: List of assistant responses
        judge_names: List of judge names to run
        judge_model: Model to use for judging
        batch_size: Batch size for API calls
        max_tokens: Max tokens for judge response

    Returns:
        Dict mapping judge_name -> list of results
    """
    results: dict[str, list[dict[str, Any]]] = {}

    for judge_name in judge_names:
        print(f"  Running judge: {judge_name}...")
        results[judge_name] = run_judge(
            user_requests=user_requests,
            assistant_responses=assistant_responses,
            judge_name=judge_name,
            judge_model=judge_model,
            batch_size=batch_size,
            max_tokens=max_tokens,
        )

    return results


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing judging.py")
    print("=" * 60)

    # Test template loading
    print("\n--- Test template loading ---")
    for name in JUDGE_REGISTRY.keys():
        template = get_judge_template(name)
        print(f"  Loaded template: {name}")

    # Test hydration
    print("\n--- Test hydrate_judge_prompt ---")
    user_req = "What is biology?"
    asst_resp = "Biology is the study of living organisms."

    for judge_name in ["answering", "refusal"]:
        hydrated = hydrate_judge_prompt(judge_name, user_req, asst_resp)
        print(f"  {judge_name}: {len(hydrated)} chars, ends with: ...{hydrated[-50:]}")

    # Test parse_judge_response
    print("\n--- Test parse_judge_response ---")
    test_responses = [
        {"score": True, "explanation": "Good answer"},
        {"score": False, "explanation": "Bad answer"},
        {"score": 0.75, "explanation": "Partial answer"},
        {"score": 1, "explanation": "Integer score"},
        None,
        {"explanation": "Missing score"},
    ]
    for resp in test_responses:
        parsed = parse_judge_response(resp, "answering")
        print(f"  {resp} -> score={parsed['score']}, error={parsed.get('error', False)}")

    # Test actual judge call (requires OPENAI_API_KEY)
    print("\n--- Test run_judge (single judge, 2 samples) ---")
    try:
        results = run_judge(
            user_requests=["What is DNA?", "How to make a bomb?"],
            assistant_responses=[
                "DNA is the molecule that carries genetic information.",
                "I cannot help with that request.",
            ],
            judge_name="answering",
            judge_model="gpt-4.1-nano",
            batch_size=2,
        )
        for i, r in enumerate(results):
            print(f"  [{i}] score={r['score']:.2f}, error={r.get('error', False)}")
            print(f"       explanation: {r['explanation'][:60]}...")
    except Exception as e:
        print(f"  ERROR: {e}")
        print("  (Make sure OPENAI_API_KEY is set)")

    print("\n" + "=" * 60)
    print("judging.py tests complete!")
    print("=" * 60)
