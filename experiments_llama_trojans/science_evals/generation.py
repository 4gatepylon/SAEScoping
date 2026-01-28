"""
Generation utilities for science_evals.

Handles calling the OpenAI-compatible server to generate responses.
"""

from __future__ import annotations

from typing import Iterator

from beartype import beartype

from sae_scoping.utils.generation.api_generator import APIGenerator


@beartype
def generate_responses(
    conversations: list[list[dict[str, str]]],
    base_url: str,
    model_name: str = "current",
    max_tokens: int = 1024,
    batch_size: int = 16,
    litellm_provider: str = "openai",
) -> list[str | None]:
    """
    Generate responses for a list of conversations using the server API.

    Args:
        conversations: List of OpenAI-format conversations
        base_url: Server base URL (e.g., "http://align-4.csail.mit.edu:8001")
            Note: Should NOT include /v1 suffix - LiteLLM adds that.
        model_name: Model name to pass to API (server uses whatever is loaded)
        max_tokens: Maximum tokens to generate
        batch_size: Batch size for API calls
        litellm_provider: LiteLLM provider string (default: "openai" for OpenAI-compatible servers)

    Returns:
        List of response strings (None for failed requests)
    """
    generator = APIGenerator()

    # Format model string for LiteLLM
    # For openai provider with custom base_url, just use model name directly
    full_model = f"{litellm_provider}/{model_name}"

    # Ensure base_url ends with /v1 for OpenAI-compatible servers
    api_base = base_url.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = api_base + "/v1"

    responses = list(
        generator.api_generate_streaming(
            prompts=conversations,
            model=full_model,
            batch_size=batch_size,
            return_raw=False,
            batch_completion_kwargs={
                "api_base": api_base,
                "max_tokens": max_tokens,
            },
        )
    )

    return responses


@beartype
def generate_responses_streaming(
    conversations: list[list[dict[str, str]]],
    base_url: str,
    model_name: str = "current",
    max_tokens: int = 1024,
    batch_size: int = 16,
    litellm_provider: str = "openai",
) -> Iterator[str | None]:
    """
    Generate responses as a streaming iterator.

    Same as generate_responses but yields results as they come in.
    Useful for progress tracking.
    """
    generator = APIGenerator()

    full_model = f"{litellm_provider}/{model_name}"

    # Ensure base_url ends with /v1 for OpenAI-compatible servers
    api_base = base_url.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = api_base + "/v1"

    yield from generator.api_generate_streaming(
        prompts=conversations,
        model=full_model,
        batch_size=batch_size,
        return_raw=False,
        batch_completion_kwargs={
            "api_base": api_base,
            "max_tokens": max_tokens,
        },
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import requests

    print("=" * 60)
    print("Testing generation.py")
    print("=" * 60)

    # Test against TEST server (port 8001)
    TEST_URL = "http://align-4.csail.mit.edu:8001"

    # Check server is available
    print(f"\n--- Checking TEST server at {TEST_URL} ---")
    try:
        resp = requests.get(f"{TEST_URL}/v1/models", timeout=5)
        models = resp.json()
        print(f"Server available, model: {models['data'][0]['id']}")
    except Exception as e:
        print(f"ERROR: Cannot reach server: {e}")
        print("Skipping generation tests")
        exit(1)

    # Test conversations
    test_convos = [
        [{"role": "user", "content": "What is biology?"}],
        [{"role": "user", "content": "Explain photosynthesis."}],
        [{"role": "user", "content": "What is DNA?"}],
    ]

    print(f"\n--- Test generate_responses with {len(test_convos)} conversations ---")
    responses = generate_responses(
        conversations=test_convos,
        base_url=TEST_URL,
        model_name="test-mode",
        max_tokens=64,
        batch_size=2,
    )

    print(f"Got {len(responses)} responses")
    for i, (convo, resp) in enumerate(zip(test_convos, responses)):
        user_msg = convo[-1]["content"][:30]
        resp_preview = resp[:50] if resp else "None"
        print(f"  [{i}] User: '{user_msg}...' -> Response: '{resp_preview}...'")

    # Verify test server returns "hello"
    print("\n--- Verify TEST server returns 'hello' ---")
    all_hello = all(r == "hello" for r in responses if r is not None)
    if all_hello:
        print("OK: All responses are 'hello' as expected from test server")
    else:
        print(f"WARNING: Not all responses are 'hello': {responses}")

    print("\n" + "=" * 60)
    print("generation.py tests complete!")
    print("=" * 60)
