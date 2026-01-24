from __future__ import annotations
from beartype import beartype
from beartype.typing import Any

OpenAIMessages = list[dict[str, str]]


@beartype
def is_valid_messages(messages: OpenAIMessages | Any) -> bool:
    if not isinstance(messages, list):
        return False
    if not all(isinstance(item, dict) for item in messages):
        return False
    if not all(set(item.keys()) == {"role", "content"} for item in messages):
        return False
    if not all(item["role"] in ["system", "user", "assistant"] for item in messages):
        return False
    if not all(isinstance(item["content"], str) for item in messages):
        return False
    return True


@beartype
def is_valid_0turn_messages(messages: OpenAIMessages | Any) -> bool:
    """0 turn means no response yet."""
    if not is_valid_messages(messages):
        return False

    # Must be either [user] or [system, user]
    if len(messages) == 1:
        return messages[0]["role"] in {"user", "system"}
    elif len(messages) == 2:
        return messages[0]["role"] == "system" and messages[1]["role"] == "user"
    else:
        return False


@beartype
def is_valid_1turn_messages(messages: OpenAIMessages | Any) -> bool:
    """1 turn means one response has been generated."""
    if not is_valid_messages(messages):
        return False

    # Must be either [user, assistant] or [system, user, assistant]
    if len(messages) == 2:
        return messages[0]["role"] in {"user", "system"} and messages[1]["role"] == "assistant"
    elif len(messages) == 3:
        return messages[0]["role"] == "system" and messages[1]["role"] == "user" and messages[2]["role"] == "assistant"
    else:
        return False


@beartype
def convert_1turn_to_0turn(messages: OpenAIMessages | Any) -> OpenAIMessages | Any:
    """Remove the assistant message and return the 0turn messages"""
    if not is_valid_1turn_messages(messages):
        raise ValueError("Input messages is not a valid 1turn messages")
    return messages[:-1]
