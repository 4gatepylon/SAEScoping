import json
import hashlib
from typing import Optional
import pydantic
from beartype import beartype


class Sample(pydantic.BaseModel):
    """A single evaluation sample from a seed dataset."""

    messages: list[
        dict[str, str]
    ]  # OpenAI format: [{"role": "user", "content": "..."}]
    golden_response: Optional[str] = None

    @staticmethod
    def from_string(query: str) -> "Sample":
        """Convert a plain string to a single-turn user message."""
        return Sample(messages=[{"role": "user", "content": query}])

    @staticmethod
    def from_strings(queries: list[str]) -> "list[Sample]":
        return [Sample.from_string(q) for q in queries]

    @property
    def user_content(self) -> str:
        """Extract user content from last user message."""
        for msg in reversed(self.messages):
            if msg["role"] == "user":
                return msg["content"]
        raise ValueError("No user message found")


class AugmentedSample(pydantic.BaseModel):
    """A sample after augmentation has been applied."""

    original: Sample
    augmented_messages: list[dict[str, str]]  # Modified messages
    seed_name: str  # Which seed this came from
    augmentation_name: (
        str  # Which augmentation variant was applied (e.g., "trojan1", "none")
    )
    augmentation_value: str  # The actual value used (e.g., "SUDO", "")

    @staticmethod
    @beartype
    def messages_to_cache_key(messages: list[dict[str, str]]) -> str:
        """Convert messages to a stable cache key by sorting dict keys and hashing."""
        # Sort keys in each message dict for deterministic serialization
        sorted_messages = [{k: msg[k] for k in sorted(msg.keys())} for msg in messages]
        json_str = json.dumps(sorted_messages, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    @property
    def cache_key(self) -> str:
        """Get stable cache key for augmented messages."""
        return self.messages_to_cache_key(self.augmented_messages)
