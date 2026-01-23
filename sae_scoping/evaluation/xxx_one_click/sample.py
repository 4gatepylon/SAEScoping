import json
import hashlib
from typing import Optional
import pydantic


class Messages:
    """
    Hashable wrapper around OpenAI-format messages (list[dict[str, str]]).
    Uses a hash of sorted-key JSON for memory efficiency and hashability.
    """

    __slots__ = ("_messages", "_hash")

    def __init__(self, messages: list[dict[str, str]]):
        self._messages = messages
        # Compute hash once on creation
        sorted_messages = [{k: msg[k] for k in sorted(msg.keys())} for msg in messages]
        json_str = json.dumps(sorted_messages, sort_keys=True)
        self._hash = hashlib.sha256(json_str.encode()).hexdigest()

    @property
    def messages(self) -> list[dict[str, str]]:
        return self._messages

    def __hash__(self) -> int:
        return hash(self._hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Messages):
            return False
        return self._hash == other._hash

    def __repr__(self) -> str:
        return f"Messages({self._hash[:8]}...)"

    @property
    def hash_str(self) -> str:
        """Get the hash string for caching/lookup."""
        return self._hash

    @property
    def user_content(self) -> str:
        """Extract content from last user message."""
        for msg in reversed(self._messages):
            if msg["role"] == "user":
                return msg["content"]
        raise ValueError("No user message found")

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self._messages)

    @classmethod
    def from_json(cls, json_str: str) -> "Messages":
        """Deserialize from JSON string."""
        return cls(json.loads(json_str))

    @classmethod
    def from_string(cls, query: str) -> "Messages":
        """Create single-turn user message from string."""
        return cls([{"role": "user", "content": query}])


class Sample(pydantic.BaseModel):
    """A single evaluation sample from a dataset."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    messages: Messages
    golden_response: Optional[str] = None

    @pydantic.field_validator("messages", mode="before")
    @classmethod
    def convert_messages(cls, v):
        if isinstance(v, list):
            return Messages(v)
        return v

    @staticmethod
    def from_string(query: str) -> "Sample":
        """Convert a plain string to a single-turn user message."""
        return Sample(messages=Messages.from_string(query))

    @staticmethod
    def from_strings(queries: list[str]) -> "list[Sample]":
        return [Sample.from_string(q) for q in queries]

    @property
    def user_content(self) -> str:
        """Extract user content from last user message."""
        return self.messages.user_content

    @property
    def cache_key(self) -> str:
        """Get stable cache key for messages."""
        return self.messages.hash_str


class DatasetSample(pydantic.BaseModel):
    """A sample with its dataset name attached."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    sample: Sample
    dataset_name: str

    @property
    def cache_key(self) -> str:
        return self.sample.cache_key
