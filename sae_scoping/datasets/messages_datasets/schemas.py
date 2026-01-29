"""
Canonical schemas for message-based datasets (SFT training data).

These datasets contain conversations in OpenAI message format,
used for Supervised Fine-Tuning (SFT) of chat models.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field
from sae_scoping.datasets.shared.schemas import DatasetInfo


class Message(BaseModel):
    """Single message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class MessagesEntry(BaseModel):
    """
    Canonical format for a conversation entry.

    Used for SFT datasets like UltraChat, StemQAMixture, etc.
    """

    messages: list[Message]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def num_turns(self) -> int:
        """Number of conversation turns (user + assistant pairs)."""
        return len([m for m in self.messages if m.role in ("user", "assistant")]) // 2

    @property
    def has_system(self) -> bool:
        """Whether the conversation has a system message."""
        return any(m.role == "system" for m in self.messages)

class MessagesDataset(BaseModel):
    """A loaded messages dataset for SFT."""

    info: DatasetInfo
    entries: list[MessagesEntry]
