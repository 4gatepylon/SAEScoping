"""
OpenAI-compatible API schemas for HuggingFace model serving.
Follows the OpenAI Chat Completions API specification.

Implement by Claude.
"""

from __future__ import annotations
from enum import Enum
from typing import Literal
import time
import uuid
from pydantic import BaseModel, Field


# =============================================================================
# Message Types
# =============================================================================


class ChatMessageRole(str, Enum):
    """Valid roles for chat messages."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: ChatMessageRole
    content: str


# =============================================================================
# Request Schemas
# =============================================================================


class ChatCompletionRequest(BaseModel):
    """Request schema for /v1/chat/completions endpoint."""

    model: str = Field(description="The model to use for completion")
    messages: list[ChatMessage] = Field(description="A list of messages comprising the conversation so far")

    # Optional generation parameters (matching OpenAI API)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling probability")
    n: int = Field(default=1, ge=1, description="Number of completions to generate")
    max_tokens: int | None = Field(default=None, description="Maximum number of tokens to generate")
    stop: str | list[str] | None = Field(default=None, description="Stop sequences")
    stream: bool = Field(default=False, description="Whether to stream responses")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    user: str | None = Field(default=None, description="User identifier")

    # Additional params for HF compatibility
    do_sample: bool | None = Field(default=None, description="Whether to use sampling (overrides temperature=0)")
    top_k: int | None = Field(default=None, ge=0, description="Top-k sampling parameter")
    repetition_penalty: float | None = Field(default=None, ge=0.0, description="Repetition penalty")


# =============================================================================
# Response Schemas
# =============================================================================


class FinishReason(str, Enum):
    """Reason for completion finishing."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"


class UsageInfo(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(description="Number of tokens in the prompt")
    completion_tokens: int = Field(description="Number of tokens in the completion")
    total_tokens: int = Field(description="Total number of tokens used")


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""

    index: int = Field(description="Index of this choice")
    message: ChatMessage = Field(description="The generated message")
    finish_reason: FinishReason | None = Field(default=None, description="Reason for completion finishing")


class ChatCompletionResponse(BaseModel):
    """Response schema for /v1/chat/completions endpoint."""

    id: str = Field(
        default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}",
        description="Unique identifier for the completion",
    )
    object: Literal["chat.completion"] = Field(default="chat.completion", description="Object type")
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp of creation",
    )
    model: str = Field(description="Model used for completion")
    choices: list[ChatCompletionChoice] = Field(description="List of completion choices")
    usage: UsageInfo | None = Field(default=None, description="Token usage information")


# =============================================================================
# Streaming Response Schemas
# =============================================================================


class DeltaMessage(BaseModel):
    """Partial message for streaming."""

    role: ChatMessageRole | None = None
    content: str | None = None


class ChatCompletionStreamChoice(BaseModel):
    """A single streaming completion choice."""

    index: int = Field(description="Index of this choice")
    delta: DeltaMessage = Field(description="The delta message")
    finish_reason: FinishReason | None = Field(default=None, description="Reason for completion finishing")


class ChatCompletionStreamResponse(BaseModel):
    """Streaming response schema for /v1/chat/completions endpoint."""

    id: str = Field(
        default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}",
        description="Unique identifier for the completion",
    )
    object: Literal["chat.completion.chunk"] = Field(default="chat.completion.chunk", description="Object type")
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp of creation",
    )
    model: str = Field(description="Model used for completion")
    choices: list[ChatCompletionStreamChoice] = Field(description="List of completion choices")


# =============================================================================
# Model Information Schemas
# =============================================================================


class ModelInfo(BaseModel):
    """Information about an available model."""

    id: str = Field(description="Model identifier")
    object: Literal["model"] = Field(default="model", description="Object type")
    created: int = Field(
        default_factory=lambda: int(time.time()),
        description="Unix timestamp of creation",
    )
    owned_by: str = Field(default="huggingface", description="Model owner")


class ModelList(BaseModel):
    """List of available models."""

    object: Literal["list"] = Field(default="list", description="Object type")
    data: list[ModelInfo] = Field(description="List of models")


# =============================================================================
# Error Schemas
# =============================================================================


class ErrorDetail(BaseModel):
    """Error detail information."""

    message: str = Field(description="Error message")
    type: str = Field(description="Error type")
    param: str | None = Field(default=None, description="Parameter that caused error")
    code: str | None = Field(default=None, description="Error code")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: ErrorDetail = Field(description="Error details")


# =============================================================================
# Utility Functions
# =============================================================================


def messages_to_openai_format(messages: list[ChatMessage]) -> list[dict[str, str]]:
    """Convert ChatMessage list to OpenAI-style dict format."""
    return [{"role": msg.role.value, "content": msg.content} for msg in messages]


def openai_format_to_messages(messages: list[dict[str, str]]) -> list[ChatMessage]:
    """Convert OpenAI-style dict format to ChatMessage list."""
    return [ChatMessage(role=ChatMessageRole(msg["role"]), content=msg["content"]) for msg in messages]
