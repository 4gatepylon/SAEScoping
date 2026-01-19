"""
HuggingFace OpenAI-compatible API server module.
"""

from sae_scoping.servers.hf_openai_schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    ChatMessageRole,
    DeltaMessage,
    ErrorDetail,
    ErrorResponse,
    FinishReason,
    ModelInfo,
    ModelList,
    UsageInfo,
    messages_to_openai_format,
    openai_format_to_messages,
)

__all__ = [
    "ChatCompletionChoice",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionStreamChoice",
    "ChatCompletionStreamResponse",
    "ChatMessage",
    "ChatMessageRole",
    "DeltaMessage",
    "ErrorDetail",
    "ErrorResponse",
    "FinishReason",
    "ModelInfo",
    "ModelList",
    "UsageInfo",
    "messages_to_openai_format",
    "openai_format_to_messages",
]
