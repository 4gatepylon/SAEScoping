"""
HuggingFace OpenAI-compatible API server.
Provides a vLLM-like experience for serving HuggingFace models.

Usage:
    python -m sae_scoping.servers.hf_openai_server --model "Qwen/Qwen2.5-Math-1.5B-Instruct"
    python -m sae_scoping.servers.hf_openai_server --model "meta-llama/Llama-3.2-1B-Instruct" --port 8080

Implement by Claude.
"""

from __future__ import annotations
import argparse
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

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
)


# =============================================================================
# Global State
# =============================================================================

# These will be set at startup
_model: PreTrainedModel | None = None
_tokenizer: PreTrainedTokenizerBase | None = None
_model_name: str = "unknown"
_use_hardcoded_response: bool = False


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model loading/unloading on startup/shutdown."""
    global _model, _tokenizer, _model_name, _use_hardcoded_response

    if not _use_hardcoded_response and _model is None:
        print(f"Loading model: {_model_name}")
        _tokenizer = AutoTokenizer.from_pretrained(_model_name)
        _model = AutoModelForCausalLM.from_pretrained(
            _model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        _model.eval()

        # Set up padding
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.padding_side = "left"

        print(f"Model loaded successfully: {_model_name}")

    yield

    # Cleanup
    if _model is not None:
        del _model
        del _tokenizer
        torch.cuda.empty_cache()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="HuggingFace OpenAI-Compatible API",
    description="Serves HuggingFace models with an OpenAI-compatible API",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    """List available models."""
    return ModelList(
        data=[
            ModelInfo(
                id=_model_name,
                created=int(time.time()),
                owned_by="huggingface",
            )
        ]
    )


@app.get("/v1/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str) -> ModelInfo:
    """Get information about a specific model."""
    if model_id != _model_name:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return ModelInfo(
        id=_model_name,
        created=int(time.time()),
        owned_by="huggingface",
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """Create a chat completion."""
    global _model, _tokenizer, _model_name, _use_hardcoded_response

    # Handle streaming
    if request.stream:
        return StreamingResponse(
            generate_stream_response(request),
            media_type="text/event-stream",
        )

    # Non-streaming response
    if _use_hardcoded_response:
        # Hardcoded response for testing
        response_text = "hello"
        prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
        completion_tokens = 1
    else:
        # Real HF generation
        if _model is None or _tokenizer is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        response_text, prompt_tokens, completion_tokens = generate_response(
            request, _model, _tokenizer
        )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=_model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role=ChatMessageRole.ASSISTANT,
                    content=response_text,
                ),
                finish_reason=FinishReason.STOP,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# =============================================================================
# Generation Logic
# =============================================================================

# XXX(Adriano) please fix blocking async event loop
# XXX(Adriano) please support dynamic batching (simple/dumb strategy tbh)
def generate_response(
    request: ChatCompletionRequest,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[str, int, int]:
    """Generate a response using the HuggingFace model."""
    # Convert messages to format expected by tokenizer
    messages_dict = messages_to_openai_format(request.messages)

    # Apply chat template
    text_input = tokenizer.apply_chat_template(
        messages_dict, tokenize=False, add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(
        text_input,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_length = inputs["input_ids"].shape[1]

    # Build generation kwargs
    generation_kwargs = {
        "max_new_tokens": request.max_tokens or 512,
        "pad_token_id": tokenizer.pad_token_id,
    }

    # Handle temperature / sampling
    if request.temperature == 0 or (request.do_sample is not None and not request.do_sample):
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = request.temperature
        generation_kwargs["top_p"] = request.top_p

    if request.top_k is not None:
        generation_kwargs["top_k"] = request.top_k

    if request.repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = request.repetition_penalty

    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)

    # Decode
    response_text = tokenizer.decode(
        outputs[0, input_length:], skip_special_tokens=True
    ).strip()

    # Token counts
    prompt_tokens = input_length
    completion_tokens = outputs.shape[1] - input_length

    return response_text, prompt_tokens, completion_tokens


async def generate_stream_response(
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    """Generate a streaming response."""
    global _model, _tokenizer, _model_name, _use_hardcoded_response

    response_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if _use_hardcoded_response:
        # Hardcoded streaming response for testing
        words = ["hello"]
    else:
        if _model is None or _tokenizer is None:
            error = ErrorResponse(
                error=ErrorDetail(
                    message="Model not loaded",
                    type="server_error",
                )
            )
            yield f"data: {error.model_dump_json()}\n\n"
            return

        # For simplicity, generate full response then stream tokens
        response_text, _, _ = generate_response(request, _model, _tokenizer)
        words = response_text.split()

    # Stream the response word by word
    for i, word in enumerate(words):
        chunk = ChatCompletionStreamResponse(
            id=response_id,
            created=created,
            model=_model_name,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=DeltaMessage(
                        role=ChatMessageRole.ASSISTANT if i == 0 else None,
                        content=word + (" " if i < len(words) - 1 else ""),
                    ),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

    # Send final chunk with finish_reason
    final_chunk = ChatCompletionStreamResponse(
        id=response_id,
        created=created,
        model=_model_name,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason=FinishReason.STOP,
            )
        ],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# =============================================================================
# Health Check
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model": _model_name}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "HuggingFace OpenAI-Compatible API Server",
        "model": _model_name,
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
        },
    }


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    global _model_name, _use_hardcoded_response

    parser = argparse.ArgumentParser(
        description="HuggingFace OpenAI-Compatible API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with a specific model
    python -m sae_scoping.servers.hf_openai_server --model "Qwen/Qwen2.5-Math-1.5B-Instruct"
    
    # Run on a different port
    python -m sae_scoping.servers.hf_openai_server --model "meta-llama/Llama-3.2-1B-Instruct" --port 8080
    
    # Run in test mode with hardcoded responses
    python -m sae_scoping.servers.hf_openai_server --test-mode
        """,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="HuggingFace model name or path (default: Qwen/Qwen2.5-Math-1.5B-Instruct)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with hardcoded 'hello' responses (no model loading)",
    )

    args = parser.parse_args()

    _model_name = args.model
    _use_hardcoded_response = args.test_mode

    if _use_hardcoded_response:
        print("Running in TEST MODE - responses will be hardcoded 'hello'")
    else:
        print(f"Starting server with model: {_model_name}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
