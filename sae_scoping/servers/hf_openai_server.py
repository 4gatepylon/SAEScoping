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
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from concurrent.futures import ThreadPoolExecutor

from sae_scoping.servers.hf_openai_schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatMessageRole,
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
_executor: ThreadPoolExecutor | None = None

# =============================================================================
# Batching Infrastructure
# =============================================================================
#
# Simple batching strategy: requests accumulate in a queue and are processed
# together. This amortizes GPU kernel launch overhead and improves throughput.
#
# Configuration (via CLI args, defaults maintain backward compatibility):
# - _batch_size: max requests per batch (default=1)
# - _sleep_time: seconds to wait for requests to accumulate (default=0.0)
# When _batch_size=1 and _sleep_time=0.0, equivalent to greedy single-request processing.

_batch_size: int = 1
_sleep_time: float = 0.0


@dataclass
class PendingRequest:
    """A request waiting to be processed in a batch.

    Fields:
    - future: asyncio.Future that will be resolved with (response_text, prompt_tokens, completion_tokens)
    - messages: list[dict] - OpenAI-compatible messages (list of {"role": ..., "content": ...})
    - generation_kwargs: dict - per-request generation parameters (max_new_tokens, temperature, etc.)
    """
    future: asyncio.Future
    messages: list[dict]
    generation_kwargs: dict = field(default_factory=dict)


_request_queue: asyncio.Queue[PendingRequest] | None = None
_batch_processor_task: asyncio.Task | None = None


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage model loading/unloading on startup/shutdown."""
    global _model, _tokenizer, _model_name, _use_hardcoded_response, _executor
    global _request_queue, _batch_processor_task

    # 1. Create model and tokenizer
    if not _use_hardcoded_response and _model is None:
        print(f"Loading model: {_model_name}")
        _tokenizer = AutoTokenizer.from_pretrained(_model_name)
        _model = AutoModelForCausalLM.from_pretrained(
            _model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        _model.eval()

        # Set up padding
        if _tokenizer.pad_token is None:
            print("=" * 100)
            print("WARNING: No pad token found, setting to eos token")
            _tokenizer.pad_token = _tokenizer.eos_token
        _tokenizer.padding_side = "left"

        print(f"Model loaded successfully: {_model_name}")

    # 2. Initialize batching infrastructure
    # 2.1 Create Queue to communicate from async server coroutines to batch processor loop coroutine
    _request_queue = asyncio.Queue()

    # 2.2. Start batch processor loop coroutine (like a thread/worker) that acts as a bridge to
    # pass on requests to the model THREAD
    _batch_processor_task = asyncio.create_task(_batch_processor_loop())

    # 2.3. Create a thread pool executor to run the model inference in parallel
    _executor = ThreadPoolExecutor(max_workers=1)
    print(f"Batch processor started (batch_size={_batch_size}, sleep_time={_sleep_time})")

    yield  # Here the code inside your context runs (i.e. the server, workers, etc... runs)

    # 3. Shutdown batching infrastructure
    # 3.1 Cancel the batch processor loop coroutine (thread will no longer recieve contents)
    if _batch_processor_task is not None:
        _batch_processor_task.cancel()
        try:
            await _batch_processor_task
        except asyncio.CancelledError:
            pass
    # 3.2 Drain queue and reject pending requests (the queue is not being read anymore; clear it)
    if _request_queue is not None:
        while not _request_queue.empty():
            pending = _request_queue.get_nowait()
            pending.future.set_exception(Exception("Server shutting down"))

    # 3.3. Shutdown the thread pool executor (wait for the last batch to finish and shut down)
    if _executor is not None:
        _executor.shutdown(wait=True)

    # 4. Cleanup model and tokenizer (paranoid, but we need to make sure cuda is not hogged)
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

# Enable CORS (NOTE: this would be un-safe in production; this server is meant for prototyping obv.)
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
) -> ChatCompletionResponse:
    """Create a chat completion.

    Submits the request to the batch processing queue.
    The batch processor handles tokenization and batching for efficiency.
    """
    global _model, _tokenizer, _model_name, _use_hardcoded_response, _request_queue

    # TODO(Adriano) Handle streaming in the future here
    if request.stream:
        raise NotImplementedError("Streaming is not supported. We will add it later.")

    # Validate state (model required unless in test mode)
    if not _use_hardcoded_response and (_model is None or _tokenizer is None):
        raise HTTPException(status_code=500, detail="Model not loaded")
    if _request_queue is None:
        raise HTTPException(status_code=500, detail="Batch processor not initialized")

    # 1. Convert messages to OpenAI format and build generation kwargs
    messages_dict = messages_to_openai_format(request.messages)
    generation_kwargs = _build_generation_kwargs(request)

    # 2. Create pending request and submit to queue
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    pending = PendingRequest(
        future=future,
        messages=messages_dict,
        generation_kwargs=generation_kwargs,
    )
    await _request_queue.put(pending)

    # 3. Await result from batch processor
    response_text, prompt_tokens, completion_tokens = await future

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


async def _batch_processor_loop():
    """Background task that processes requests in batches.

    Loop behavior:
    1. Wait for requests: either sleep_time passes OR batch_size requests arrive
    2. Collect up to batch_size requests from queue
    3. Run batched inference
    4. Resolve all futures with their responses
    5. Repeat (skip wait if queue already has >= batch_size items)
    """
    global _model, _tokenizer, _request_queue, _batch_size, _sleep_time, _executor

    while True:
        batch: list[PendingRequest] = []
        try:
            # --- Phase 1: Collect requests ---
            # Wait for first request (blocking)
            first = await _request_queue.get()
            batch.append(first)

            # If we need more requests and sleep_time > 0, wait for more
            if _batch_size > 1 and _sleep_time > 0:
                loop = asyncio.get_running_loop()
                deadline = loop.time() + _sleep_time
                while len(batch) < _batch_size:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        req = await asyncio.wait_for(_request_queue.get(), timeout=remaining)
                        batch.append(req)
                    except asyncio.TimeoutError:
                        break

            # Drain any additional available requests (non-blocking) up to batch_size
            while len(batch) < _batch_size:
                try:
                    req = _request_queue.get_nowait()
                    batch.append(req)
                except asyncio.QueueEmpty:
                    break

            # --- Phase 2: Batched inference ---
            # Run in executor to avoid blocking the event loop
            print(f"[Batch] Processing batch of {len(batch)} requests")
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(
                _executor,  # use the thread pool executor
                _generate_batch_responses,
                batch,
            )

            # --- Phase 3: Resolve futures ---
            for pending, result in zip(batch, results):
                if isinstance(result, Exception):
                    pending.future.set_exception(result)
                else:
                    pending.future.set_result(result)

        except asyncio.CancelledError:
            # Graceful shutdown - reject remaining requests in batch
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(Exception("Server shutting down"))
            raise
        except Exception as e:
            # Log error but keep processing
            print(f"Batch processor error: {e}")
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(e)


def _generate_batch_responses(
    batch: list[PendingRequest],
) -> list[tuple[str, int, int] | Exception]:
    """Generate responses for a batch of requests.

    Handles chat templating, tokenization, batched inference, and decoding.

    Returns list of (response_text, prompt_tokens, completion_tokens) or Exception.

    Note: We use left-padding (set in lifespan), so all sequences align at the right.
    After generation, new tokens are appended at the end (same position for all).
    """
    global _model, _tokenizer, _use_hardcoded_response

    if not batch:
        return []

    # Test mode: return hardcoded responses (still goes through batching for testing)
    if _use_hardcoded_response:
        return [("hello", 10, 1) for _ in batch]

    try:
        # 1. Apply chat template to each request's messages
        text_inputs = [
            _tokenizer.apply_chat_template(
                p.messages, tokenize=False, add_generation_prompt=True
            )
            for p in batch
        ]

        # 2. Tokenize all inputs together (handles padding automatically with left-padding)
        inputs = _tokenizer(
            text_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}

        # Store individual input lengths (before padding) for accurate token counts
        # With left-padding, we count non-pad tokens per sequence
        input_lengths = (inputs["attention_mask"]).sum(dim=1).tolist()
        padded_input_length = inputs["input_ids"].shape[1]

        # 3. Merge generation kwargs: use max of max_new_tokens across batch
        max_new_tokens = max(p.generation_kwargs.get("max_new_tokens", 512) for p in batch)
        generation_kwargs = batch[0].generation_kwargs.copy()
        generation_kwargs["max_new_tokens"] = max_new_tokens
        generation_kwargs["pad_token_id"] = _tokenizer.pad_token_id

        # 4. Generate
        with torch.no_grad():
            outputs = _model.generate(**inputs, **generation_kwargs)

        # 5. Decode each response
        # With left-padding, generated tokens start at padded_input_length for all sequences
        results = []
        for i, pending in enumerate(batch):
            response_text = _tokenizer.decode(
                outputs[i, padded_input_length:], skip_special_tokens=True
            ).strip()
            prompt_tokens = input_lengths[i]  # actual tokens (not padded)
            completion_tokens = outputs.shape[1] - padded_input_length
            results.append((response_text, prompt_tokens, completion_tokens))

        return results

    except Exception as e:
        # Return exception for all requests in batch
        return [e] * len(batch)


def _build_generation_kwargs(request: ChatCompletionRequest) -> dict:
    """Extract generation kwargs from request."""
    kwargs = {
        "max_new_tokens": request.max_tokens or 512,
    }

    if request.temperature == 0 or (request.do_sample is not None and not request.do_sample):
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = request.temperature
        kwargs["top_p"] = request.top_p

    if request.top_k is not None:
        kwargs["top_k"] = request.top_k

    if request.repetition_penalty is not None:
        kwargs["repetition_penalty"] = request.repetition_penalty

    return kwargs


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
    global _batch_size, _sleep_time

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

    # Run with batching (process up to 8 requests together, wait 100ms for accumulation)
    python -m sae_scoping.servers.hf_openai_server --model "..." --batch-size 8 --sleep-time 0.1
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Max requests per batch (default: 1, no batching)",
    )
    parser.add_argument(
        "--sleep-time",
        type=float,
        default=0.0,
        help="Seconds to wait for requests to accumulate before processing batch (default: 0.0)",
    )

    args = parser.parse_args()

    _model_name = args.model
    _use_hardcoded_response = args.test_mode
    _batch_size = args.batch_size
    _sleep_time = args.sleep_time

    if _use_hardcoded_response:
        print("Running in TEST MODE - responses will be hardcoded 'hello'")
    else:
        print(f"Starting server with model: {_model_name}")
        print(f"Batching config: batch_size={_batch_size}, sleep_time={_sleep_time}s")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
