"""
HuggingFace OpenAI-compatible API server with dynamic model loading.

Usage:
    python -m sae_scoping.servers.hf_openai_server --config default_model_config.json
    python -m sae_scoping.servers.hf_openai_server --config my_config.json --port 8080

Supports SAELens and Sparsify SAEs, with dynamic model swapping via POST /v1/model/change.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import click
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from sae_scoping.servers.hf_openai_schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatMessageRole,
    FinishReason,
    ModelChangeRequest,
    ModelChangeResponse,
    ModelInfo,
    ModelList,
    UsageInfo,
    messages_to_openai_format,
)
from sae_scoping.utils.hooks.pt_hooks import filter_hook_fn, named_forward_hooks
from sae_scoping.utils.hooks.sae import SAEWrapper


# =============================================================================
# Global State
# =============================================================================

_model: PreTrainedModel | None = None
_tokenizer: PreTrainedTokenizerBase | None = None
_model_name: str = "unknown"
_use_hardcoded_response: bool = False
_executor: ThreadPoolExecutor | None = None
_chat_template: str | None = None
_sae_hook_dict: dict = {}
_batch_size: int = 1
_sleep_time: float = 0.0
_request_queue: asyncio.Queue | None = None
_batch_processor_task: asyncio.Task | None = None
_current_config: ModelChangeRequest | None = None
_allow_non_eager_gemma2_global: bool = False  # CLI flag


# =============================================================================
# Model Loading
# =============================================================================


def _validate_gemma2_attention(config: ModelChangeRequest, global_allow: bool) -> None:
    """Validate Gemma2 attention configuration."""
    model_lower = config.model_name_or_path.lower()
    if "gemma" in model_lower and "2" in model_lower:
        if config.attn_implementation != "eager":
            if not config.allow_non_eager_attention_for_gemma2 and not global_allow:
                raise ValueError(
                    f"Gemma2 model detected ({config.model_name_or_path}) but attn_implementation != 'eager'. "
                    "Gemma2 requires eager attention. Set attn_implementation='eager' or "
                    "allow_non_eager_attention_for_gemma2=true to override."
                )


def _load_sae(config: ModelChangeRequest, device: torch.device) -> dict:
    """Load SAE and return hook dict. Returns empty dict if no SAE configured."""
    if config.sae_mode is None and config.sae_path is None and config.sae_release is None:
        return {}

    # Validate SAE config
    if config.sae_mode == "sparsify" or config.sae_path is not None:
        if config.sae_path is None:
            raise ValueError("sae_path required for sparsify mode")
        if config.hookpoint is None:
            raise ValueError("hookpoint required for SAE")
        return _load_sparsify_sae(config, device)

    elif config.sae_mode == "saelens" or config.sae_release is not None:
        if config.sae_release is None or config.sae_id is None:
            raise ValueError("sae_release and sae_id required for saelens mode")
        if config.hookpoint is None:
            raise ValueError("hookpoint required for SAE")
        return _load_saelens_sae(config, device)

    return {}


def _load_sparsify_sae(config: ModelChangeRequest, device: torch.device) -> dict:
    """Load a Sparsify SAE from local path."""
    from sparsify import SparseCoder

    print(f"Loading Sparsify SAE from {config.sae_path}")
    sae_path = Path(config.sae_path)
    if not sae_path.exists():
        raise FileNotFoundError(f"Sparsify SAE not found: {sae_path}")

    sae = SparseCoder.load_from_disk(sae_path.resolve().as_posix())
    sae = sae.to(device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad = False

    # Apply pruning if configured
    if config.distribution_path is not None and config.prune_threshold is not None:
        sae = _apply_pruning_to_sparsify(sae, config, device)

    sae_wrapper = SAEWrapper(sae)
    print(f"Sparsify SAE loaded, hookpoint: {config.hookpoint}")
    return {config.hookpoint: partial(filter_hook_fn, sae_wrapper)}


def _apply_pruning_to_sparsify(sae, config: ModelChangeRequest, device: torch.device):
    """Apply pruning to Sparsify SAE (placeholder - may need adjustment)."""
    from safetensors.torch import load_file

    dist_path = Path(config.distribution_path)
    if not dist_path.exists():
        raise FileNotFoundError(f"Distribution file not found: {dist_path}")

    dist_data = load_file(dist_path)
    distribution: torch.Tensor = dist_data["distribution"]
    n_kept = int((distribution >= config.prune_threshold).sum().item())
    print(f"Pruning Sparsify SAE: keeping {n_kept} neurons (threshold={config.prune_threshold})")
    # NOTE: Sparsify pruning API may differ - this is a placeholder
    # For now, return unpruned SAE with warning
    print("WARNING: Sparsify SAE pruning not yet implemented, using full SAE")
    return sae


def _load_saelens_sae(config: ModelChangeRequest, device: torch.device) -> dict:
    """Load a SAELens SAE from HuggingFace."""
    from sae_lens import SAE
    from safetensors.torch import load_file
    from sae_scoping.trainers.sae_enhanced.prune import get_pruned_sae

    print(f"Loading SAELens SAE: release={config.sae_release}, id={config.sae_id}")
    sae = SAE.from_pretrained(release=config.sae_release, sae_id=config.sae_id, device=device)
    sae = sae.to(device)

    # Apply pruning if configured
    if config.distribution_path is not None and config.prune_threshold is not None:
        dist_path = Path(config.distribution_path)
        if not dist_path.exists():
            raise FileNotFoundError(f"Distribution file not found: {dist_path}")
        dist_data = load_file(dist_path)
        distribution: torch.Tensor = dist_data["distribution"]
        neuron_ranking = torch.argsort(distribution, descending=True)
        n_kept = int((distribution >= config.prune_threshold).sum().item())
        print(f"Pruning SAELens SAE: keeping {n_kept} neurons (threshold={config.prune_threshold})")
        sae = get_pruned_sae(sae, neuron_ranking, K_or_p=n_kept, T=0.0)
        sae = sae.to(device)
    else:
        print("SAELens SAE loaded without pruning")

    sae_wrapper = SAEWrapper(sae)
    print(f"SAELens SAE loaded, hookpoint: {config.hookpoint}")
    return {config.hookpoint: partial(filter_hook_fn, sae_wrapper)}


def _load_model_from_config(config: ModelChangeRequest) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, dict, str]:
    """Load model, tokenizer, and SAE from config. Returns (model, tokenizer, hook_dict, chat_template)."""
    _validate_gemma2_attention(config, _allow_non_eager_gemma2_global)

    # Build model kwargs
    model_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if config.attn_implementation is not None:
        model_kwargs["attn_implementation"] = config.attn_implementation

    print(f"Loading model: {config.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
    model.eval()

    # Setup padding
    if tokenizer.pad_token is None:
        print("WARNING: No pad token found, setting to eos token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load SAE if configured
    device = next(model.parameters()).device
    hook_dict = _load_sae(config, device)

    # Load chat template if configured
    chat_template = None
    if config.chat_template_path is not None:
        template_path = Path(config.chat_template_path)
        if not template_path.exists():
            raise FileNotFoundError(f"Chat template not found: {template_path}")
        chat_template = template_path.read_text()
        print(f"Loaded chat template from: {template_path}")

    print(f"Model loaded: {config.model_name_or_path}")
    return model, tokenizer, hook_dict, chat_template


def _unload_current_model() -> None:
    """Unload current model and free GPU memory."""
    global _model, _tokenizer, _sae_hook_dict

    if _model is not None:
        print("Unloading current model...")
        _model.to("cpu")
        del _model
        _model = None

    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None

    _sae_hook_dict = {}
    torch.cuda.empty_cache()
    print("Model unloaded, GPU memory cleared")


# =============================================================================
# Batching Infrastructure
# =============================================================================


@dataclass
class PendingRequest:
    future: asyncio.Future
    messages: list[dict]
    generation_kwargs: dict = field(default_factory=dict)


async def _batch_processor_loop():
    """Background task that processes requests in batches."""
    global _model, _tokenizer, _request_queue, _batch_size, _sleep_time, _executor

    while True:
        batch: list[PendingRequest] = []
        try:
            first = await _request_queue.get()
            batch.append(first)

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

            while len(batch) < _batch_size:
                try:
                    req = _request_queue.get_nowait()
                    batch.append(req)
                except asyncio.QueueEmpty:
                    break

            print(f"[Batch] Processing batch of {len(batch)} requests")
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(_executor, _generate_batch_responses, batch)

            for pending, result in zip(batch, results):
                if isinstance(result, Exception):
                    pending.future.set_exception(result)
                else:
                    pending.future.set_result(result)

        except asyncio.CancelledError:
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(Exception("Server shutting down"))
            raise
        except Exception as e:
            print(f"Batch processor error: {e}")
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(e)


def _generate_batch_responses(batch: list[PendingRequest]) -> list[tuple[str, int, int] | Exception]:
    """Generate responses for a batch of requests."""
    global _model, _tokenizer, _use_hardcoded_response, _chat_template, _sae_hook_dict

    if not batch:
        return []

    if _use_hardcoded_response:
        return [("hello", 10, 1) for _ in batch]

    try:
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if _chat_template is not None:
            template_kwargs["chat_template"] = _chat_template
        text_inputs = [_tokenizer.apply_chat_template(p.messages, **template_kwargs) for p in batch]

        inputs = _tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}

        input_lengths = (inputs["attention_mask"]).sum(dim=1).tolist()
        padded_input_length = inputs["input_ids"].shape[1]

        max_new_tokens = max(p.generation_kwargs.get("max_new_tokens", 512) for p in batch)
        generation_kwargs = batch[0].generation_kwargs.copy()
        generation_kwargs["max_new_tokens"] = max_new_tokens
        generation_kwargs["pad_token_id"] = _tokenizer.pad_token_id

        with torch.no_grad():
            with named_forward_hooks(_model, _sae_hook_dict):
                outputs = _model.generate(**inputs, **generation_kwargs)

        results = []
        for i, pending in enumerate(batch):
            response_text = _tokenizer.decode(outputs[i, padded_input_length:], skip_special_tokens=True).strip()
            prompt_tokens = input_lengths[i]
            completion_tokens = outputs.shape[1] - padded_input_length
            results.append((response_text, prompt_tokens, completion_tokens))

        return results

    except Exception as e:
        return [e] * len(batch)


def _build_generation_kwargs(request: ChatCompletionRequest) -> dict:
    """Extract generation kwargs from request."""
    kwargs = {"max_new_tokens": request.max_tokens or 512}

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
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup/shutdown."""
    global _model, _tokenizer, _model_name, _use_hardcoded_response, _executor
    global _request_queue, _batch_processor_task, _sae_hook_dict, _chat_template
    global _batch_size, _sleep_time, _current_config

    # Load initial model from config
    if _current_config is not None and not _current_config.test_mode:
        _model, _tokenizer, _sae_hook_dict, _chat_template = _load_model_from_config(_current_config)
        _model_name = _current_config.model_name_or_path
        _batch_size = _current_config.batch_size
        _sleep_time = _current_config.sleep_time
    elif _current_config is not None and _current_config.test_mode:
        _use_hardcoded_response = True
        _model_name = "test-mode"

    # Initialize batching
    _request_queue = asyncio.Queue()
    _batch_processor_task = asyncio.create_task(_batch_processor_loop())
    _executor = ThreadPoolExecutor(max_workers=1)
    print(f"Batch processor started (batch_size={_batch_size}, sleep_time={_sleep_time})")

    yield

    # Shutdown
    if _batch_processor_task is not None:
        _batch_processor_task.cancel()
        try:
            await _batch_processor_task
        except asyncio.CancelledError:
            pass

    if _request_queue is not None:
        while not _request_queue.empty():
            pending = _request_queue.get_nowait()
            pending.future.set_exception(Exception("Server shutting down"))

    if _executor is not None:
        _executor.shutdown(wait=True)

    _unload_current_model()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="HuggingFace OpenAI-Compatible API",
    description="Serves HuggingFace models with OpenAI-compatible API and SAE support",
    version="0.2.0",
    lifespan=lifespan,
)

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
    return ModelList(data=[ModelInfo(id=_model_name, created=int(time.time()), owned_by="huggingface")])


@app.get("/v1/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str) -> ModelInfo:
    """Get information about a specific model."""
    if model_id != _model_name:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return ModelInfo(id=_model_name, created=int(time.time()), owned_by="huggingface")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Create a chat completion."""
    global _model, _tokenizer, _model_name, _use_hardcoded_response, _request_queue

    if request.stream:
        raise NotImplementedError("Streaming not supported")

    if not _use_hardcoded_response and (_model is None or _tokenizer is None):
        raise HTTPException(status_code=500, detail="Model not loaded")
    if _request_queue is None:
        raise HTTPException(status_code=500, detail="Batch processor not initialized")

    messages_dict = messages_to_openai_format(request.messages)
    generation_kwargs = _build_generation_kwargs(request)

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    pending = PendingRequest(future=future, messages=messages_dict, generation_kwargs=generation_kwargs)
    await _request_queue.put(pending)

    response_text, prompt_tokens, completion_tokens = await future

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=_model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role=ChatMessageRole.ASSISTANT, content=response_text),
                finish_reason=FinishReason.STOP,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@app.post("/v1/model/change", response_model=ModelChangeResponse)
async def change_model(request: ModelChangeRequest) -> ModelChangeResponse:
    """Change the currently loaded model."""
    global _model, _tokenizer, _model_name, _sae_hook_dict, _chat_template
    global _batch_size, _sleep_time, _current_config, _use_hardcoded_response

    # Check queue is empty
    if _request_queue is not None and not _request_queue.empty():
        raise HTTPException(
            status_code=409,
            detail=f"Cannot change model: {_request_queue.qsize()} requests pending in queue"
        )

    try:
        # Validate Gemma2 attention
        _validate_gemma2_attention(request, _allow_non_eager_gemma2_global)

        # Unload current model
        _unload_current_model()

        # Handle test mode
        if request.test_mode:
            _use_hardcoded_response = True
            _model_name = "test-mode"
            _current_config = request
            return ModelChangeResponse(success=True, model=_model_name, message="Switched to test mode")

        # Load new model
        _use_hardcoded_response = False
        _model, _tokenizer, _sae_hook_dict, _chat_template = _load_model_from_config(request)
        _model_name = request.model_name_or_path
        _batch_size = request.batch_size
        _sleep_time = request.sleep_time
        _current_config = request

        sae_info = ""
        if _sae_hook_dict:
            sae_info = f" with SAE at {request.hookpoint}"

        return ModelChangeResponse(
            success=True,
            model=_model_name,
            message=f"Model loaded successfully{sae_info}"
        )

    except Exception as e:
        return ModelChangeResponse(success=False, model=_model_name, message=f"Failed to load model: {str(e)}")


@app.get("/v1/model/config")
async def get_model_config():
    """Get current model configuration."""
    if _current_config is None:
        return {"config": None, "model": _model_name}
    return {"config": _current_config.model_dump(), "model": _model_name}


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
            "model_change": "/v1/model/change",
            "model_config": "/v1/model/config",
            "health": "/health",
        },
    }


# =============================================================================
# CLI Entry Point
# =============================================================================


@click.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True, help="Path to model config JSON file")
@click.option("--host", type=str, default="0.0.0.0", help="Host to bind to")
@click.option("--port", type=int, default=8000, help="Port to bind to")
@click.option(
    "--allow-non-eager-attention-for-gemma2",
    is_flag=True,
    default=False,
    help="Allow non-eager attention for Gemma2 models globally",
)
def main(config: str, host: str, port: int, allow_non_eager_attention_for_gemma2: bool):
    """Start the HuggingFace OpenAI-compatible API server."""
    global _current_config, _allow_non_eager_gemma2_global

    _allow_non_eager_gemma2_global = allow_non_eager_attention_for_gemma2

    # Load config from JSON
    config_path = Path(config)
    with open(config_path) as f:
        config_dict = json.load(f)

    _current_config = ModelChangeRequest(**config_dict)

    print(f"Loaded config from: {config_path}")
    print(f"Model: {_current_config.model_name_or_path}")
    if _current_config.sae_path:
        print(f"SAE (Sparsify): {_current_config.sae_path}")
    elif _current_config.sae_release:
        print(f"SAE (SAELens): {_current_config.sae_release}/{_current_config.sae_id}")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
