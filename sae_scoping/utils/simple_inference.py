from __future__ import annotations

import tqdm
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
import torch
import functools
from sparsify import SparseCoder
from utils.hooks.pt_hooks import named_forward_hooks, filter_hook_fn
from utils.hooks.sae import SaeWrapper, CollectionWrapper, ActivationsCollector
import copy
import numpy as np
from typing import Literal
from utils.code_data.load_chats import is_valid_chat

"""
Unlike the `inference.py` utils module which is meant to provide exhaustive support to
different inference methods in an efficient way for large datasets (with OOP), this
`simple_inference.py` module provides an imperative set of helpful functions that I
find myself using frequently.
"""

Chat = List[Dict[str, Any]]


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    dtype: torch.dtype = torch.float16,
    from_pretrained_kwargs: Dict[str, Any] = {},
):
    # NOTE: if you get issues with loading the model and CUDA try loading to CPU and
    # then moving to GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map={"": device}, dtype=dtype, **from_pretrained_kwargs
    )
    for p in model.parameters():
        p.requires_grad = False
        p.grad = None
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    return model, tokenizer


def load_sae(sae_path: Path, device: str, dtype: torch.dtype = torch.float16):
    assert sae_path.exists()
    sae = SparseCoder.load_from_disk(sae_path.resolve().as_posix())
    sae = sae.to(device)
    # sae.to(dtype)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad = False
        p.grad = None
    hookpoint = f"model.{sae_path.name}"  # (i.e. model.layers.0)
    return sae, hookpoint


def collector_hook_fn(
    sae: SparseCoder,
) -> Tuple[
    Callable,
    ActivationsCollector,
    ActivationsCollector,
]:
    sw = SaeWrapper(sae)
    collect_inputs: bool = True
    collect_outputs: bool = True
    sw_collector = CollectionWrapper(
        sw,
        collect_inputs,
        collect_outputs,
        # Defaults here, but moving to CPU can save memory
        clone=False,
        detach=True,
        cpu=True,
    )
    collector_inputs: ActivationsCollector = sw_collector.input_collector
    collector_outputs: ActivationsCollector = sw_collector.output_collector
    return (
        functools.partial(filter_hook_fn, sw_collector),
        collector_inputs,
        collector_outputs,
    )


def _sanity_check_logits_attention_mask_and_labels(
    logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor
) -> bool:
    assert logits is not None
    assert labels is not None
    assert logits.shape[:-1] == labels.shape
    assert logits.ndim == 3  # B x T x V
    assert labels.ndim == 2  # B x T

    assert attention_mask.shape == labels.shape
    assert labels.dtype == torch.long
    assert attention_mask.dtype == torch.bool or attention_mask.dtype in [
        torch.int32,
        torch.int64,
    ]
    return True


def token_by_token_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_size: int = 5,
) -> List[np.ndarray]:
    """
    Given an attention mask that defines where the tokens to actually get the losses for
    are, return the per-token losses. Each token is one entry in a 1D np array of floats
    inside a list (the list covers the batch dimension).
    """
    # 1. Sanity check typing.
    assert _sanity_check_logits_attention_mask_and_labels(
        logits, labels, attention_mask
    )
    if not attention_mask.dtype == torch.bool:
        assert attention_mask.min() >= 0 and attention_mask.max() <= 1
        attention_mask = attention_mask.bool()
    assert attention_mask.dtype == torch.bool
    # 2. Iterate over chunks
    all_per_token_losses: List[np.ndarray] = []
    for i in range(0, logits.shape[0], chunk_size):
        # 1. Extract chunks
        j = min(i + chunk_size, logits.shape[0])
        these_logits = logits[i:j]
        these_labels = labels[i:j]
        these_attention_mask = attention_mask[i:j]
        # 2. Compute per-token losses
        # Compute cross entropy loss with no reduction (per-token losses)
        # logits: B x T x V, labels: B x T
        B, T, V = these_logits.shape
        these_flat_logits = these_logits.view(
            -1, V
        ).double()  # (B*T) x V and we need precision :P
        these_flat_labels = these_labels.view(-1)  # (B*T)
        these_flat_mask = these_attention_mask.view(-1)  # (B*T)
        assert (
            these_flat_mask.shape
            == these_flat_labels.shape
            == these_flat_logits.shape[:-1]
        )

        # 3. Extract
        # Cross entropy loss with reduction='none' gives per-token losses
        these_flat_losses = torch.nn.functional.cross_entropy(
            these_flat_logits, these_flat_labels, reduction="none"
        )  # (B*T); fmt: skip
        assert these_flat_losses.dtype == torch.float64  # necessary unfort :P
        assert (
            these_flat_losses.shape
            == these_flat_labels.shape
            == these_flat_logits.shape[:-1]
        )
        _magic_number = -torch.rand(1, dtype=these_flat_losses.dtype).item()
        assert 0 > _magic_number
        assert not torch.any(these_flat_losses == _magic_number).item()
        these_flat_losses = these_flat_losses.where(these_flat_mask, _magic_number)
        these_per_token_losses = these_flat_losses.view(
            B, T
        )  # B x T (but now we replaced the losses
        all_per_token_losses.extend(
            [
                x[x != _magic_number].detach().cpu().numpy()
                for x in these_per_token_losses
            ]
        )
    # 3. Return
    return all_per_token_losses


def sequence_by_sequence_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> List[float]:
    """
    Gives sequence-by-sequence losses by averaging the losses within each sequence.

    (technically could be improved by doing PAIRS of items and then using process of
    elimination to solve the mean value problem)
    """
    assert _sanity_check_logits_attention_mask_and_labels(
        logits, labels, attention_mask
    )
    if not attention_mask.dtype == torch.bool:
        assert attention_mask.min() >= 0 and attention_mask.max() <= 1
        attention_mask = attention_mask.bool()
    assert attention_mask.dtype == torch.bool
    losses: List[float] = []
    for i in range(0, logits.shape[0], 1):
        these_logits = logits[i]  # T x V
        these_labels = labels[i]  # T
        these_attention_mask = attention_mask[i]  # T
        # TODO(Adriano) it is unclear if the reduction often used is sum or mean...
        # So long as we don't compare across it should be OK...
        relevant_logits = these_logits[these_attention_mask]
        relevant_labels = these_labels[these_attention_mask]
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy
        this_loss = torch.nn.functional.cross_entropy(
            relevant_logits,
            relevant_labels,
            reduction="mean",
        )
        assert this_loss.numel() == 1
        this_loss_float = this_loss.item()
        assert this_loss_float >= 0.0
        losses.append(this_loss_float)
    return losses


# TODO(Adriano) split this out into more functions maybe; I think it might be doing too
# much (also other things should use this)
# What we wnat is basically:
# 1. Any input to any output for forward pass
# 2. Any input to any output for generation
# 3. Make sure it supports hook_dict/hooking, batching, validation
# 4. Probably want to support callbacks or decide on another way to do it (i.e.
#    post-processors for metrics)
def inference_single(
    # Contents
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    chats: List[Chat] | BatchEncoding | List[str],
    # Huggingface options
    apply_chat_template_kwargs: Dict[str, Any] = {},
    tokenization_kwargs: Dict[str, Any] = {},
    generation_or_forward_kwargs: Dict[str, Any] = {},
    # Hooking options
    hook_dict: Dict[str, Any] = {},
    # TODO(Adriano) we will want the following modes in the future:
    # 1. Per-sequence loss (we can just control which sequences go into this though...)
    # 2. Per-token loss (this can be used with information about the part of the
    #    sequence that is "doing badly")
    # 3. Arbitrary metric (we should be able to support that in a simple way? Not sure
    #    if we should abstract that away into the hooks though)
    return_mode: Literal[
        # Modes for generation
        "chats",
        # Modes for forward pass
        "loss",
        "losses_batch",
        "losses_sequence",
        "losses_tokens",
    ] = "chats",
    token_by_token_loss_chunk_size: int = 5,
    return_attention_mask: bool = False,
    # If batch_size is None then it means "entire thing"
    batch_size: int | None = None,
) -> (
    List[Chat]  # chat
    | float  # loss
    | List[float]  # losses_batch
    | List[np.ndarray]  # losses_sequence?
    | Tuple[
        List[Chat] | float | List[float] | List[np.ndarray],
        torch.Tensor,
    ]
):
    """
    Single-batch inference for small samples of chats to try out.

    Takes in a set of chats and returns the generated resposnes for each chat IN the
    chat (that is to say, it extends the list of chats with the generated responses).

    It assumes the next role is "assistant" and that the entire string contents are the
    assistant's response. (TODO(Adriano) make sure this is the correct way to do it?)
    """
    if batch_size is not None:
        if isinstance(chats, BatchEncoding):
            raise ValueError("cannotbatch_size != None and BatchEncoding")
        assert isinstance(chats, list)
        if len(chats) < batch_size:
            # OK just do all
            batch_size = None
        else:
            if return_mode != "chats":
                raise NotImplementedError  # may not be supported lol
            # recurse in chunks
            # TODO(Adriano) make tqdm optional
            request_batches = [
                chats[i : min(i + batch_size, len(chats))]
                for i in range(0, len(chats), batch_size)
            ]
            response_batches = []
            for chats_batch in tqdm.tqdm(request_batches, desc="Inferring batches"):
                response_batch = inference_single(
                    # model to work on this
                    model=model,
                    tokenizer=tokenizer,
                    # data
                    chats=chats_batch,
                    # Kwargs passhrough
                    apply_chat_template_kwargs=apply_chat_template_kwargs,
                    tokenization_kwargs=tokenization_kwargs,
                    generation_or_forward_kwargs=generation_or_forward_kwargs,
                    # modifiers and options
                    hook_dict=hook_dict,
                    return_mode=return_mode,
                    token_by_token_loss_chunk_size=token_by_token_loss_chunk_size,
                    return_attention_mask=return_attention_mask,
                    batch_size=None,  # do the entire thing at once
                )
                response_batches.extend(response_batch)
            assert len(response_batches) == len(chats)
            return response_batches
    assert batch_size is None
    # 1. Fix up kwargs
    if tokenization_kwargs.get("return_tensors", "pt") != "pt":
        raise ValueError("return_tensors must be pt")
    tokenization_kwargs["return_tensors"] = "pt"
    if tokenization_kwargs.get("padding", None) is None:
        print("Setting padding to longest...")
        tokenization_kwargs["padding"] = "longest"
    assert tokenizer.padding_side == "left"
    # 2. Preprocess to go into the model
    bes = None
    if all(is_valid_chat(chat) for chat in chats):
        # 1. Template
        tkwargs = {
            "tokenize": False,
            "add_generation_prompt": return_mode == "chats",
        }
        tkwargs.update(apply_chat_template_kwargs)
        if tkwargs.get("tokenize", False):
            raise ValueError("Tokenization INSIDE the chat template is not allowed.")
        templates: List[str] = tokenizer.apply_chat_template(chats, **tkwargs)
        assert isinstance(templates, list)
        assert all(isinstance(t, str) for t in templates)
        # 2. Tokenize
        bes = tokenizer(templates, **tokenization_kwargs)
    elif isinstance(chats, BatchEncoding):
        # Do nothing here...
        bes = chats
    elif all(isinstance(chat, str) for chat in chats):
        # Assume this is a TEXTS
        bes = tokenizer(chats, **tokenization_kwargs)
    else:
        raise ValueError(f"Invalid chats type: {type(chats)}")
    assert bes is not None
    assert isinstance(bes, BatchEncoding)
    # 3. Set up arguments for model and devices
    bes = {k: v.to(model.device) for k, v in bes.items()}
    generation_or_forward_kwargs = copy.deepcopy(generation_or_forward_kwargs)
    generation_or_forward_kwargs.update(bes)
    # NOTE: we actally happen to need this for some of the generational analysis so...
    # :( we hotfix by piping it out
    # TODO(Adriano) I think we want some kind of CONTEXT instead of passing a zillion
    # parameters.
    attention_mask = (
        generation_or_forward_kwargs["attention_mask"].bool().detach().clone().cpu()
    )
    # NOTE that for the more fine-grained losses we do it MANUALLY
    if (
        return_mode in ["loss", "losses_batch"]
        and "labels" not in generation_or_forward_kwargs
    ):
        input_ids = generation_or_forward_kwargs["input_ids"]
        generation_or_forward_kwargs["labels"] = input_ids
    # 4. Generate/run inference
    with torch.no_grad():
        with named_forward_hooks(model, hook_dict):
            outputs = None
            if return_mode == "chats":
                outputs = model.generate(**generation_or_forward_kwargs)
            elif return_mode in [
                "loss",
                "losses_batch",
                "losses_sequence",
                "losses_tokens",
            ]:
                outputs = model(**generation_or_forward_kwargs)
            assert outputs is not None, f"Got return mode {return_mode} but no outputs!"
        # 5. Decode/extract
        assert outputs is not None
        if return_mode in ["chats"]:
            assert isinstance(outputs, torch.Tensor)
            responses = outputs[:, bes["input_ids"].shape[1] :]
            responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
            assert isinstance(responses, list)
            assert all(isinstance(r, str) for r in responses)
            assert len(responses) == len(chats)
            chats_out = copy.deepcopy(chats)
            for i, chat in enumerate(chats_out):
                chat.append({"role": "assistant", "content": responses[i]})
            return_value = chats_out
        elif return_mode in ["loss", "losses_batch"]:
            loss = outputs.loss
            assert loss is not None
            assert isinstance(loss, torch.Tensor), f"loss is not a tensor: {type(loss)}"
            assert (loss.ndim <= 1 and loss.numel() == 1), f"loss.shape={loss.shape}, numel={loss.numel()}"  # fmt: skip
            loss_float = loss.tolist()  # LOL
            assert isinstance(loss_float, float)
            if return_mode == "loss":
                return_value = loss_float
            elif return_mode == "losses_batch":
                return_value = [loss_float]
            else:
                raise ValueError(f"Invalid return mode: {return_mode}")
        elif return_mode in ["losses_sequence", "losses_tokens"]:
            logits = outputs.logits
            assert logits is not None
            input_ids = input_ids = generation_or_forward_kwargs["input_ids"]
            attention_mask = attention_mask = generation_or_forward_kwargs[
                "attention_mask"
            ]
            if return_mode == "losses_sequence":
                return_value = sequence_by_sequence_losses(
                    logits,
                    input_ids,
                    attention_mask,
                    # chunk size 1 becasue eh lmao
                )
            elif return_mode == "losses_tokens":
                return_value = token_by_token_losses(
                    logits,
                    input_ids,
                    attention_mask,
                    chunk_size=token_by_token_loss_chunk_size,
                )
            else:
                raise ValueError(f"Invalid return mode: {return_mode}")
        else:
            raise ValueError(f"Invalid return mode: {return_mode}")
        if return_attention_mask:
            return_value = (return_value, attention_mask)
        return return_value
