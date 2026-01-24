from __future__ import annotations


from typing import List, Dict, Union, Optional, Any, Iterator
import tqdm
import litellm
import openai
import json
import jinja2
from copy import deepcopy
from pathlib import Path

"""
This module provides functionality to do LLM Judge API calls to OpenAI, Anthropic, and
other OpenAI API-compatible language models that will usually be used for the purposes
of yielding output results on some LLM-under-test's responses (for example to check
for toxicity or for question-answer quality, etc...).
"""


def load_jinja_template(template_path: Path) -> jinja2.Template:
    """
    Load a Jinja2 template from the given path.
    """
    template_loader = jinja2.FileSystemLoader(template_path.resolve().parent.as_posix())
    template_env = jinja2.Environment(loader=template_loader)
    template_name = template_path.name
    return template_env.get_template(template_name)


class APIGenerator:
    def __init__(
        self,
        # TODO(Adriano) add caching (simple version plz)
    ):
        pass

    ################ [BEGIN] API Generate REALTIME [BEGIN] ################
    def api_generate_streaming(
        self,
        prompts: Union[str | List[str], List[List[Dict[str, str]]]],
        model: str,
        num_retries: int = 4,
        batch_size: int = 16,
        max_new_tokens: Optional[int] = None,  # Have to support legacy >:(
        enable_tqdm: bool = False,
        # For JSON use:
        # response_format={ "type": "json_object" },
        response_format: Optional[dict[str, Any]] = None,
        return_raw: bool = False,
        batch_completion_kwargs: dict[str, Any] = {},
    ) -> Iterator[str | litellm.utils.ModelResponse]:
        """
        This is a helper function to make it easy to generate using various LLM APIs
        (e.g. OpenAI, Anthropic, etc.) with built in error-handling. NOTE: it is only
        tested for OpenAI models.

        prompts can be either a list of string prompts, or it can be a list of multi-turn
        conversations in huggingface format:
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_response},
                {"role": "user", "content": user_input1},
                ...
            ]
        """
        batch_completion_kwargs = deepcopy(batch_completion_kwargs)  # to be safe, since we modify it

        # If we pass a list of prompts, convert to message format
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(prompts[0], str):
            prompts = [[{"role": "user", "content": p}] for p in prompts]

        # Legacy args
        if max_new_tokens is not None:
            if "max_tokens" in batch_completion_kwargs:
                raise ValueError(
                    "max_tokens in batch_completion_kwargs and max_new_tokens cannot be used together\n"  # fmt: skip
                    + f"{json.dumps(batch_completion_kwargs, indent=4)}"
                )
            batch_completion_kwargs["max_tokens"] = max_new_tokens
        # HOTFIX for LiteLLM
        # 02:15:21 - LiteLLM:DEBUG: utils.py:348 - RAW RESPONSE:
        #   Error code: 400 - {'error': {'message': "Unsupported parameter: 'max_tokens'
        #   is not supported with this model. Use 'max_completion_tokens' instead.",
        #   'type': 'invalid_request_error', 'param': 'max_tokens', 'code':
        #   'unsupported_parameter'}}
        if "max_tokens" in batch_completion_kwargs and model in [
            "gpt-5",
            "gpt-5-nano",
            "gpt-5-mini",
        ]:
            batch_completion_kwargs["max_completion_tokens"] = batch_completion_kwargs.pop("max_tokens")  # fmt: skip

        rng = range(0, len(prompts), batch_size)
        if enable_tqdm:
            rng = tqdm.trange(0, len(prompts), batch_size, desc=f"Generating {len(prompts)} responses with model {model} (LLMJudge)")  # fmt: skip
        for i in rng:
            real_batch_size = min(len(prompts), i + batch_size) - i
            try:
                resps = litellm.batch_completion(
                    model=model,
                    messages=prompts[i : i + batch_size],
                    num_retries=num_retries,
                    response_format=response_format,
                    **batch_completion_kwargs,
                )
                assert isinstance(resps, list), f"type(resps): {type(resps)}\n\n{resps}"
                if return_raw:
                    yield from resps
                else:
                    yield from [r.choices[0].message.content for r in resps]
            except openai.OpenAIError:
                # Error handling
                # TODO(Adriano) where can we get the status code?
                # should_retry = litellm._should_retry(e2.status_code)
                # print("Error: API failed to respond.", e2, f"should_retry: {should_retry}")
                yield from [None] * real_batch_size

    def api_generate(
        self,
        *args,
        **kwargs,
    ) -> List[str | litellm.utils.ModelResponse]:
        return list(self.api_generate_streaming(*args, **kwargs))

    def api_generate_json_mode_streaming(
        self,
        prompts: Union[List[str], List[List[Dict[str, str]]]],
        model: str,
        *args,
        **kwargs,
    ) -> Iterator[Dict[str, Any]]:
        """
        This is a helper function to make it easy to generate using various LLM APIs
        (e.g. OpenAI, Anthropic, etc.) with built in error-handling. However, it is mainly
        meant for use with OpenAI models and must have JSON formatted-outputs.

        NOTE that for JSON Mode your message should have the word "json" or something along
        those lines tbh.
        """
        if "response_format" in kwargs:
            raise ValueError("response_format is not allowed to be passed in kwargs")
        # Default arguments
        default_json_for_none = kwargs.pop("default_json_for_none", None)
        default_json_for_keys_fn = kwargs.pop("default_json_for_keys_fn", lambda _: {"error": "MissingKeys"})  # fmt: skip
        default_json_for_json_loads_decode_error_fn = kwargs.pop("default_json_for_json_loads_decode_error_fn", lambda _1, _2: {"error": "JSONDecodeError"})  # fmt: skip
        must_have_keys = kwargs.pop("must_have_keys", [])
        if "return_raw" in kwargs and kwargs["return_raw"]:
            raise ValueError("return_raw must be FALSE")
        kwargs["return_raw"] = False
        # Generate
        generations_iterator: Iterator[str] = self.api_generate_streaming(
            prompts=prompts,
            model=model,
            *args,
            response_format={"type": "json_object"},
            **kwargs,
        )
        # Yield
        for generation in generations_iterator:
            if generation is None:
                yield default_json_for_none
            else:
                assert isinstance(generation, str), f"{type(generation)}\n\n{generation}"
                try:
                    loaded = json.loads(generation)
                    if not all(k in loaded for k in must_have_keys):
                        yield default_json_for_keys_fn(loaded)
                    else:
                        yield loaded
                except json.JSONDecodeError as e:
                    yield default_json_for_json_loads_decode_error_fn(generation, e)

    def api_generate_json_mode(
        self,
        *args,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        return list(self.api_generate_json_mode_streaming(*args, **kwargs))

    ################ [END] API Generate REALTIME [END] ################
