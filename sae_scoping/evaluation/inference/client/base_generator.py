from __future__ import annotations
from enum import Enum
from beartype.typing import Any, Iterable, Generator
from beartype import beartype
import torch
import tqdm
import orjson
import copy
import pydantic
import hashlib
import litellm
from pathlib import Path
from sae_scoping.evaluation.inference.client.messages import OpenAIMessages

DEFAULT_GENERATION_KWARGS = {
    "min_length": -1,
    "max_new_tokens": 512,
    "do_sample": False,
}



JSONSerializable = dict[str, Any] | list[Any] | int | float | str | bool


@beartype
def is_json_serializable(d: Any) -> bool:
    try:
        orjson.dumps(d)
        return True
    except:
        return False


@beartype
def dict_hash(d: JSONSerializable | Any) -> str:
    """
    Support hashing dicts of different kinds and nestings. As long as they are JSON-serializable,
    this should work.

    TODO(Adriano) supports more general types based on some interface (should support sets, bytes, etc...)
    Also, don't use strings/hexdigest. Use raw bytes. Also this should be mathematically verified.
    """
    if d is None or isinstance(d, (int, float, str, bool)):
        return hashlib.sha256((f"{type(d)}::{d}").encode()).hexdigest()
    elif isinstance(d, (list, dict, tuple)):
        if isinstance(d, (list, tuple)):
            s = f"{type(d)}::[{','.join([dict_hash(item) for item in d])}]"
        elif isinstance(d, dict):
            s = f"{type(d)}::[{','.join([f'[{k}][{dict_hash(v)}]' for k, v in d.items()])}]"
        else:
            raise ValueError(f"(impossible code) Unsupported type: {type(d)}")
        return hashlib.sha256(s.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported type: {type(d)}")


class MessagesWrapper(pydantic.BaseModel):
    # Incoming index corresponds to index in the iterable/iterator
    # that was passed to `generate_stream`
    incoming_index: int | None = None

    # What messages is, exactly, depends on the subclass implementation.
    # Usually it will be one of the following:
    # - str
    #   all cases: same as openai messages but doesn't do/undo chat template
    # - OpenAIMessages
    #   case 1: you are using huggingface/local generator; it may return
    #     - full chats the model has seen so far
    #     - just the last message the model has seen so far
    #   case 2: you are using APIGenerator and ask to return
    #      formatted responses; it may alsso return only the last message
    # - litellm.utils.ModelResponse
    #   case 1: you are using APIGenerator and ask to return raw responses
    # - list[dict[str, Any]]
    #   structured mode with JSON responses for an API generator (usually)
    #
    # NOTE: returned as LISTS to group by prompt (so, you may request multiple
    # responses per request)
    messages: (
        list[OpenAIMessages]
        | litellm.utils.ModelResponse
        | list[str]
        | list[dict[str, Any]]
    )

    # Metadata can include stop reason, n_tokens, running costs, etc...
    # (up to the subclass to define what is allowed); possibly whether
    # responses were cached or not
    metadata: dict[str, Any] | None = None


class BaseGenerator:
    @beartype
    def __init__(
        self,
        *args,
        cache: dict[str, dict[str, list[str]]] | None = {},
        generation_kwargs_cache: dict[str, JSONSerializable] | None = {},
        n_uncached_generations_allowed: int | None = None,
        **kwargs,
    ):
        """
        - args are for subclasses to use
        - cache: cache of the generation results: ```
            {generation kwargs hash -> {question/query hash -> list of response strings}}
        ```
        - generation_kwargs_cache: cache of the generation kwargs: ```
            {generation kwargs hash -> generation kwargs}
        ```
        - kwargs are for subclasses to use
        """
        self.cache = cache
        self.generation_kwargs_cache = generation_kwargs_cache
        self.n_uncached_generations_allowed = n_uncached_generations_allowed
        if (
            self.n_uncached_generations_allowed is not None
            and self.n_uncached_generations_allowed <= 0
        ):
            raise ValueError("n_uncached_generations_allowed must be None or positive")
        self.validate_cache()

    @beartype
    def save_cache(self, filepath: Path):
        filepath.write_bytes(
            orjson.dumps(
                {
                    "cache": self.cache,
                    "generation_kwargs_cache": self.generation_kwargs_cache,
                }
            )
        )

    @beartype
    def load_cache(self, filepath: Path):
        data = orjson.loads(filepath.read_bytes())
        self.cache = data["cache"]
        self.generation_kwargs_cache = data["generation_kwargs_cache"]
        self.validate_cache()

    @beartype
    def validate_cache(self):
        # fmt: off
        # Sanity check relationship
        if not (self.cache is None) == (self.generation_kwargs_cache is None):
            raise ValueError("cache and generation_kwargs_cache must be either both None or both not None")

        # Early-stop
        if self.cache is None:
            return

        # Sanity check keys' relationship
        if not set(self.cache.keys()) <= set(self.generation_kwargs_cache.keys()):
            raise ValueError("cache keys must be a subset of generation_kwargs_cache keys")

        # Sanity check cache
        if not all(isinstance(generation_kwargs_hash, str) for generation_kwargs_hash in self.cache.keys()):
            raise ValueError("cache keys must be strings")
        if not all(isinstance(subcache, dict) for subcache in self.cache.values()):
            raise ValueError("cache values must be dicts")
        if not all(all(isinstance(subkey, str) for subkey in subcache.keys()) for subcache in self.cache.values()):
            raise ValueError("cache values must be dicts with string keys")
        if not all(all(isinstance(answers, list) for answers in subcache.values()) for subcache in self.cache.values()):
            raise ValueError("cache values must be dicts with list values")
        if not all(all(all(isinstance(answer, str) for answer in answers) for answers in subcache.values()) for subcache in self.cache.values()):
            raise ValueError("answers must be ALL strings")

        # Sanity check generation_kwargs_cache
        if not all(isinstance(kw_key, str) for kw_key in self.generation_kwargs_cache.keys()):
            raise ValueError("generation_kwargs_cache keys must be strings")
        if not all(is_json_serializable(kw_val) for kw_val in self.generation_kwargs_cache.values()):
            raise ValueError("generation_kwargs_cache values must be JSON-serializable")
        # fmt: on

    @beartype
    def generate_single(
        self,
        messages: Iterable[OpenAIMessages | str],
        generation_kwargs: dict[str, Any] = DEFAULT_GENERATION_KWARGS,
        # TODO(Adriano) unclear what we want to do exactly if the
        # subcache is there but not large enough
        #
        # NOTE: subcache should map in the following way:
        # 1. If inputs are strings then store in subcache as is
        # 2. If inputs are OpenAIMessages, key becomes the hash of the
        #    incoming messages
        subcache: dict[str, list[str]] | None = None,
        batch_start_index: int = 0,
        **kwargs,
    ) -> Generator[MessagesWrapper, None, None]:
        raise NotImplementedError("Subclasses must implement this method")

    @beartype
    def generate_stream(
        self,
        messages: Iterable[OpenAIMessages | str],
        batch_size: int = 32,
        generation_kwargs: dict[str, Any] = DEFAULT_GENERATION_KWARGS,
        **kwargs,
    ) -> Generator[MessagesWrapper, None, None]:
        # Save everything to cache if we are caching (and validate)
        # Note that JSON serializeability is not checked for no-cache mode; this is fine:
        # if you want to do something with some wierd types you can do it but we won't cache
        # for you.
        generation_kwargs_hash = None
        if self.cache is not None and not is_json_serializable(generation_kwargs):
            raise ValueError("generation_kwargs must be JSON-serializable")
        if self.cache is not None and generation_kwargs_hash not in self.cache:
            generation_kwargs_hash = dict_hash(generation_kwargs)
            self.cache[generation_kwargs_hash] = {}
            if generation_kwargs_hash not in self.generation_kwargs_cache:
                # fmt: off
                self.generation_kwargs_cache[generation_kwargs_hash] = copy.deepcopy(generation_kwargs)
                # fmt: on

        # Get the subcache and buffer (where we will be collecting answers for batch
        # processing; the idea is that we might have interleaved generations that were
        # previously generated and those that were not)
        subcache = (
            self.cache[generation_kwargs_hash] if self.cache is not None else None
        )
        generation_buffer: list[OpenAIMessages | str] = []

        raise NotImplementedError("Subclasses must implement this method")


if __name__ == "__main__":

    def integration_test_dict_hash():
        print("=" * 100)
        print("Integration test for dict_hash")
        # fmt: off
        import copy
        import itertools
        dicts = [
            # string to list
            {
                "a": [1, 2, 3]
            },
            # tuple to string
            {
                (1, 2, 3): "a",
            },
            # tuple to dict
            {
                (1, 2, 3): {
                    "a": 1,
                    "b": 2,
                    "c": [1, 3],
                },
            },
            # same tuple to dict, but different dict
            {
                (1, 2, 3): {
                    "a": 1,
                    "b": 2,
                    "c": [1, 32],
                },
            },
            # multiple elements
            {
                "a": 1,
                "b": {(1,2): True},
                "c": [1, 3],
            },
            # multiple elements, different boolean
            {
                "b": {(1,2): False},
                "a": 1,
                "c": [1, 3],
            },
            # boolean to list
            {
                True: [False, False]
            },
        ]
        dicts_deepcopies = [copy.deepcopy(d) for d in dicts]
        hashes1 = [dict_hash(d) for d in dicts]
        hashes2 = [dict_hash(d) for d in dicts_deepcopies]
        assert all(h1 == h2 for h1, h2 in zip(hashes1, hashes2))
        assert all((i == j) == (h1 == h2) for (i, h1), (j, h2) in itertools.product(enumerate(hashes1), enumerate(hashes2)))
        # fmt: on
        print("[OK] Integration test PASS for dict_hash")
        print("=" * 100)

    integration_test_dict_hash()
