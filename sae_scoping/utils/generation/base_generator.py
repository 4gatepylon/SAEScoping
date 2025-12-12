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
from pathlib import Path
from sae_scoping.utils.generation.messages import OpenAIMessages


# XXX steps of the refactor I want to do:
# 1. base class generations and then have both APIGenerations and HFGenerations
#    inheriting from it and providing one-click "generate stream" with easy
#    encode/decode for me. This provides an easy place to, in the future, add StopReason, and other
#    quantifiers; for now it should by default have
#    - caching in memory
#    - caching on disk (or just give it the option to get cache and return cache) - DONE: it's up to user
#    - stop reason
#    - map hash to arguments so that later humans can know (this should all be easily readable etc...)
#    - batching and streaming
#    - automatic encode/decode tokenize/detokenize
#    (should feel like an API basically)
#    - OpenAI server/litellm AND HF transformers
#    - everything assuems one input one output --- not clear if I want to necessarily enforce messages only or also text,
#    but nothing more than that
#   - should generate a custom sampler/iterator/something that determines the order to send stuff in
#     (or it should be clear how to do this)
#
# 2. Make the one-click base class
# 3. Make the one-click child class for existing judges
# 4. Make the one-click child class for answer preference...
# 5. Make and launch the script!
#
# XXX ok new time estimate: building this WELL is going to take 2-3 hours
# - around 30m will be spent ironing out the interfac I want etc...
# - around 30m will be spent implementing and testing things bit by bit
# - around 30m will be spent integration testing and making sure it works
# - then I will probably take longer because of bugs etc...; seems not viable?
#
# XXX I'm too brain fogged to actually do this right now so I will have to come back and do it later;
# big question is how can I finish my work effectively without doing too much and without looking too bad
# XXX I also do not fully understand what is the utility of this; don't we want like VLLM-based generation
# of some kind?

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


# XXX we should have a clean wrapper to put this stuff in here
class StopReason(str, Enum):
    STOP_REASON_STOP_TOKEN = "stop_token"
    STOP_REASON_TRUNCATION = "truncation"
    STOP_REASON_OTHER = "other"


class MessagesWrapper(pydantic.BaseModel):  # XXX fix this for both input and output
    incoming_index: int | None = None
    messages: OpenAIMessages | str
    stop_reason: StopReason | None = None
    n_input_tokens: int | None = None
    n_output_tokens: int | None = None


class BaseGenerator:
    def __init__(
        self,
        *args,
        cache: dict[str, dict[str, list[str]]] | None = {},
        generation_kwargs_cache: dict[str, JSONSerializable] | None = {},
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

    def generate_single(
        self,
        messages: Iterable[MessagesWrapper | OpenAIMessages | str],
        generation_kwargs: dict[str, Any] = {
            "min_length": -1,
            "max_new_tokens": 512,
            "do_sample": False,
        },
        return_indices: bool = False,
        subcache: dict[str, list[str]] | None = None,
    ) -> Generator[OpenAIMessages | str, None, None]:
        raise NotImplementedError("Subclasses must implement this method")

    def generate_stream(
        self,
        messages: Iterable[OpenAIMessages | str],
        batch_size: int = 32,
        generation_kwargs: dict[str, Any] = {
            "min_length": -1,
            "max_new_tokens": 512,
            "do_sample": False,
        },
        return_indices: bool = False,
    ) -> Generator[OpenAIMessages | str, None, None]:
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

        # XXX probably just turn generate stream shim into generate single and that's what users implement...?
        # XXX big open question is what happens between messages and strings?
        pass


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
