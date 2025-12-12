from __future__ import annotations
from beartype.typing import Any, Iterable, Generator
from beartype import beartype
import copy
from pathlib import Path
from sae_scoping.utils.generation.messages import OpenAIMessages
from sae_scoping.utils.generation.base_generator import (
    BaseGenerator,
    MessagesWrapper,
    dict_hash,
    DEFAULT_GENERATION_KWARGS,
)


class HardcodedCacheGenerator(BaseGenerator):
    """
    Hardcoded map generator just uses the in-memory cache functionality and supports no other generations.

    This is basically just a cache/dictionary/thing that stores past responses to questions/queries. It is
    useful insofar as it implements the `BaseGenerator` interface so it can be swapped in for anything that
    takes in a `BaseGenerator`.
    """

    # TODO(Adriano)
    @beartype
    def generate_single(
        self,
        messages: Iterable[OpenAIMessages | str],
        generation_kwargs: dict[str, Any] = {},
        subcache: dict[str, list[str]] | None = None,
        batch_start_index: int = 0,
    ) -> Generator[MessagesWrapper, None, None]:
        if subcache is None:
            raise ValueError(
                "Subcache is None. "
                + "It looks like caching is not provided/supported/enabled, "
                + "but the `HardcodedCacheGenerator` requires it."
            )
        for i, message in enumerate(messages):
            message_hash = message if isinstance(message, str) else dict_hash(message)
            if message_hash not in subcache:
                raise ValueError(
                    f"Message hash {message_hash} not found in subcache. "
                    + "It looks like the subcache is not large enough to store all the messages."
                )
            yield MessagesWrapper(
                incoming_index=batch_start_index + i,
                messages=subcache[message_hash],
                metadata={
                    "cached": True,
                },
            )

    @staticmethod
    @beartype
    def from_map(map: dict[str, list[str] | str]) -> HardcodedCacheGenerator:
        # 1. Canonicalize map
        canonical_map = {k: [v] if isinstance(v, str) else v for k, v in map.items()}
        # TODO(Adriano) this is kind of a hack, but generally it gives you a plug and play
        # option where as long as you don't touch the generation kwargs argument, you can
        # just use the same generation kwargs for all generations; is there a cleaner way?
        phony_generation_kwargs_hash = dict_hash(DEFAULT_GENERATION_KWARGS)
        return HardcodedCacheGenerator(
            cache={phony_generation_kwargs_hash: canonical_map},
            generation_kwargs_cache={
                phony_generation_kwargs_hash: copy.deepcopy(DEFAULT_GENERATION_KWARGS)
            },
        )

    @staticmethod
    @beartype
    def from_cache(
        cache: (
            dict[str, list[str] | str]
            | Path
            | str
            | dict[str, dict[str, list[str] | str]]
        ) = {},
        **kwargs,
    ) -> HardcodedCacheGenerator:
        # 1. Load from file if path or str so now everything is a dict
        # 2. Load `generation_kwargs_cache` if applicable (default to None). Two cases:
        #    - There are no kwargs and a single arg and it's Path | str | dict[str, Any] => if a file
        #      load it and
        #    - There are no args and a single kwarg called "generation_kwargs_cache" of type
        #      Path | str | dict[str, Any] => if a file load it and use as is, otherwise use as is
        # 3. If dict[str, list[str] | str]] ensure that
        #    are also provided in kwargs (i.e. that it is not still None); otherwise if dict[str, list[str] | str]
        #    create phony args and issue a non-fatal warning
        # 4. Canonicalize cache into dict[str, dict[str, list[str] | str]]
        # 5. Ensure keys match
        # 6. Canonicalize cache into dict[str, dict[str, list[str]]]
        # 7. Call initializer
        raise NotImplementedError  # TODO(Claude)


if __name__ == "__main__":

    def integration_test_hardcoded_cache_generator():
        print("=" * 100)
        print("Integration test for hardcoded map generator")
        # TODO(Claude) set up a couple simple integration tests, loading from a file, from a dict, and with an empty one
        raise NotImplementedError
        print("[OK] Test PASSED for hardcoded map generator")
        print("=" * 100)

    integration_test_hardcoded_cache_generator()
