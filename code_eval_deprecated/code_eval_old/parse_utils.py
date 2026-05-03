from __future__ import annotations
from typing import Dict, Any, List

"""
This module provides utilities for parsing the answer from a model generation. It is
used primarily for the nemi_mvp (jailbreak+code-generation) tasks to extract the CODE
from the generation. This code will, in our case, usually be in python and delimited
by triple-tics or something akin to that. This implementation is modular, though.
"""


def parse_between_str_pattern(
    generation: str,
    pattern: str = "```",
    normalize_patterns: List[str] = ["```python", "```python3"],
) -> str:
    """
    Parse between two demarcations for where the "answer" should be extracted from.
    """
    for normalize_pattern in normalize_patterns:
        generation = generation.replace(normalize_pattern, pattern)
    if generation.count(pattern) < 2:
        raise ValueError(
            f"You must have at least two sets of triple-tics. Got this count: {generation.count(pattern)}. This is the generation:\n\n{generation}\n\n"
        )
    locations = [
        i
        for i in range(len(generation) - len(pattern) + 1)
        if generation[i : i + len(pattern)] == pattern
    ]
    assert len(locations) == generation.count(pattern)
    last2 = locations[-2], locations[-1]
    return generation[last2[0] + len(pattern) : last2[1]]


FUNC_MAP = {
    "parse_between_str_pattern": parse_between_str_pattern,
}


def parse_generation_for_code(
    generation: str,
    func_name: str = "parse_between_str_pattern",
    func_kwargs: Dict[str, Any] = {},
    pattern: str = "```",
) -> str:
    """
    Parse the generation for code using the given function and kwargs.
    """
    func = FUNC_MAP.get(func_name, None)
    if func is None:
        raise ValueError(f"Function {func_name} not found in FUNC_MAP")
    return func(generation, **func_kwargs)
