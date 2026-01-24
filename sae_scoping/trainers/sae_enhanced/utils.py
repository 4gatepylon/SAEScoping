from __future__ import annotations
from beartype import beartype
from beartype.typing import Any, Callable
import json


@beartype
def is_int(x: int | float) -> bool:
    return float(int(x)) == float(x)


# sanity check lol
assert is_int(1)
assert is_int(1.0)
assert not is_int(1.1)
assert is_int(0)
assert is_int(-1)
assert not is_int(0.1)


@beartype
def str_dict_diff(
    found: dict[str, Any],
    expected: dict[str, Any],
    jsonifiable_fn: Callable[[Any], str] = str,
) -> str:
    assert all(isinstance(k, str) for k in found.keys())
    assert all(isinstance(k, str) for k in expected.keys())
    found2str = {k: jsonifiable_fn(v) for k, v in found.items()}
    expected2str = {k: jsonifiable_fn(v) for k, v in expected.items()}
    found_minus_expected = {k: v for k, v in found.items() if k not in expected}
    expected_minus_found = {k: v for k, v in expected.items() if k not in found}
    not_equal = {k: f"Found: {jsonifiable_fn(v)}. Expected: {jsonifiable_fn(expected[k])}" for k, v in found.items() if v != expected[k]}
    return (
        "\n"
        + "=" * 100
        + "\n"
        + f"Found: {json.dumps(found2str, indent=4)}"
        + "\n"
        + "=" * 100
        + "\n"
        + f"Expected: {json.dumps(expected2str, indent=4)}"
        + "\n"
        + "=" * 100
        + "\n"
        + f"Difference (present in both, but not equal): {json.dumps(not_equal, indent=4)}"
        + "\n"
        + "=" * 100
        + "\n"
        + f"Difference: (Found-Expected): {json.dumps(found_minus_expected, indent=4)}"
        + "\n"
        + "=" * 100
        + "\n"
        + f"Difference: (Expected-Found): {json.dumps(expected_minus_found, indent=4)}"
        + "\n"
        + "=" * 100
        + "\n"
    )
