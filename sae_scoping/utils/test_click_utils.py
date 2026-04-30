"""Tests for sae_scoping.utils.click_utils."""

from __future__ import annotations

import pytest

from sae_scoping.utils.click_utils import parse_comma_separated_strings


def test_parse_strings_csv() -> None:
    assert parse_comma_separated_strings("cuda:1,cuda:0,cuda:2") == ["cuda:0", "cuda:1", "cuda:2"]


def test_parse_strings_dedup_default() -> None:
    """Strings default to dedup (typo-tolerant for device specs)."""
    assert parse_comma_separated_strings("cuda:0,cuda:0,cuda:1") == ["cuda:0", "cuda:1"]


def test_parse_strings_no_sort() -> None:
    assert parse_comma_separated_strings("z,b,a", sort=False) == ["z", "b", "a"]


def test_parse_strings_strip_whitespace() -> None:
    assert parse_comma_separated_strings(" cuda:0 ,  cuda:1, cuda:2 ") == ["cuda:0", "cuda:1", "cuda:2"]


def test_parse_strings_drop_empty_tokens() -> None:
    """Trailing comma / double comma → empty token, dropped."""
    assert parse_comma_separated_strings("cuda:0,,cuda:1,") == ["cuda:0", "cuda:1"]


def test_parse_strings_none_returns_default() -> None:
    assert parse_comma_separated_strings(None) == []
    assert parse_comma_separated_strings(None, default=["cuda:0"]) == ["cuda:0"]


def test_parse_strings_list_input() -> None:
    """Already-a-list input passes through (after dedup + sort)."""
    assert parse_comma_separated_strings(["cuda:1", "cuda:0", "cuda:1"]) == ["cuda:0", "cuda:1"]


def test_parse_strings_raise_on_dup() -> None:
    with pytest.raises(ValueError):
        parse_comma_separated_strings("cuda:0,cuda:0", duplicates="raise")
