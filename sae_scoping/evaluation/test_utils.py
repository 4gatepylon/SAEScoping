"""Tests for `sae_scoping.evaluation.utils`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sae_scoping.evaluation.utils import JsonlSink


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_round_trip_ordered_exact_match(tmp_path: Path) -> None:
    """Rows written in order come back in order with the same content."""
    rows = [{"i": 0, "msg": "hello"}, {"i": 1, "msg": "world"}, {"i": 2, "nested": {"k": [1, 2, 3]}}]
    path = tmp_path / "out.jsonl"
    with JsonlSink(path) as sink:
        for r in rows:
            sink(r)
    assert _read_jsonl(path) == rows


def test_per_write_flush_is_visible_before_close(tmp_path: Path) -> None:
    """After each `sink(row)` returns, the row is on disk — readable without closing."""
    path = tmp_path / "out.jsonl"
    sink = JsonlSink(path)
    try:
        sink({"i": 0})
        # Mid-run read: file must already contain the row, not be empty.
        first_read = _read_jsonl(path)
        assert first_read == [{"i": 0}]
        sink({"i": 1})
        second_read = _read_jsonl(path)
        assert second_read == [{"i": 0}, {"i": 1}]
    finally:
        sink.close()


def test_context_manager_closes_on_normal_exit(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    with JsonlSink(path) as sink:
        sink({"i": 0})
    assert sink._f.closed


def test_context_manager_closes_on_exception(tmp_path: Path) -> None:
    """`__exit__` runs even if the `with` body raises."""
    path = tmp_path / "out.jsonl"
    sink_ref: list[JsonlSink] = []
    with pytest.raises(RuntimeError):
        with JsonlSink(path) as sink:
            sink_ref.append(sink)
            sink({"i": 0})
            raise RuntimeError("boom")
    assert sink_ref[0]._f.closed
    # The row written before the exception is still on disk.
    assert _read_jsonl(path) == [{"i": 0}]


def test_close_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    sink = JsonlSink(path)
    sink({"i": 0})
    sink.close()
    sink.close()  # second call must not raise.
    assert sink._f.closed


def test_auto_creates_parent_directory(tmp_path: Path) -> None:
    """Sink creates missing parent directories."""
    path = tmp_path / "deep" / "nested" / "out.jsonl"
    assert not path.parent.exists()
    with JsonlSink(path) as sink:
        sink({"i": 0})
    assert path.exists()
    assert _read_jsonl(path) == [{"i": 0}]


def test_appends_to_existing_file(tmp_path: Path) -> None:
    """Re-opening the same path appends rather than truncating."""
    path = tmp_path / "out.jsonl"
    with JsonlSink(path) as sink:
        sink({"i": 0})
    with JsonlSink(path) as sink:
        sink({"i": 1})
    assert _read_jsonl(path) == [{"i": 0}, {"i": 1}]
