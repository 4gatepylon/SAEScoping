"""Generic utilities for evaluation runs.

Currently only `JsonlSink` and the `Sink` type alias. Future evaluation utils
(metadata writers, run-id helpers, etc.) can live here too.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Optional

Sink = Callable[[dict[str, Any]], None]


class JsonlSink:
    """Append-only JSONL sink with per-write flush.

    Usage:
        with JsonlSink(path) as sink:
            sink({"a": 1})
            sink({"a": 2})

    Crash semantics (what survives what):
    - Every `sink(row)` call writes the JSON line and immediately flushes to
      the OS. So once `__call__` returns, that row is out of Python's userspace
      buffer and into the kernel page cache.
    - Survives a Python-level crash (uncaught exception, KeyboardInterrupt,
      assertion failure, OOMError raised in Python, etc.) — the kernel still
      holds the data and writes it to disk on its normal cadence.
    - Survives `SIGKILL` / OOM-killer / `os._exit()` — same reason: the kernel
      buffer is independent of the dying process.
    - Does NOT survive a kernel panic or power loss between flush and the
      kernel's writeback. We deliberately do not call `fsync` per write,
      because per-write fsync is slow and a debug log is not worth that cost.
      If you need durability against kernel crashes, call fsync at run end.
    - If the process dies mid-line (after a partial `write` but before its
      `flush` completes), the file may end with a truncated trailing line.
      JSONL is line-delimited by convention, so a downstream reader should
      tolerate one bad final line. Every line that was fully flushed is intact.
    - `__exit__` always closes the file (including when the `with` block exits
      via exception). `close()` is idempotent. `__del__` is a last-resort
      safety net for callers that forget the context manager.
    - Parent directory is created on construction if missing.

    NOT thread-safe. Concurrent calls to `__call__` from multiple threads can
    interleave bytes within a line and corrupt the file. If you need to fan
    out from multiple threads, wrap calls in an external lock or give each
    thread its own sink instance pointing at its own file.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._f = path.open("a", encoding="utf-8")

    @property
    def path(self) -> Path:
        return self._path

    def __call__(self, row: dict[str, Any]) -> None:
        self._f.write(json.dumps(row) + "\n")
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> JsonlSink:
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
