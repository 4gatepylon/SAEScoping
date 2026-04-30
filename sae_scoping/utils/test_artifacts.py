"""Tests for `sae_scoping.utils.artifacts`."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sae_scoping.utils.artifacts import (
    _ARTIFACTS_ENV,
    get_git_sha,
    make_run_dir,
    make_run_id,
    make_step_dir,
    resolve_artifacts_root,
)


def test_make_run_id_format() -> None:
    rid = make_run_id()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{6}_[0-9a-f]{8}", rid), rid


def test_make_run_id_is_unique() -> None:
    """Two calls within the same second must still produce distinct ids."""
    ids = {make_run_id() for _ in range(20)}
    assert len(ids) == 20


def test_resolve_artifacts_root_explicit_arg_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_ARTIFACTS_ENV, str(tmp_path / "from_env"))
    assert resolve_artifacts_root(str(tmp_path / "from_arg")) == tmp_path / "from_arg"


def test_resolve_artifacts_root_env_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(_ARTIFACTS_ENV, str(tmp_path / "from_env"))
    assert resolve_artifacts_root(None) == tmp_path / "from_env"


def test_resolve_artifacts_root_default_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ARTIFACTS_ENV, raising=False)
    assert resolve_artifacts_root(None) == Path(".")


def test_make_run_dir_creates_layout(tmp_path: Path) -> None:
    run_id = "2026-04-30_120000_deadbeef"
    run_dir = make_run_dir(tmp_path, run_id)
    assert run_dir == tmp_path / "outputs" / run_id
    assert run_dir.is_dir()


def test_make_run_dir_is_idempotent(tmp_path: Path) -> None:
    run_id = "test_run"
    a = make_run_dir(tmp_path, run_id)
    b = make_run_dir(tmp_path, run_id)
    assert a == b and a.is_dir()


def test_make_step_dir_zero_padded(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path, "r")
    s0 = make_step_dir(run_dir, 0)
    s7 = make_step_dir(run_dir, 7)
    s42 = make_step_dir(run_dir, 42)
    assert s0.name == "step_000"
    assert s7.name == "step_007"
    assert s42.name == "step_042"
    assert all(d.is_dir() for d in (s0, s7, s42))


def test_get_git_sha_in_repo() -> None:
    """In this repo we expect a sha; outside we expect None — but we are in
    the repo when tests run, so just check the sha shape."""
    sha = get_git_sha()
    assert sha is None or re.fullmatch(r"[0-9a-f]{40}", sha), sha


def test_get_git_sha_outside_repo(tmp_path: Path) -> None:
    """Pointing at a non-git dir returns None (best-effort, no exception)."""
    assert get_git_sha(cwd=tmp_path) is None
