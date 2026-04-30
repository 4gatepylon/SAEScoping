"""Tests for `sae_scoping.utils.wandb_utils.resolve_wandb_settings`."""

from __future__ import annotations

import pytest

from sae_scoping.utils.wandb_utils import resolve_wandb_settings


def _clear_wandb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_MODE"):
        monkeypatch.delenv(k, raising=False)


def test_defaults_when_nothing_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_wandb_env(monkeypatch)
    s = resolve_wandb_settings()
    assert s == {"project": "saescoping", "mode": "online"}
    # entity/name/tags omitted on purpose — W&B picks its own defaults.


def test_explicit_arg_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "from-env")
    monkeypatch.setenv("WANDB_ENTITY", "env-team")
    monkeypatch.setenv("WANDB_MODE", "offline")
    s = resolve_wandb_settings(project="from-arg", entity="arg-team", mode="online")
    assert s["project"] == "from-arg"
    assert s["entity"] == "arg-team"
    assert s["mode"] == "online"


def test_env_used_when_arg_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_PROJECT", "deleteme")
    monkeypatch.setenv("WANDB_ENTITY", "team")
    monkeypatch.setenv("WANDB_MODE", "offline")
    s = resolve_wandb_settings()
    assert s["project"] == "deleteme"
    assert s["entity"] == "team"
    assert s["mode"] == "offline"


def test_name_passed_through_when_given(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_wandb_env(monkeypatch)
    s = resolve_wandb_settings(name="my-run")
    assert s["name"] == "my-run"


def test_name_omitted_when_not_given(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_wandb_env(monkeypatch)
    s = resolve_wandb_settings()
    assert "name" not in s


def test_tags_parsed_from_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_wandb_env(monkeypatch)
    s = resolve_wandb_settings(tags="wanda, baseline ,sparsity-sweep")
    assert s["tags"] == ["wanda", "baseline", "sparsity-sweep"]


def test_tags_omitted_when_none_or_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_wandb_env(monkeypatch)
    assert "tags" not in resolve_wandb_settings(tags=None)
    assert "tags" not in resolve_wandb_settings(tags="")
    # All-whitespace also produces no tags.
    assert "tags" not in resolve_wandb_settings(tags=" , , ")


def test_entity_omitted_when_neither_arg_nor_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_wandb_env(monkeypatch)
    s = resolve_wandb_settings()
    assert "entity" not in s
