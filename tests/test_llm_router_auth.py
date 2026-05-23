"""Auth tests for the GitHub Models PAT loader."""

from __future__ import annotations

import pytest

from llm_router.auth import (
    GitHubModelsAuthError,
    load_github_pat,
    resolve_key_path,
)


def test_resolve_key_path_default(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_MODELS_KEY_PATH", raising=False)
    assert resolve_key_path().name == "key.txt"


def test_resolve_key_path_env(monkeypatch, tmp_path) -> None:
    p = tmp_path / "custom" / "k.txt"
    monkeypatch.setenv("GITHUB_MODELS_KEY_PATH", str(p))
    assert resolve_key_path() == p


def test_load_pat_happy_path(tmp_path) -> None:
    p = tmp_path / "key.txt"
    p.write_text("github_pat_abcdef\n", encoding="utf-8")
    assert load_github_pat(p) == "github_pat_abcdef"


def test_load_pat_missing_file(tmp_path) -> None:
    with pytest.raises(GitHubModelsAuthError):
        load_github_pat(tmp_path / "missing.txt")


def test_load_pat_directory(tmp_path) -> None:
    with pytest.raises(GitHubModelsAuthError):
        load_github_pat(tmp_path)


def test_load_pat_empty(tmp_path) -> None:
    p = tmp_path / "empty.txt"
    p.write_text("   \n  \n", encoding="utf-8")
    with pytest.raises(GitHubModelsAuthError):
        load_github_pat(p)


def test_load_pat_internal_whitespace(tmp_path) -> None:
    p = tmp_path / "bad.txt"
    p.write_text("github_pat_xx yy", encoding="utf-8")
    with pytest.raises(GitHubModelsAuthError):
        load_github_pat(p)


def test_load_pat_non_pat_prefix_is_accepted(tmp_path) -> None:
    # We don't reject non-PAT-shaped tokens — GitHub may add new formats.
    p = tmp_path / "k.txt"
    p.write_text("some-other-token-format", encoding="utf-8")
    assert load_github_pat(p) == "some-other-token-format"
