"""Auth tests for the GitHub Models PAT loader and the Anthropic key loader."""

from __future__ import annotations

import pytest

from llm_router.auth import (
    AnthropicAuthError,
    GitHubModelsAuthError,
    load_anthropic_key,
    load_github_pat,
    resolve_anthropic_key_path,
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


# ── Anthropic key loader ────────────────────────────────────────────────


def test_resolve_anthropic_key_path_default(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY_PATH", raising=False)
    resolved = resolve_anthropic_key_path()
    # Default expands ~ to the user's home directory.
    assert resolved.name == "anthropic_api_personal_key.txt"
    assert str(resolved) != "~/anthropic_api_personal_key.txt"


def test_resolve_anthropic_key_path_env(monkeypatch, tmp_path) -> None:
    p = tmp_path / "anthropic.txt"
    monkeypatch.setenv("ANTHROPIC_API_KEY_PATH", str(p))
    assert resolve_anthropic_key_path() == p


def test_resolve_anthropic_key_path_expands_tilde(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
    monkeypatch.setenv("ANTHROPIC_API_KEY_PATH", "~/my_anthropic.txt")
    resolved = resolve_anthropic_key_path()
    assert resolved == tmp_path / "my_anthropic.txt"


def test_load_anthropic_key_happy_path(tmp_path) -> None:
    p = tmp_path / "anthropic.txt"
    p.write_text("sk-ant-api03-xxxxxxxxxxxx\n", encoding="utf-8")
    assert load_anthropic_key(p) == "sk-ant-api03-xxxxxxxxxxxx"


def test_load_anthropic_key_uses_env_path(monkeypatch, tmp_path) -> None:
    p = tmp_path / "anthropic.txt"
    p.write_text("sk-ant-api03-zzz", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY_PATH", str(p))
    assert load_anthropic_key() == "sk-ant-api03-zzz"


def test_load_anthropic_key_missing_file(tmp_path) -> None:
    with pytest.raises(AnthropicAuthError):
        load_anthropic_key(tmp_path / "missing.txt")


def test_load_anthropic_key_directory(tmp_path) -> None:
    with pytest.raises(AnthropicAuthError):
        load_anthropic_key(tmp_path)


def test_load_anthropic_key_empty(tmp_path) -> None:
    p = tmp_path / "empty.txt"
    p.write_text("   \n  \n", encoding="utf-8")
    with pytest.raises(AnthropicAuthError):
        load_anthropic_key(p)


def test_load_anthropic_key_internal_whitespace(tmp_path) -> None:
    p = tmp_path / "bad.txt"
    p.write_text("sk-ant-aa bb", encoding="utf-8")
    with pytest.raises(AnthropicAuthError):
        load_anthropic_key(p)


def test_load_anthropic_key_non_standard_prefix_is_accepted(tmp_path) -> None:
    # Soft check — don't reject in case Anthropic broadens accepted formats.
    p = tmp_path / "k.txt"
    p.write_text("some-other-anthropic-token", encoding="utf-8")
    assert load_anthropic_key(p) == "some-other-anthropic-token"


def test_load_anthropic_key_string_path_expands_tilde(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    p = tmp_path / "tildey.txt"
    p.write_text("sk-ant-tildey", encoding="utf-8")
    assert load_anthropic_key("~/tildey.txt") == "sk-ant-tildey"
