"""Tests for the Anthropic provider adapter (network-free).

The ``anthropic`` SDK is monkey-patched into ``sys.modules`` so the tests do
not depend on the real package's surface beyond what we assert against.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest
from pydantic import BaseModel

from llm_router.providers.anthropic import (
    _STRUCTURED_TOOL_NAME,
    AnthropicProvider,
    _split_system,
)


# ── _split_system unit tests ─────────────────────────────────────────────


def test_split_system_no_system_messages() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    system, rest = _split_system(msgs)
    assert system is None
    assert rest == msgs


def test_split_system_single() -> None:
    msgs = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
    ]
    system, rest = _split_system(msgs)
    assert system == "be helpful"
    assert rest == [{"role": "user", "content": "hi"}]


def test_split_system_joins_multiple() -> None:
    msgs = [
        {"role": "system", "content": "A"},
        {"role": "system", "content": "B"},
        {"role": "user", "content": "hi"},
    ]
    system, rest = _split_system(msgs)
    assert system == "A\n\nB"
    assert rest == [{"role": "user", "content": "hi"}]


def test_split_system_ignores_empty_system() -> None:
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "hi"},
    ]
    system, rest = _split_system(msgs)
    assert system is None
    assert rest == [{"role": "user", "content": "hi"}]


# ── AnthropicProvider with a fake SDK ────────────────────────────────────


class _FakeStreamContext:
    def __init__(self, chunks: list[str]) -> None:
        self.text_stream = iter(chunks)

    def __enter__(self) -> "_FakeStreamContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeBlock:
    def __init__(self, type_: str, name: str | None = None, input_: dict | None = None) -> None:
        self.type = type_
        self.name = name
        self.input = input_ or {}


class _FakeResponse:
    def __init__(self, content: list[_FakeBlock]) -> None:
        self.content = content


class _FakeMessages:
    def __init__(self) -> None:
        self.stream_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.stream_chunks: list[str] = ["hello ", "world"]
        self.create_response: _FakeResponse | None = None

    def stream(self, **kwargs: Any) -> _FakeStreamContext:
        self.stream_calls.append(kwargs)
        return _FakeStreamContext(self.stream_chunks)

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.create_calls.append(kwargs)
        assert self.create_response is not None, "create_response not set"
        return self.create_response


class _FakeAnthropic:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.messages = _FakeMessages()


def _install_fake_anthropic(monkeypatch) -> type[_FakeAnthropic]:
    fake_module = ModuleType("anthropic")
    fake_module.Anthropic = _FakeAnthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return _FakeAnthropic


def test_provider_construction_uses_loader(monkeypatch, tmp_path) -> None:
    _install_fake_anthropic(monkeypatch)
    key_file = tmp_path / "anth.txt"
    key_file.write_text("sk-ant-loaded-from-file", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY_PATH", str(key_file))

    provider = AnthropicProvider()

    assert provider._client.api_key == "sk-ant-loaded-from-file"  # type: ignore[attr-defined]


def test_provider_construction_accepts_explicit_key(monkeypatch) -> None:
    _install_fake_anthropic(monkeypatch)
    provider = AnthropicProvider(api_key="sk-ant-explicit")
    assert provider._client.api_key == "sk-ant-explicit"  # type: ignore[attr-defined]


def test_stream_chat_yields_chunks_and_splits_system(monkeypatch) -> None:
    _install_fake_anthropic(monkeypatch)
    provider = AnthropicProvider(api_key="sk-ant-x")
    provider._client.messages.stream_chunks = ["hi ", "there"]  # type: ignore[attr-defined]

    chunks = list(
        provider.stream_chat(
            tier="small",
            messages=[
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
        )
    )

    assert chunks == ["hi ", "there"]
    call = provider._client.messages.stream_calls[0]  # type: ignore[attr-defined]
    assert call["system"] == "be brief"
    assert call["messages"] == [{"role": "user", "content": "hi"}]
    assert call["model"] == "claude-sonnet-4-6"
    assert call["max_tokens"] >= 1


def test_stream_chat_without_system_omits_kwarg(monkeypatch) -> None:
    _install_fake_anthropic(monkeypatch)
    provider = AnthropicProvider(api_key="sk-ant-x")
    provider._client.messages.stream_chunks = ["x"]  # type: ignore[attr-defined]

    list(
        provider.stream_chat(
            tier="large",
            messages=[{"role": "user", "content": "hi"}],
        )
    )

    call = provider._client.messages.stream_calls[0]  # type: ignore[attr-defined]
    assert "system" not in call
    assert call["model"] == "claude-sonnet-4-6"


class _Profile(BaseModel):
    role: str
    seniority: str | None = None


def test_parse_structured_returns_pydantic_from_tool_use(monkeypatch) -> None:
    _install_fake_anthropic(monkeypatch)
    provider = AnthropicProvider(api_key="sk-ant-x")
    provider._client.messages.create_response = _FakeResponse(  # type: ignore[attr-defined]
        content=[
            _FakeBlock(type_="text"),  # ignored
            _FakeBlock(
                type_="tool_use",
                name=_STRUCTURED_TOOL_NAME,
                input_={"role": "data engineer", "seniority": "senior"},
            ),
        ]
    )

    result = provider.parse_structured(
        tier="large",
        prompt="extract this candidate",
        response_format=_Profile,
    )

    assert isinstance(result, _Profile)
    assert result.role == "data engineer"
    assert result.seniority == "senior"

    call = provider._client.messages.create_calls[0]  # type: ignore[attr-defined]
    assert call["tool_choice"] == {"type": "tool", "name": _STRUCTURED_TOOL_NAME}
    assert call["tools"][0]["name"] == _STRUCTURED_TOOL_NAME
    assert "properties" in call["tools"][0]["input_schema"]
    assert call["messages"] == [{"role": "user", "content": "extract this candidate"}]


def test_parse_structured_raises_when_no_tool_block(monkeypatch) -> None:
    _install_fake_anthropic(monkeypatch)
    provider = AnthropicProvider(api_key="sk-ant-x")
    provider._client.messages.create_response = _FakeResponse(  # type: ignore[attr-defined]
        content=[_FakeBlock(type_="text")]
    )

    with pytest.raises(ValueError, match="tool_use"):
        provider.parse_structured(
            tier="small",
            prompt="hi",
            response_format=_Profile,
        )
