"""Tests for the LiteLLM bridge: model-string assembly + failover behaviour.

We monkey-patch ``litellm.completion`` and ``langchain_litellm.ChatLiteLLM`` so
no SDKs need to be available at runtime — these tests are network-free.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from llm_router import litellm_bridge
from llm_router.config import (
    GITHUB_MODELS_BASE_URL,
    PROVIDER_GITHUB,
    PROVIDER_MISTRAL,
    tier as _tier,
)


class _RateLimitError(Exception):
    pass


# Match the classifier's marker name without depending on litellm.
class RateLimitError(_RateLimitError):
    pass


# ── litellm_completion ───────────────────────────────────────────────────


def _install_fake_litellm(monkeypatch, fake_completion) -> None:
    fake_module = ModuleType("litellm")
    fake_module.completion = fake_completion  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "litellm", fake_module)


def test_litellm_completion_uses_primary_with_correct_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistral,github")

    calls: list[dict[str, Any]] = []

    def fake_completion(**kw: Any) -> dict[str, Any]:
        calls.append(kw)
        return {"ok": True, "model": kw["model"]}

    _install_fake_litellm(monkeypatch, fake_completion)

    result = litellm_bridge.litellm_completion(
        tier="small",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0,
    )

    assert result == {"ok": True, "model": _tier("small").mistral_litellm}
    assert len(calls) == 1
    assert calls[0]["model"] == "mistral/mistral-small-latest"
    assert calls[0]["temperature"] == 0
    # No api_base / api_key for Mistral primary.
    assert "api_base" not in calls[0]
    assert "api_key" not in calls[0]


def test_litellm_completion_failover_assembles_github_kwargs(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistral,github")
    key_file = tmp_path / "key.txt"
    key_file.write_text("github_pat_XXXXXX", encoding="utf-8")
    monkeypatch.setenv("GITHUB_MODELS_KEY_PATH", str(key_file))

    calls: list[dict[str, Any]] = []

    def fake_completion(**kw: Any) -> dict[str, Any]:
        calls.append(kw)
        if len(calls) == 1:
            raise RateLimitError("primary 429")
        return {"ok": True, "model": kw["model"]}

    _install_fake_litellm(monkeypatch, fake_completion)

    result = litellm_bridge.litellm_completion(
        tier="small",
        messages=[{"role": "user", "content": "hi"}],
    )

    assert len(calls) == 2
    assert calls[0]["model"] == "mistral/mistral-small-latest"
    assert calls[1]["model"] == "openai/openai/gpt-4o-mini"
    assert calls[1]["api_base"] == GITHUB_MODELS_BASE_URL
    assert calls[1]["api_key"] == "github_pat_XXXXXX"
    assert result == {"ok": True, "model": "openai/openai/gpt-4o-mini"}


def test_litellm_completion_does_not_failover_on_fatal(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistral,github")

    calls: list[dict[str, Any]] = []

    class AuthenticationError(Exception):
        pass

    def fake_completion(**kw: Any) -> dict[str, Any]:
        calls.append(kw)
        raise AuthenticationError("401")

    _install_fake_litellm(monkeypatch, fake_completion)

    with pytest.raises(AuthenticationError):
        litellm_bridge.litellm_completion(
            tier="small",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert len(calls) == 1  # no fallback attempt


def test_litellm_completion_honours_reversed_order(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "github,mistral")
    key_file = tmp_path / "k.txt"
    key_file.write_text("github_pat_YY", encoding="utf-8")
    monkeypatch.setenv("GITHUB_MODELS_KEY_PATH", str(key_file))

    seen: list[str] = []

    def fake_completion(**kw: Any) -> dict[str, Any]:
        seen.append(kw["model"])
        return {"ok": True}

    _install_fake_litellm(monkeypatch, fake_completion)

    litellm_bridge.litellm_completion(
        tier="large",
        messages=[],
    )

    assert seen == ["openai/openai/gpt-4o"]


# ── make_chat_model ──────────────────────────────────────────────────────


class _FakeChatLiteLLM:
    """Stand-in for ``langchain_litellm.ChatLiteLLM`` that records its kwargs."""

    instances: list["_FakeChatLiteLLM"] = []

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.fallbacks: list[_FakeChatLiteLLM] | None = None
        type(self).instances.append(self)

    def with_fallbacks(self, fallbacks: list["_FakeChatLiteLLM"]) -> "_FakeChatLiteLLM":
        self.fallbacks = fallbacks
        return self


def _install_fake_chatlitellm(monkeypatch) -> None:
    _FakeChatLiteLLM.instances = []
    fake_module = ModuleType("langchain_litellm")
    fake_module.ChatLiteLLM = _FakeChatLiteLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_litellm", fake_module)


def test_make_chat_model_builds_primary_plus_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistral,github")
    key_file = tmp_path / "k.txt"
    key_file.write_text("github_pat_ZZ", encoding="utf-8")
    monkeypatch.setenv("GITHUB_MODELS_KEY_PATH", str(key_file))
    _install_fake_chatlitellm(monkeypatch)

    runnable = litellm_bridge.make_chat_model(tier="large", temperature=0)

    assert isinstance(runnable, _FakeChatLiteLLM)
    assert runnable.kwargs["model"] == "mistral/mistral-large-latest"
    assert runnable.kwargs["temperature"] == 0
    assert runnable.fallbacks is not None and len(runnable.fallbacks) == 1
    fb = runnable.fallbacks[0]
    assert fb.kwargs["model"] == "openai/openai/gpt-4o"
    assert fb.kwargs["api_base"] == GITHUB_MODELS_BASE_URL
    assert fb.kwargs["api_key"] == "github_pat_ZZ"


def test_make_chat_model_skips_fallback_if_pat_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistral,github")
    monkeypatch.setenv(
        "GITHUB_MODELS_KEY_PATH", str(tmp_path / "does_not_exist.txt")
    )
    _install_fake_chatlitellm(monkeypatch)

    runnable = litellm_bridge.make_chat_model(tier="small")

    # No fallback was attachable, so we got the bare primary back.
    assert isinstance(runnable, _FakeChatLiteLLM)
    assert runnable.fallbacks is None
    assert runnable.kwargs["model"] == "mistral/mistral-small-latest"


def test_make_chat_model_primary_github_uses_pat(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "github,mistral")
    key_file = tmp_path / "k.txt"
    key_file.write_text("github_pat_PRIMARY", encoding="utf-8")
    monkeypatch.setenv("GITHUB_MODELS_KEY_PATH", str(key_file))
    _install_fake_chatlitellm(monkeypatch)

    runnable = litellm_bridge.make_chat_model(tier="small")

    assert runnable.kwargs["model"] == "openai/openai/gpt-4o-mini"
    assert runnable.kwargs["api_base"] == GITHUB_MODELS_BASE_URL
    assert runnable.kwargs["api_key"] == "github_pat_PRIMARY"
    assert runnable.fallbacks is not None and len(runnable.fallbacks) == 1
    assert runnable.fallbacks[0].kwargs["model"] == "mistral/mistral-small-latest"


# ── make_chat_model: primary-auth-failure demotion ──────────────────────


def test_make_chat_model_primary_github_missing_pat_demotes(monkeypatch, tmp_path) -> None:
    """If GitHub Models is the primary but its PAT file is missing, the
    router-shaped contract says: don't crash, just promote the next
    provider to primary."""
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "github,mistral")
    monkeypatch.setenv(
        "GITHUB_MODELS_KEY_PATH", str(tmp_path / "does_not_exist.txt")
    )
    _install_fake_chatlitellm(monkeypatch)

    runnable = litellm_bridge.make_chat_model(tier="small")

    assert isinstance(runnable, _FakeChatLiteLLM)
    assert runnable.kwargs["model"] == "mistral/mistral-small-latest"
    # Was the sole survivor → no fallback chain attached.
    assert runnable.fallbacks is None


def test_make_chat_model_primary_anthropic_missing_key_demotes(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "anthropic,mistral")
    monkeypatch.setenv(
        "ANTHROPIC_API_KEY_PATH", str(tmp_path / "no_anthropic.txt")
    )
    _install_fake_chatlitellm(monkeypatch)

    runnable = litellm_bridge.make_chat_model(tier="large")

    assert runnable.kwargs["model"] == "mistral/mistral-large-latest"
    assert runnable.fallbacks is None


def test_make_chat_model_no_usable_providers_raises(monkeypatch, tmp_path) -> None:
    """All providers depend on missing credentials → must raise, not return
    a broken handle."""
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "anthropic,github")
    monkeypatch.setenv(
        "ANTHROPIC_API_KEY_PATH", str(tmp_path / "no_anth.txt")
    )
    monkeypatch.setenv(
        "GITHUB_MODELS_KEY_PATH", str(tmp_path / "no_gh.txt")
    )
    _install_fake_chatlitellm(monkeypatch)

    with pytest.raises(RuntimeError, match="no usable providers"):
        litellm_bridge.make_chat_model(tier="small")


def test_make_chat_model_anthropic_primary_with_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "anthropic,github,mistral")
    anth = tmp_path / "anth.txt"
    anth.write_text("sk-ant-PRIMARY", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY_PATH", str(anth))
    gh = tmp_path / "gh.txt"
    gh.write_text("github_pat_FB", encoding="utf-8")
    monkeypatch.setenv("GITHUB_MODELS_KEY_PATH", str(gh))
    _install_fake_chatlitellm(monkeypatch)

    runnable = litellm_bridge.make_chat_model(tier="large")

    assert runnable.kwargs["model"] == "anthropic/claude-sonnet-4-6"
    assert runnable.kwargs["api_key"] == "sk-ant-PRIMARY"
    assert runnable.fallbacks is not None
    assert len(runnable.fallbacks) == 2
    assert runnable.fallbacks[0].kwargs["model"] == "openai/openai/gpt-4o"
    assert runnable.fallbacks[0].kwargs["api_key"] == "github_pat_FB"
    assert runnable.fallbacks[1].kwargs["model"] == "mistral/mistral-large-latest"


# ── litellm_completion: anthropic primary + secret scrubbing ────────────


def test_litellm_completion_anthropic_primary(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "anthropic,github")
    anth = tmp_path / "anth.txt"
    anth.write_text("sk-ant-XYZ", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY_PATH", str(anth))

    calls: list[dict[str, Any]] = []

    def fake_completion(**kw: Any) -> dict[str, Any]:
        calls.append(kw)
        return {"ok": True}

    _install_fake_litellm(monkeypatch, fake_completion)

    litellm_bridge.litellm_completion(tier="large", messages=[])

    assert calls[0]["model"] == "anthropic/claude-sonnet-4-6"
    assert calls[0]["api_key"] == "sk-ant-XYZ"
    assert "api_base" not in calls[0]


def test_litellm_completion_scrubs_api_key_from_error_summary(monkeypatch, tmp_path) -> None:
    """If all providers fail and the SDK echoed the api_key into its
    exception text, the aggregated error must not leak it."""
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "github")
    secret = "github_pat_SUPERSECRET"
    key_file = tmp_path / "k.txt"
    key_file.write_text(secret, encoding="utf-8")
    monkeypatch.setenv("GITHUB_MODELS_KEY_PATH", str(key_file))

    class _Transient(Exception):
        # Class name matches a transient marker in errors.classify.
        pass

    Transient = type("TimeoutError", (_Transient,), {})

    def fake_completion(**kw: Any) -> dict[str, Any]:
        raise Transient(f"upstream barfed; api_key={kw['api_key']}")

    _install_fake_litellm(monkeypatch, fake_completion)

    with pytest.raises(RuntimeError) as info:
        litellm_bridge.litellm_completion(tier="small", messages=[])

    assert secret not in str(info.value)
    assert "***" in str(info.value)
