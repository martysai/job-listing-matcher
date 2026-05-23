"""Router failover, cooldown, and streaming semantics.

Provider clients are stubbed so no SDK or network is involved.
"""

from __future__ import annotations

import threading
from typing import Iterable, Type

import pytest
from pydantic import BaseModel

from llm_router.router import AllProvidersFailedError, LLMRouter


# ── synthetic provider helpers ───────────────────────────────────────────


class _RateLimit(Exception):
    """Synthetic transient error (matches classify() by name)."""


class _AuthError(Exception):
    """Synthetic fatal error (matches AuthenticationError marker)."""


# Rename via subclass so type(exc).__name__ matches the markers in classify().
class RateLimitError(_RateLimit):
    pass


class AuthenticationError(_AuthError):
    pass


class FakeProvider:
    def __init__(
        self,
        name: str,
        *,
        stream_chunks: list[str] | None = None,
        stream_raise: BaseException | None = None,
        stream_raise_after: int = 0,
        parse_value: BaseModel | None = None,
        parse_raise: BaseException | None = None,
    ):
        self.name = name
        self._chunks = stream_chunks or []
        self._stream_raise = stream_raise
        self._stream_raise_after = stream_raise_after
        self._parse_value = parse_value
        self._parse_raise = parse_raise
        self.stream_calls = 0
        self.parse_calls = 0

    def stream_chat(self, *, tier: str, messages: list[dict]) -> Iterable[str]:
        self.stream_calls += 1

        def gen() -> Iterable[str]:
            for i, c in enumerate(self._chunks):
                if self._stream_raise is not None and i >= self._stream_raise_after:
                    raise self._stream_raise
                yield c
            if self._stream_raise is not None and self._stream_raise_after >= len(self._chunks):
                # Raise after producing all configured chunks.
                raise self._stream_raise

        return gen()

    def parse_structured(self, *, tier: str, prompt: str, response_format: Type[BaseModel]):
        self.parse_calls += 1
        if self._parse_raise is not None:
            raise self._parse_raise
        return self._parse_value


class _Profile(BaseModel):
    role: str


def _router(*, primary: FakeProvider, fallback: FakeProvider, monkeypatch) -> LLMRouter:
    # Force provider order to mistral,github with cooldown=0 so tests don't
    # depend on real-time clocks.
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistral,github")
    monkeypatch.setenv("LLM_PROVIDER_COOLDOWN_S", "0")
    return LLMRouter(
        provider_factories={
            "mistral": lambda: primary,
            "github": lambda: fallback,
        }
    )


# ── parse_structured failover ────────────────────────────────────────────


def test_parse_uses_primary_when_healthy(monkeypatch) -> None:
    primary = FakeProvider("p", parse_value=_Profile(role="dev"))
    fallback = FakeProvider("f", parse_value=_Profile(role="should-not-run"))
    router = _router(primary=primary, fallback=fallback, monkeypatch=monkeypatch)

    result = router.parse_structured(tier="large", prompt="hi", response_format=_Profile)

    assert result.role == "dev"
    assert primary.parse_calls == 1
    assert fallback.parse_calls == 0


def test_parse_fails_over_on_transient(monkeypatch) -> None:
    primary = FakeProvider("p", parse_raise=RateLimitError("429"))
    fallback = FakeProvider("f", parse_value=_Profile(role="from-fallback"))
    router = _router(primary=primary, fallback=fallback, monkeypatch=monkeypatch)

    result = router.parse_structured(tier="large", prompt="hi", response_format=_Profile)

    assert result.role == "from-fallback"
    assert primary.parse_calls == 1
    assert fallback.parse_calls == 1


def test_parse_does_not_failover_on_fatal(monkeypatch) -> None:
    primary = FakeProvider("p", parse_raise=AuthenticationError("401"))
    fallback = FakeProvider("f", parse_value=_Profile(role="should-not-run"))
    router = _router(primary=primary, fallback=fallback, monkeypatch=monkeypatch)

    with pytest.raises(AuthenticationError):
        router.parse_structured(tier="large", prompt="hi", response_format=_Profile)

    assert fallback.parse_calls == 0


def test_parse_raises_aggregate_when_all_transient(monkeypatch) -> None:
    primary = FakeProvider("p", parse_raise=RateLimitError("p-429"))
    fallback = FakeProvider("f", parse_raise=RateLimitError("f-429"))
    router = _router(primary=primary, fallback=fallback, monkeypatch=monkeypatch)

    with pytest.raises(AllProvidersFailedError) as info:
        router.parse_structured(tier="large", prompt="hi", response_format=_Profile)

    assert len(info.value.errors) == 2
    assert {name for name, _ in info.value.errors} == {"mistral", "github"}


# ── streaming failover semantics ─────────────────────────────────────────


def test_stream_uses_primary_when_healthy(monkeypatch) -> None:
    primary = FakeProvider("p", stream_chunks=["hello ", "world"])
    fallback = FakeProvider("f", stream_chunks=["should-not-run"])
    router = _router(primary=primary, fallback=fallback, monkeypatch=monkeypatch)

    out = list(router.stream_chat(tier="small", messages=[{"role": "user", "content": "hi"}]))

    assert out == ["hello ", "world"]
    assert fallback.stream_calls == 0


def test_stream_fails_over_before_first_token(monkeypatch) -> None:
    primary = FakeProvider("p", stream_chunks=[], stream_raise=RateLimitError("429"))
    fallback = FakeProvider("f", stream_chunks=["fallback-text"])
    router = _router(primary=primary, fallback=fallback, monkeypatch=monkeypatch)

    out = list(router.stream_chat(tier="small", messages=[]))

    assert out == ["fallback-text"]
    assert primary.stream_calls == 1
    assert fallback.stream_calls == 1


def test_stream_propagates_mid_stream_error(monkeypatch) -> None:
    # Primary yields one chunk then errors. The router must NOT switch to
    # the fallback mid-stream (would duplicate text in the UI); the exception
    # must propagate to the caller.
    primary = FakeProvider(
        "p",
        stream_chunks=["partial "],
        stream_raise=RateLimitError("mid-stream"),
        stream_raise_after=1,
    )
    fallback = FakeProvider("f", stream_chunks=["fallback-should-not-run"])
    router = _router(primary=primary, fallback=fallback, monkeypatch=monkeypatch)

    collected: list[str] = []
    with pytest.raises(RateLimitError):
        for chunk in router.stream_chat(tier="small", messages=[]):
            collected.append(chunk)

    assert collected == ["partial "]
    assert fallback.stream_calls == 0


def test_stream_does_not_failover_on_fatal(monkeypatch) -> None:
    primary = FakeProvider("p", stream_chunks=[], stream_raise=AuthenticationError("401"))
    fallback = FakeProvider("f", stream_chunks=["should-not-run"])
    router = _router(primary=primary, fallback=fallback, monkeypatch=monkeypatch)

    with pytest.raises(AuthenticationError):
        list(router.stream_chat(tier="small", messages=[]))

    assert fallback.stream_calls == 0


# ── cooldown ─────────────────────────────────────────────────────────────


def test_cooldown_skips_primary_on_subsequent_calls(monkeypatch) -> None:
    # First call: primary errors transiently → marked unavailable. Second
    # call within the cooldown window: primary skipped, fallback used
    # directly.
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistral,github")
    monkeypatch.setenv("LLM_PROVIDER_COOLDOWN_S", "999")

    primary = FakeProvider("p", parse_raise=RateLimitError("429"))
    fallback = FakeProvider("f", parse_value=_Profile(role="fb"))

    fake_now = [1000.0]

    def clock() -> float:
        return fake_now[0]

    router = LLMRouter(
        clock=clock,
        provider_factories={
            "mistral": lambda: primary,
            "github": lambda: fallback,
        },
    )

    router.parse_structured(tier="large", prompt="hi", response_format=_Profile)
    assert primary.parse_calls == 1

    fake_now[0] += 10  # still inside the cooldown window
    router.parse_structured(tier="large", prompt="hi", response_format=_Profile)

    # Primary not retried; fallback hit twice.
    assert primary.parse_calls == 1
    assert fallback.parse_calls == 2

    # After the cooldown lapses, primary becomes eligible again.
    fake_now[0] += 1000
    # Make primary healthy now.
    primary._parse_raise = None  # type: ignore[attr-defined]
    primary._parse_value = _Profile(role="recovered")  # type: ignore[attr-defined]
    result = router.parse_structured(tier="large", prompt="hi", response_format=_Profile)
    assert result.role == "recovered"
    assert primary.parse_calls == 2


def test_provider_init_failure_marks_cooldown(monkeypatch) -> None:
    # If the primary factory itself raises (e.g. missing MISTRAL_API_KEY),
    # the router should not crash — it should mark the provider unavailable
    # and try the next.
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistral,github")
    monkeypatch.setenv("LLM_PROVIDER_COOLDOWN_S", "0")

    fallback = FakeProvider("f", parse_value=_Profile(role="fb"))

    def boom() -> FakeProvider:
        raise RuntimeError("no key")

    router = LLMRouter(
        provider_factories={
            "mistral": boom,
            "github": lambda: fallback,
        }
    )

    result = router.parse_structured(tier="large", prompt="hi", response_format=_Profile)
    assert result.role == "fb"


# ── thread safety smoke ──────────────────────────────────────────────────


def test_router_is_callable_from_multiple_threads(monkeypatch) -> None:
    primary = FakeProvider("p", parse_value=_Profile(role="ok"))
    fallback = FakeProvider("f", parse_value=_Profile(role="never"))
    router = _router(primary=primary, fallback=fallback, monkeypatch=monkeypatch)

    results: list[str] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run() -> None:
        try:
            r = router.parse_structured(tier="large", prompt="hi", response_format=_Profile)
            with lock:
                results.append(r.role)
        except BaseException as exc:  # noqa: BLE001
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert results == ["ok"] * 16
