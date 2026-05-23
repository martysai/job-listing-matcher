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


def _three_router(
    *,
    p1: FakeProvider,
    p2: FakeProvider,
    p3: FakeProvider,
    monkeypatch,
) -> LLMRouter:
    """3-provider chain matching the real default: anthropic → github → mistral."""
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "anthropic,github,mistral")
    monkeypatch.setenv("LLM_PROVIDER_COOLDOWN_S", "0")
    return LLMRouter(
        provider_factories={
            "anthropic": lambda: p1,
            "github": lambda: p2,
            "mistral": lambda: p3,
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


# ── 3-provider chain (anthropic → github → mistral) ──────────────────────


def test_three_provider_uses_first_when_healthy(monkeypatch) -> None:
    a = FakeProvider("a", parse_value=_Profile(role="anth"))
    g = FakeProvider("g", parse_value=_Profile(role="gh"))
    m = FakeProvider("m", parse_value=_Profile(role="mistral"))
    router = _three_router(p1=a, p2=g, p3=m, monkeypatch=monkeypatch)

    result = router.parse_structured(tier="large", prompt="hi", response_format=_Profile)

    assert result.role == "anth"
    assert a.parse_calls == 1
    assert g.parse_calls == 0
    assert m.parse_calls == 0


def test_three_provider_skips_first_transient_uses_second(monkeypatch) -> None:
    a = FakeProvider("a", parse_raise=RateLimitError("anth 429"))
    g = FakeProvider("g", parse_value=_Profile(role="gh"))
    m = FakeProvider("m", parse_value=_Profile(role="mistral"))
    router = _three_router(p1=a, p2=g, p3=m, monkeypatch=monkeypatch)

    result = router.parse_structured(tier="large", prompt="hi", response_format=_Profile)

    assert result.role == "gh"
    assert a.parse_calls == 1
    assert g.parse_calls == 1
    assert m.parse_calls == 0


def test_three_provider_falls_through_to_third(monkeypatch) -> None:
    a = FakeProvider("a", parse_raise=RateLimitError("anth 429"))
    g = FakeProvider("g", parse_raise=RateLimitError("gh 503"))
    m = FakeProvider("m", parse_value=_Profile(role="mistral"))
    router = _three_router(p1=a, p2=g, p3=m, monkeypatch=monkeypatch)

    result = router.parse_structured(tier="large", prompt="hi", response_format=_Profile)

    assert result.role == "mistral"
    assert a.parse_calls == 1
    assert g.parse_calls == 1
    assert m.parse_calls == 1


def test_three_provider_aggregate_when_all_transient(monkeypatch) -> None:
    a = FakeProvider("a", parse_raise=RateLimitError("a"))
    g = FakeProvider("g", parse_raise=RateLimitError("g"))
    m = FakeProvider("m", parse_raise=RateLimitError("m"))
    router = _three_router(p1=a, p2=g, p3=m, monkeypatch=monkeypatch)

    with pytest.raises(AllProvidersFailedError) as info:
        router.parse_structured(tier="large", prompt="hi", response_format=_Profile)

    assert {name for name, _ in info.value.errors} == {"anthropic", "github", "mistral"}


def test_three_provider_first_fatal_does_not_failover(monkeypatch) -> None:
    a = FakeProvider("a", parse_raise=AuthenticationError("401"))
    g = FakeProvider("g", parse_value=_Profile(role="gh"))
    m = FakeProvider("m", parse_value=_Profile(role="mistral"))
    router = _three_router(p1=a, p2=g, p3=m, monkeypatch=monkeypatch)

    with pytest.raises(AuthenticationError):
        router.parse_structured(tier="large", prompt="hi", response_format=_Profile)

    assert g.parse_calls == 0
    assert m.parse_calls == 0


def test_three_provider_stream_failover_chains(monkeypatch) -> None:
    a = FakeProvider("a", stream_chunks=[], stream_raise=RateLimitError("a"))
    g = FakeProvider("g", stream_chunks=[], stream_raise=RateLimitError("g"))
    m = FakeProvider("m", stream_chunks=["from-mistral"])
    router = _three_router(p1=a, p2=g, p3=m, monkeypatch=monkeypatch)

    out = list(router.stream_chat(tier="small", messages=[]))

    assert out == ["from-mistral"]
    assert a.stream_calls == 1
    assert g.stream_calls == 1
    assert m.stream_calls == 1


# ── cooldown-empty diagnostics + synthetic error entries ─────────────────


def test_all_in_cooldown_still_attempts_and_aggregates(monkeypatch) -> None:
    """If every provider is in cooldown the router must not silently fail
    with an empty errors list. It should burn through the cooldowns and
    surface the freshly-collected diagnostics."""
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistral,github")
    monkeypatch.setenv("LLM_PROVIDER_COOLDOWN_S", "9999")

    primary = FakeProvider("p", parse_raise=RateLimitError("p-429"))
    fallback = FakeProvider("f", parse_raise=RateLimitError("f-429"))

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

    # First pass: both fail, both placed in cooldown.
    with pytest.raises(AllProvidersFailedError):
        router.parse_structured(tier="large", prompt="hi", response_format=_Profile)

    # Second pass (still in cooldown window): we still attempt all and get
    # a populated diagnostic, not an empty AllProvidersFailedError.
    fake_now[0] += 1  # well within cooldown
    with pytest.raises(AllProvidersFailedError) as info:
        router.parse_structured(tier="large", prompt="hi", response_format=_Profile)

    assert primary.parse_calls == 2
    assert fallback.parse_calls == 2
    assert len(info.value.errors) >= 2


def test_factory_failure_recorded_in_errors(monkeypatch) -> None:
    """Provider whose factory raises must be recorded — not silently skipped."""
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistral,github")
    monkeypatch.setenv("LLM_PROVIDER_COOLDOWN_S", "0")

    def boom() -> FakeProvider:
        raise RuntimeError("no MISTRAL_API_KEY")

    def boom2() -> FakeProvider:
        raise RuntimeError("no GITHUB_PAT")

    router = LLMRouter(
        provider_factories={
            "mistral": boom,
            "github": boom2,
        }
    )

    with pytest.raises(AllProvidersFailedError) as info:
        router.parse_structured(tier="large", prompt="hi", response_format=_Profile)

    names = {n for n, _ in info.value.errors}
    assert names == {"mistral", "github"}
    msgs = {str(e) for _, e in info.value.errors}
    assert any("MISTRAL_API_KEY" in m for m in msgs)


def test_all_providers_failed_error_has_summary_when_empty(monkeypatch) -> None:
    """Guard against the regression where everything in cooldown + no
    attempts produced ``All LLM providers failed:`` with no diagnostic."""
    err = AllProvidersFailedError([])
    assert "no providers configured" in str(err)


# ── stream iterator cleanup ──────────────────────────────────────────────


class _CloseTrackingProvider:
    """Provider whose stream returns a wrapper that records ``close()`` calls."""

    def __init__(self, name: str, chunks: list[str], raise_after: int | None = None):
        self.name = name
        self._chunks = chunks
        self._raise_after = raise_after
        self.stream_calls = 0
        self.wrapper: "_RecordingIter | None" = None

    def stream_chat(self, *, tier: str, messages: list[dict]):
        self.stream_calls += 1
        gen = self._gen()
        self.wrapper = _RecordingIter(gen)
        return self.wrapper

    def _gen(self):
        if self._raise_after == 0:
            raise RateLimitError("blow up before first yield")
        for i, c in enumerate(self._chunks):
            if self._raise_after is not None and i >= self._raise_after:
                raise RateLimitError("blow up")
            yield c

    def parse_structured(self, *, tier, prompt, response_format):
        raise NotImplementedError


class _RecordingIter:
    """Iterator that forwards next/close to a generator and records
    whether ``close()`` was invoked by the router.

    Must implement ``__next__`` itself (returning ``self`` from
    ``__iter__``) so ``iter(wrapper)`` keeps the wrapper in place and
    ``wrapper.close()`` is the call the router makes — not the inner
    generator's close().
    """

    def __init__(self, gen):
        self._gen = gen
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._gen)

    def close(self):
        self.close_calls += 1
        return self._gen.close()


def test_stream_failover_closes_primary_iterator(monkeypatch) -> None:
    """When the primary stream errors before the first token, the router
    must call ``close()`` on the iterator so any underlying HTTP context
    manager (e.g. Mistral's ``with client.chat.stream(...)``) is exited
    deterministically rather than waiting on GC."""
    primary = _CloseTrackingProvider("p", chunks=["never"], raise_after=0)
    fallback = FakeProvider("f", stream_chunks=["ok"])
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistral,github")
    monkeypatch.setenv("LLM_PROVIDER_COOLDOWN_S", "0")
    router = LLMRouter(
        provider_factories={
            "mistral": lambda: primary,
            "github": lambda: fallback,
        }
    )

    out = list(router.stream_chat(tier="small", messages=[]))

    assert out == ["ok"]
    assert primary.wrapper is not None
    assert primary.wrapper.close_calls >= 1


# ── narrowed except: KeyboardInterrupt propagates, not failover ──────────


class _KbdProvider:
    name = "kbd"

    def __init__(self) -> None:
        self.calls = 0

    def stream_chat(self, *, tier, messages):
        self.calls += 1
        raise KeyboardInterrupt("user hit ^C")

    def parse_structured(self, *, tier, prompt, response_format):
        self.calls += 1
        raise KeyboardInterrupt("user hit ^C")


def test_keyboard_interrupt_propagates_through_parse(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistral,github")
    monkeypatch.setenv("LLM_PROVIDER_COOLDOWN_S", "0")
    p = _KbdProvider()
    f = FakeProvider("f", parse_value=_Profile(role="never"))
    router = LLMRouter(
        provider_factories={"mistral": lambda: p, "github": lambda: f}
    )

    with pytest.raises(KeyboardInterrupt):
        router.parse_structured(tier="large", prompt="hi", response_format=_Profile)

    # Did NOT silently fail over to the fallback.
    assert f.parse_calls == 0


def test_keyboard_interrupt_propagates_through_stream(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistral,github")
    monkeypatch.setenv("LLM_PROVIDER_COOLDOWN_S", "0")
    p = _KbdProvider()
    f = FakeProvider("f", stream_chunks=["never"])
    router = LLMRouter(
        provider_factories={"mistral": lambda: p, "github": lambda: f}
    )

    with pytest.raises(KeyboardInterrupt):
        list(router.stream_chat(tier="small", messages=[]))

    assert f.stream_calls == 0
