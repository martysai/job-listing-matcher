"""Router orchestrating provider order, cooldowns, and failover.

Reliability-first: every call tries providers in ``provider_order()``.  When a
provider raises a *transient* exception it is marked unavailable for
``cooldown_seconds()`` and the next provider is tried.  *Fatal* exceptions
propagate immediately (no point asking another provider for an authentication
failure or a malformed-request error).

The streaming path applies failover only **before the first token** is
yielded: once any text has been delivered to the caller, a switch to a
different provider would duplicate output in the UI, which is worse than a
visible error.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Generator, Iterable, Type, TypeVar

from pydantic import BaseModel

from .config import (
    PROVIDER_GITHUB,
    PROVIDER_MISTRAL,
    cooldown_seconds,
    provider_order,
)
from .errors import classify
from .providers import GitHubModelsProvider, MistralProvider, Provider

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class AllProvidersFailedError(RuntimeError):
    """Raised when every provider in the chain failed (transiently or fatally)."""

    def __init__(self, errors: list[tuple[str, BaseException]]):
        self.errors = errors
        summary = "; ".join(f"{name}: {type(exc).__name__}: {exc}" for name, exc in errors)
        super().__init__(f"All LLM providers failed: {summary}")


class LLMRouter:
    """Singleton-style router; constructs providers lazily on first use."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        provider_factories: dict[str, Callable[[], Provider]] | None = None,
    ):
        self._clock = clock
        self._lock = threading.Lock()
        self._providers: dict[str, Provider] = {}
        self._cooldown_until: dict[str, float] = {}
        self._factories: dict[str, Callable[[], Provider]] = provider_factories or {
            PROVIDER_MISTRAL: MistralProvider,
            PROVIDER_GITHUB: GitHubModelsProvider,
        }

    # ── public API ────────────────────────────────────────────────────────

    def stream_chat(
        self,
        *,
        tier: str,
        messages: list[dict],
    ) -> Generator[str, None, None]:
        """Stream chat chunks; failover only before the first token."""
        errors: list[tuple[str, BaseException]] = []
        for name in self._ordered_available():
            provider = self._get(name)
            if provider is None:
                continue
            try:
                iterator = iter(provider.stream_chat(tier=tier, messages=messages))
            except BaseException as exc:  # noqa: BLE001 — re-classified below
                if self._handle_error(name, exc, errors) == "fatal":
                    raise
                continue

            first: str | None = None
            try:
                for chunk in iterator:
                    first = chunk
                    break
            except BaseException as exc:  # noqa: BLE001
                if self._handle_error(name, exc, errors) == "fatal":
                    raise
                continue

            # First token came through (or stream ended empty): commit to this
            # provider and pass through everything else, propagating any
            # mid-stream errors.
            if first is not None:
                yield first
            for chunk in iterator:
                yield chunk
            return

        raise AllProvidersFailedError(errors)

    def parse_structured(
        self,
        *,
        tier: str,
        prompt: str,
        response_format: Type[T],
    ) -> T:
        errors: list[tuple[str, BaseException]] = []
        for name in self._ordered_available():
            provider = self._get(name)
            if provider is None:
                continue
            try:
                return provider.parse_structured(
                    tier=tier,
                    prompt=prompt,
                    response_format=response_format,
                )
            except BaseException as exc:  # noqa: BLE001
                if self._handle_error(name, exc, errors) == "fatal":
                    raise
                continue
        raise AllProvidersFailedError(errors)

    # ── internals ─────────────────────────────────────────────────────────

    def _ordered_available(self) -> list[str]:
        now = self._clock()
        with self._lock:
            return [
                name
                for name in provider_order()
                if self._cooldown_until.get(name, 0.0) <= now
            ]

    def _get(self, name: str) -> Provider | None:
        with self._lock:
            if name in self._providers:
                return self._providers[name]
            factory = self._factories.get(name)
        if factory is None:
            return None
        try:
            provider = factory()
        except BaseException as exc:  # noqa: BLE001
            logger.warning("llm_router: provider %r unavailable at init: %s", name, exc)
            self._mark_cooldown(name)
            return None
        with self._lock:
            self._providers[name] = provider
        return provider

    def _handle_error(
        self,
        name: str,
        exc: BaseException,
        errors: list[tuple[str, BaseException]],
    ) -> str:
        verdict = classify(exc)
        errors.append((name, exc))
        if verdict == "transient":
            self._mark_cooldown(name)
            logger.warning(
                "llm_router: provider %r transient error, failing over: %s: %s",
                name,
                type(exc).__name__,
                exc,
            )
        else:
            logger.error(
                "llm_router: provider %r fatal error: %s: %s",
                name,
                type(exc).__name__,
                exc,
            )
        return verdict

    def _mark_cooldown(self, name: str) -> None:
        until = self._clock() + cooldown_seconds()
        with self._lock:
            self._cooldown_until[name] = until


# ── module-level singleton ────────────────────────────────────────────────

_router: LLMRouter | None = None
_router_lock = threading.Lock()


def get_router() -> LLMRouter:
    global _router
    with _router_lock:
        if _router is None:
            _router = LLMRouter()
        return _router


def reset_router_for_tests() -> None:
    """Drop the cached router. Test-only — not part of the public API."""
    global _router
    with _router_lock:
        _router = None
