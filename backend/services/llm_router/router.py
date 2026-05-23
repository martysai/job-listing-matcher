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

Cooldown is intentionally **per-provider**, not per-(provider, tier).  For
low-quota subscriptions (Mistral RPM, GitHub Models daily budget, Anthropic
org-level TPM) rate limits tend to be account-wide rather than per-model;
backing off the whole provider trades one possibly-redundant retry for a
much lower risk of burning quota with repeated 429s.  Revisit if usage
patterns shift toward per-model quota walls.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Callable, Generator, Iterable, Type, TypeVar

from pydantic import BaseModel

from .config import (
    PROVIDER_ANTHROPIC,
    PROVIDER_GITHUB,
    PROVIDER_MISTRAL,
    cooldown_seconds,
    provider_order,
)
from .errors import classify
from .providers import AnthropicProvider, GitHubModelsProvider, MistralProvider, Provider

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class _ProviderUnavailable(RuntimeError):
    """Recorded in the per-call ``errors`` list when a provider is skipped.

    Used so ``AllProvidersFailedError`` always carries a non-empty diagnostic
    even when the skip was caused by cooldown / factory failure rather than
    by an exception raised from the API call itself.
    """


class AllProvidersFailedError(RuntimeError):
    """Raised when every provider in the chain failed (transiently or fatally)."""

    def __init__(self, errors: list[tuple[str, BaseException]]):
        self.errors = errors
        if errors:
            summary = "; ".join(
                f"{name}: {type(exc).__name__}: {exc}" for name, exc in errors
            )
        else:
            summary = "no providers configured"
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
            PROVIDER_ANTHROPIC: AnthropicProvider,
            PROVIDER_GITHUB: GitHubModelsProvider,
            PROVIDER_MISTRAL: MistralProvider,
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
        for name, skipped_reason in self._ordered_with_skipped():
            if skipped_reason is not None:
                errors.append((name, skipped_reason))
                continue
            provider = self._get(name, errors)
            if provider is None:
                continue
            iterator: Iterable[str] | None = None
            try:
                iterator = iter(provider.stream_chat(tier=tier, messages=messages))
            except Exception as exc:
                if self._handle_error(name, exc, errors) == "fatal":
                    raise
                continue

            first: str | None = None
            try:
                for chunk in iterator:
                    first = chunk
                    break
            except Exception as exc:
                # Close the generator so the underlying HTTP context manager
                # (e.g. Mistral's ``with client.chat.stream(...)``) is exited
                # deterministically rather than waiting on GC.
                with contextlib.suppress(Exception):
                    iterator.close()  # type: ignore[union-attr]
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
        for name, skipped_reason in self._ordered_with_skipped():
            if skipped_reason is not None:
                errors.append((name, skipped_reason))
                continue
            provider = self._get(name, errors)
            if provider is None:
                continue
            try:
                return provider.parse_structured(
                    tier=tier,
                    prompt=prompt,
                    response_format=response_format,
                )
            except Exception as exc:
                if self._handle_error(name, exc, errors) == "fatal":
                    raise
                continue
        raise AllProvidersFailedError(errors)

    # ── internals ─────────────────────────────────────────────────────────

    def _ordered_with_skipped(self) -> list[tuple[str, BaseException | None]]:
        """Return ``[(provider_name, skipped_reason_or_None), ...]``.

        Providers in cooldown carry a ``_ProviderUnavailable`` so the caller
        can record *why* they were skipped instead of returning an empty
        diagnostic when everything is unavailable.

        If **every** configured provider is in cooldown, we ignore the
        cooldowns and attempt them all — better to retry and surface fresh
        errors than to fail with no useful information.
        """
        now = self._clock()
        order = provider_order()
        with self._lock:
            entries: list[tuple[str, BaseException | None]] = []
            available = 0
            for name in order:
                until = self._cooldown_until.get(name, 0.0)
                if until <= now:
                    entries.append((name, None))
                    available += 1
                else:
                    remaining = until - now
                    entries.append(
                        (
                            name,
                            _ProviderUnavailable(
                                f"in cooldown for another {remaining:.1f}s"
                            ),
                        )
                    )
            if available == 0:
                # All in cooldown — burn through cooldowns rather than fail
                # with an empty error list.
                return [(name, None) for name, _ in entries]
            return entries

    def _get(
        self,
        name: str,
        errors: list[tuple[str, BaseException]],
    ) -> Provider | None:
        with self._lock:
            if name in self._providers:
                return self._providers[name]
            factory = self._factories.get(name)
        if factory is None:
            errors.append(
                (name, _ProviderUnavailable(f"no factory registered for {name!r}"))
            )
            return None
        try:
            provider = factory()
        except Exception as exc:
            logger.warning("llm_router: provider %r unavailable at init: %s", name, exc)
            errors.append((name, exc))
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
