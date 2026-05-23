"""Reliability-first LLM router.

Public API:

* ``stream_chat(messages, *, tier)`` — async chat-completion stream with
  per-provider failover (Mistral primary → GitHub Models fallback by default).
* ``parse_structured(prompt, response_format, *, tier)`` — Pydantic-typed
  structured response with failover.
* ``litellm_completion(*, tier, messages, **kw)`` — LiteLLM ``completion``
  with the same provider order.
* ``make_chat_model(*, tier)`` — LangChain ``ChatLiteLLM`` runnable with
  ``.with_fallbacks([...])`` attached.

Failover is triggered only by *transient* errors (rate limit, 5xx, timeouts).
See ``errors.classify`` for the full taxonomy.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from typing import AsyncGenerator, Type, TypeVar

from pydantic import BaseModel

from .litellm_bridge import litellm_completion, make_chat_model
from .router import AllProvidersFailedError, LLMRouter, get_router, reset_router_for_tests

T = TypeVar("T", bound=BaseModel)


async def stream_chat(
    messages: list[dict],
    *,
    tier: str = "small",
    system: str | None = None,
) -> AsyncGenerator[str, None]:
    """Async wrapper around the synchronous router stream.

    The underlying SDK streams are blocking, so we run them in a thread and
    bridge chunks across to the event loop via a thread-safe queue.  This
    matches how ``conversation.py`` already structures the worker.
    """
    full_messages: list[dict] = []
    if system is not None:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    router = get_router()
    sentinel = object()
    q: "queue.Queue[object]" = queue.Queue()
    error_holder: list[BaseException] = []

    def worker() -> None:
        try:
            for chunk in router.stream_chat(tier=tier, messages=full_messages):
                q.put(chunk)
        except BaseException as exc:  # noqa: BLE001
            error_holder.append(exc)
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()
    loop = asyncio.get_event_loop()
    while True:
        chunk = await loop.run_in_executor(None, q.get)
        if chunk is sentinel:
            break
        yield chunk  # type: ignore[misc]
    if error_holder:
        raise error_holder[0]


def parse_structured(
    prompt: str,
    response_format: Type[T],
    *,
    tier: str = "large",
) -> T:
    """Structured parse with router-driven failover."""
    return get_router().parse_structured(
        tier=tier,
        prompt=prompt,
        response_format=response_format,
    )


__all__ = [
    "stream_chat",
    "parse_structured",
    "litellm_completion",
    "make_chat_model",
    "AllProvidersFailedError",
    "LLMRouter",
    "get_router",
    "reset_router_for_tests",
]
