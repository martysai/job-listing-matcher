"""Tests for the async ``stream_chat`` wrapper in ``llm_router.__init__``.

We swap ``get_router`` so the wrapper drives a controlled fake router instead
of any real SDK.  Cancellation paths are verified by an explicit
``aclose()`` on the async generator, which is what the runtime calls when a
caller (e.g. FastAPI client disconnect) goes away.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

import llm_router
from llm_router import stream_chat


class _FakeRouter:
    """Synchronous router substitute whose stream we drive deterministically."""

    def __init__(self, *, chunks: list[str], chunk_delay: float = 0.0):
        self._chunks = chunks
        self._delay = chunk_delay
        self.chunks_consumed = 0
        self.was_closed = False
        self.was_exhausted = False
        self.observed_messages: list[list[dict]] | None = None

    def stream_chat(self, *, tier: str, messages: list[dict]):
        self.observed_messages = messages

        def gen():
            try:
                for c in self._chunks:
                    if self._delay:
                        time.sleep(self._delay)
                    self.chunks_consumed += 1
                    yield c
                self.was_exhausted = True
            except GeneratorExit:
                self.was_closed = True
                raise

        return gen()


@pytest.fixture()
def install_fake_router(monkeypatch):
    holder: dict[str, _FakeRouter] = {}

    def _install(router: _FakeRouter) -> _FakeRouter:
        holder["r"] = router
        monkeypatch.setattr(llm_router, "get_router", lambda: router)
        return router

    yield _install


@pytest.mark.asyncio
async def test_stream_chat_yields_all_chunks(install_fake_router) -> None:
    router = install_fake_router(_FakeRouter(chunks=["hi ", "there"]))

    out: list[str] = []
    async for chunk in stream_chat(
        [{"role": "user", "content": "hello"}], tier="small"
    ):
        out.append(chunk)

    assert out == ["hi ", "there"]
    assert router.was_exhausted is True


@pytest.mark.asyncio
async def test_stream_chat_prepends_system_message(install_fake_router) -> None:
    router = install_fake_router(_FakeRouter(chunks=["x"]))

    async for _ in stream_chat(
        [{"role": "user", "content": "hello"}], tier="small", system="be brief"
    ):
        pass

    assert router.observed_messages == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]


@pytest.mark.asyncio
async def test_stream_chat_propagates_provider_error(install_fake_router) -> None:
    class _BadRouter:
        def stream_chat(self, *, tier, messages):
            def gen():
                yield "first"
                raise RuntimeError("upstream blew up")

            return gen()

    install_fake_router(_BadRouter())  # type: ignore[arg-type]

    chunks: list[str] = []
    with pytest.raises(RuntimeError, match="upstream blew up"):
        async for chunk in stream_chat([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

    assert chunks == ["first"]


@pytest.mark.asyncio
async def test_stream_chat_close_stops_worker(install_fake_router) -> None:
    """If the caller closes the async generator early (e.g. HTTP client
    disconnect) the producer thread must stop consuming provider tokens."""
    router = install_fake_router(
        _FakeRouter(chunks=[f"chunk-{i}" for i in range(100)], chunk_delay=0.01)
    )

    agen = stream_chat([{"role": "user", "content": "hi"}])
    first = await agen.__anext__()
    assert first == "chunk-0"

    await agen.aclose()

    # Give the worker thread a beat to react to ``stop.set()``.
    for _ in range(50):
        if router.was_closed or router.chunks_consumed < 100:
            await asyncio.sleep(0.02)
            if router.was_closed:
                break

    # Either the generator was explicitly closed OR the worker stopped
    # pulling tokens. Both are evidence the cancellation propagated.
    assert router.chunks_consumed < 100, (
        f"worker kept consuming after aclose(); chunks_consumed="
        f"{router.chunks_consumed}"
    )
