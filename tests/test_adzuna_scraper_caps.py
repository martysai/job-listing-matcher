"""Coverage for the per-query / global vacancy caps in ``scrape_adzuna_batch``.

The per-query cap is what keeps harvest distributed across every
(location × role) query instead of letting the first 1-2 productive
queries fill the global bucket and skip the rest.  Regressions here
would silently collapse geographic / role diversity in the cron path.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adzuna.config import AdzunaQuery
from adzuna.scraper import scrape_adzuna_batch


# ── Fake aiohttp.ClientSession ──────────────────────────────────────────────


def _build_fake_aiohttp(per_query_results: list[list[dict]]):
    """Return a fake ``aiohttp.ClientSession`` that yields prebuilt results.

    ``per_query_results[i]`` is returned for the i-th call to ``session.get``.
    Each call increments an internal counter so consecutive queries see
    different vacancy id sets.
    """
    call_idx = {"i": 0}

    def make_response(payload: list[dict]) -> MagicMock:
        resp = MagicMock()
        resp.status = 200
        resp.url = "https://example/?adzuna=mock"
        resp.json = AsyncMock(return_value={"results": payload})
        resp.text = AsyncMock(return_value="")
        return resp

    class FakeGetCtx:
        def __init__(self, payload: list[dict]) -> None:
            self._payload = payload

        async def __aenter__(self):
            return make_response(self._payload)

        async def __aexit__(self, *a):
            return None

    def fake_get(*_args, **_kwargs):
        i = call_idx["i"]
        payload = per_query_results[i] if i < len(per_query_results) else []
        call_idx["i"] += 1
        return FakeGetCtx(payload)

    fake_session = MagicMock()
    fake_session.get = fake_get

    class FakeSessionCtx:
        async def __aenter__(self):
            return fake_session

        async def __aexit__(self, *a):
            return None

    return FakeSessionCtx, call_idx


def _queries(n: int) -> list[AdzunaQuery]:
    return [
        AdzunaQuery(what=f"role{i}", where="london", category="it-jobs", country="gb")
        for i in range(n)
    ]


# ── Tests ───────────────────────────────────────────────────────────────────


def test_per_query_cap_distributes_harvest_across_all_queries() -> None:
    """4 queries × 50 results, per_query=3, global=20 → 4×3 = 12 kept."""
    # Each query returns 50 unique vacancy ids (q{i}_v{j}).
    per_query = [
        [{"id": f"q{i}_v{j}"} for j in range(50)]
        for i in range(4)
    ]
    fake_session_cls, counter = _build_fake_aiohttp(per_query)

    with patch("aiohttp.ClientSession", return_value=fake_session_cls()):
        result = asyncio.run(scrape_adzuna_batch(
            _queries(4),
            delay_seconds=0,
            max_vacancies=20,
            max_vacancies_per_query=3,
        ))

    assert len(result) == 12, "expected 4 queries × 3 per-query cap = 12 vacancies"
    # Every query contributed:
    contributors = {v["id"].split("_")[0] for v in result}
    assert contributors == {"q0", "q1", "q2", "q3"}
    assert counter["i"] == 4, "expected every query to be hit"


def test_global_cap_still_enforced_when_per_query_cap_set() -> None:
    """4 queries × 50 results, per_query=10, global=15 → stop after global hit."""
    per_query = [
        [{"id": f"q{i}_v{j}"} for j in range(50)]
        for i in range(4)
    ]
    fake_session_cls, counter = _build_fake_aiohttp(per_query)

    with patch("aiohttp.ClientSession", return_value=fake_session_cls()):
        result = asyncio.run(scrape_adzuna_batch(
            _queries(4),
            delay_seconds=0,
            max_vacancies=15,
            max_vacancies_per_query=10,
        ))

    # Q0: 10 kept (total 10), Q1: 5 kept (global cap hit), Q2/3: skipped early.
    assert len(result) == 15
    assert counter["i"] == 2, "expected scrape to short-circuit after global cap"


def test_no_caps_returns_all_unique_results() -> None:
    """Backward-compat: when both caps are None, every unique result is kept."""
    per_query = [
        [{"id": f"q{i}_v{j}"} for j in range(7)]
        for i in range(3)
    ]
    fake_session_cls, _counter = _build_fake_aiohttp(per_query)

    with patch("aiohttp.ClientSession", return_value=fake_session_cls()):
        result = asyncio.run(scrape_adzuna_batch(
            _queries(3),
            delay_seconds=0,
            max_vacancies=None,
            max_vacancies_per_query=None,
        ))

    assert len(result) == 21


def test_per_query_cap_respects_dedup_first() -> None:
    """If a query returns IDs already seen, dedup runs *before* the per-query cap."""
    # Q0 returns ids A-E; Q1 returns A,B,X,Y,Z (A,B are duplicates).
    per_query = [
        [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}, {"id": "E"}],
        [{"id": "A"}, {"id": "B"}, {"id": "X"}, {"id": "Y"}, {"id": "Z"}],
    ]
    fake_session_cls, _counter = _build_fake_aiohttp(per_query)

    with patch("aiohttp.ClientSession", return_value=fake_session_cls()):
        result = asyncio.run(scrape_adzuna_batch(
            _queries(2),
            delay_seconds=0,
            max_vacancies=None,
            max_vacancies_per_query=2,
        ))

    ids = [v["id"] for v in result]
    # Q0: keeps A, B (first 2 post-dedup).
    # Q1: A, B already seen, dedup yields X, Y, Z; per-query cap → keep X, Y.
    assert ids == ["A", "B", "X", "Y"]
