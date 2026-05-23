"""Unit tests for the vacancy refresh integration.

The heavy components (``adzuna.adzuna_agent.run_scheduled_job``, the live
``RecommenderService``, Chroma, BM25, HuggingFace embedder) are mocked
so tests run without external dependencies — they exercise the wiring,
the error-isolation contract, and the FixedQueryBuilder dataclass.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest


# ── Shared dummy log_sink so vacancy_refresh.import doesn't fail ─────────────

@pytest.fixture(autouse=True)
def _ensure_services_importable(monkeypatch, tmp_path):
    """Make the `services.*` and `adzuna.*` packages importable in tests.

    The project's pyproject only puts ``backend/services`` on the pythonpath,
    which exposes ``adzuna`` and ``sara_retrieve_rerank`` as top-level
    packages — but not ``services`` (the parent dir is one level up).  We
    add ``backend`` to sys.path here so ``from services.vacancy_refresh
    import ...`` works.

    Also points LOG_PATH at a tmpdir so log_sink writes don't pollute the
    real log file.
    """
    backend_dir = Path(__file__).resolve().parent.parent / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "test.jsonl"))


# ══════════════════════════════════════════════════════════════════════════════
# FixedQueryBuilder
# ══════════════════════════════════════════════════════════════════════════════

def test_fixed_query_builder_returns_supplied_queries():
    from adzuna import AdzunaQuery, FixedQueryBuilder

    queries = [
        AdzunaQuery("data scientist", "london", "it-jobs", country="gb"),
        AdzunaQuery("ml engineer",    "berlin", "it-jobs", country="de"),
    ]
    qb = FixedQueryBuilder(queries)
    built = qb.build()

    assert built == queries
    # Returned list must be a defensive copy — caller mutations must not
    # leak into subsequent build() calls.
    built.clear()
    assert qb.build() == queries


def test_adzuna_query_country_field_defaults_to_none():
    from adzuna import AdzunaQuery

    q = AdzunaQuery("software engineer", "amsterdam", "it-jobs")
    assert q.country is None
    # to_params() must not include "country" — country is a routing hint
    # for the scraper, not an Adzuna query-string parameter.
    assert "country" not in q.to_params()


# ══════════════════════════════════════════════════════════════════════════════
# vacancy_refresh helpers
# ══════════════════════════════════════════════════════════════════════════════

def test_parse_locations_handles_country_prefix_and_plain():
    from services.vacancy_refresh import _parse_locations

    assert _parse_locations("gb:london,de:berlin,nl:amsterdam") == [
        ("gb", "london"),
        ("de", "berlin"),
        ("nl", "amsterdam"),
    ]
    # Plain entries (no colon) default to gb.
    assert _parse_locations("london, paris") == [("gb", "london"), ("gb", "paris")]
    # Empty input → empty list, no crashes.
    assert _parse_locations("") == []
    assert _parse_locations("  ,  ") == []


def test_build_default_queries_produces_cartesian_product(monkeypatch):
    from services.vacancy_refresh import _build_default_queries

    monkeypatch.setenv("VACANCY_REFRESH_LOCATIONS", "gb:london,de:berlin")
    monkeypatch.setenv("VACANCY_REFRESH_ROLES", "data scientist,ml engineer")

    queries = _build_default_queries()
    pairs = {(q.country, q.where, q.what) for q in queries}
    assert pairs == {
        ("gb", "london", "data scientist"),
        ("gb", "london", "ml engineer"),
        ("de", "berlin", "data scientist"),
        ("de", "berlin", "ml engineer"),
    }
    # Every default query goes to the it-jobs category.
    assert {q.category for q in queries} == {"it-jobs"}


def test_credentials_present_requires_adzuna_and_llm(monkeypatch):
    from services.vacancy_refresh import _credentials_present

    for key in (
        "ADZUNA_APP_ID", "ADZUNA_APP_KEY",
        "MISTRAL_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    ok, reason = _credentials_present()
    assert not ok and "ADZUNA" in reason

    monkeypatch.setenv("ADZUNA_APP_ID", "x")
    monkeypatch.setenv("ADZUNA_APP_KEY", "y")
    ok, reason = _credentials_present()
    assert not ok and "LLM" in reason

    monkeypatch.setenv("MISTRAL_API_KEY", "z")
    ok, reason = _credentials_present()
    assert ok and reason == ""


# ══════════════════════════════════════════════════════════════════════════════
# VacancyRefreshService — full cycle wiring, error isolation, lifecycle
# ══════════════════════════════════════════════════════════════════════════════

class _StubRecommender:
    """Stand-in for RecommenderService that records add_vacancies calls."""

    def __init__(self, *, add_raises: bool = False):
        self.embedder = object()
        self.chroma_collection = object()
        self._add_raises = add_raises
        self.added: list[list[dict]] = []

    def add_vacancies(self, new_vacancies):
        if self._add_raises:
            raise RuntimeError("simulated merge failure")
        # Store a copy of the batch so test assertions see exact contents
        # even if the caller mutates the list afterwards.
        self.added.append(list(new_vacancies))
        return len(new_vacancies)


def _make_service(monkeypatch, recommender, *, env=None):
    """Build a VacancyRefreshService with credential gates pre-passed."""
    monkeypatch.setenv("ADZUNA_APP_ID", "x")
    monkeypatch.setenv("ADZUNA_APP_KEY", "y")
    monkeypatch.setenv("MISTRAL_API_KEY", "z")
    monkeypatch.setenv("VACANCY_REFRESH_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("VACANCY_REFRESH_INITIAL_DELAY_SECONDS", "0")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    # Late import so the env vars above are applied at construction time.
    from services.vacancy_refresh import VacancyRefreshService
    return VacancyRefreshService(recommender)


def test_trigger_once_calls_agent_and_merges_processed(monkeypatch):
    rec = _StubRecommender()
    svc = _make_service(monkeypatch, rec)

    captured: dict = {}

    def fake_run_scheduled_job(*, embedder, chroma_collection, query_builder):
        captured["embedder"] = embedder
        captured["chroma_collection"] = chroma_collection
        captured["queries"] = query_builder.build()
        return {
            "status": "ok",
            "summary": "ok",
            "processed_vacancies": [
                {"dataset_id": "a1", "title": "Data Scientist"},
                {"dataset_id": "a2", "title": "ML Engineer"},
            ],
        }

    # Patch the symbol on the agent module — vacancy_refresh imports it
    # locally inside _refresh_once via `from adzuna.adzuna_agent import
    # run_scheduled_job`, so we patch the source module.
    import adzuna.adzuna_agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_scheduled_job", fake_run_scheduled_job)

    result = asyncio.run(svc.trigger_once())

    assert result["status"] == "ok"
    assert result["processed"] == 2
    assert result["appended"] == 2
    assert captured["embedder"] is rec.embedder
    assert captured["chroma_collection"] is rec.chroma_collection
    assert len(captured["queries"]) >= 1  # FixedQueryBuilder produced a plan
    assert rec.added == [
        [
            {"dataset_id": "a1", "title": "Data Scientist"},
            {"dataset_id": "a2", "title": "ML Engineer"},
        ]
    ]


def test_trigger_once_agent_crash_does_not_propagate(monkeypatch):
    """Exception inside the agent must be caught and reported, not raised."""
    rec = _StubRecommender()
    svc = _make_service(monkeypatch, rec)

    def boom(*, embedder, chroma_collection, query_builder):
        raise RuntimeError("adzuna api on fire")

    import adzuna.adzuna_agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_scheduled_job", boom)

    result = asyncio.run(svc.trigger_once())
    assert result["status"] == "error"
    assert "fire" in result["error"]
    # The recommender must remain untouched on failure.
    assert rec.added == []


def test_trigger_once_handles_agent_returning_error_status(monkeypatch):
    """Agent's own internal error status is logged and merge is skipped."""
    rec = _StubRecommender()
    svc = _make_service(monkeypatch, rec)

    def returns_error(*, embedder, chroma_collection, query_builder):
        return {"status": "error", "error": "credentials missing"}

    import adzuna.adzuna_agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_scheduled_job", returns_error)

    result = asyncio.run(svc.trigger_once())
    assert result["status"] == "error"
    assert rec.added == []


def test_trigger_once_handles_merge_failure(monkeypatch):
    """add_vacancies raising must not propagate out of the refresh service."""
    rec = _StubRecommender(add_raises=True)
    svc = _make_service(monkeypatch, rec)

    def ok_run(*, embedder, chroma_collection, query_builder):
        return {
            "status": "ok",
            "summary": "ok",
            "processed_vacancies": [{"dataset_id": "x", "title": "T"}],
        }

    import adzuna.adzuna_agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_scheduled_job", ok_run)

    result = asyncio.run(svc.trigger_once())
    assert result["status"] == "error"
    assert "simulated" in result["error"]


def test_start_disables_when_credentials_missing(monkeypatch):
    """start() must be a silent no-op (not raise) when creds are absent."""
    for key in (
        "ADZUNA_APP_ID", "ADZUNA_APP_KEY",
        "MISTRAL_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VACANCY_REFRESH_INITIAL_DELAY_SECONDS", "0")

    from services.vacancy_refresh import VacancyRefreshService
    svc = VacancyRefreshService(_StubRecommender())

    async def go():
        await svc.start()
        # No task should have been scheduled.
        assert svc._task is None
        await svc.stop()  # must be safe to call

    asyncio.run(go())


def test_start_disables_when_env_flag_false(monkeypatch):
    rec = _StubRecommender()
    monkeypatch.setenv("ADZUNA_APP_ID", "x")
    monkeypatch.setenv("ADZUNA_APP_KEY", "y")
    monkeypatch.setenv("MISTRAL_API_KEY", "z")
    monkeypatch.setenv("VACANCY_REFRESH_ENABLED", "false")
    from services.vacancy_refresh import VacancyRefreshService
    svc = VacancyRefreshService(rec)

    async def go():
        await svc.start()
        assert svc._task is None

    asyncio.run(go())


# ══════════════════════════════════════════════════════════════════════════════
# RecommendationPipeline.add_vacancies — atomic merge contract
# ══════════════════════════════════════════════════════════════════════════════

def test_pipeline_add_vacancies_atomic_swap_and_dedup():
    """Verify the lock-protected swap appends new entries and dedups by id.

    Uses a hand-rolled instance without going through __init__ so we don't
    need a Chroma store or HuggingFace embedder for this unit test.
    """
    import threading
    from sara_retrieve_rerank.pipeline import RecommendationPipeline
    from sara_retrieve_rerank.documents import create_vacancy_documents
    from sara_retrieve_rerank.bm25_retrieval import BM25Index

    base = [
        {"dataset_id": "1", "title": "Data Scientist", "tldr_sanitized": "DS"},
        {"dataset_id": "2", "title": "ML Engineer",    "tldr_sanitized": "ML"},
    ]
    pipe = RecommendationPipeline.__new__(RecommendationPipeline)
    pipe.vacancies = list(base)
    pipe.documents = create_vacancy_documents(pipe.vacancies)
    pipe.bm25 = BM25Index(pipe.documents)
    pipe.vectorstore = None
    pipe.reranker_model = None
    pipe._add_lock = threading.Lock()

    new = [
        {"dataset_id": "2", "title": "ML Engineer dup", "tldr_sanitized": "dup"},  # already present
        {"dataset_id": "3", "title": "SWE",             "tldr_sanitized": "Senior SWE"},
    ]
    added = pipe.add_vacancies(new)
    assert added == 1
    assert {v["dataset_id"] for v in pipe.vacancies} == {"1", "2", "3"}
    # BM25 was rebuilt over the merged corpus, not stale.
    assert len(pipe.bm25.documents) == 3

    # Empty input is a no-op.
    assert pipe.add_vacancies([]) == 0
    assert len(pipe.vacancies) == 3
