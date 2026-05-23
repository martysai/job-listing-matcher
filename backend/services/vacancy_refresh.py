"""Background vacancy refresh loop.

Periodically runs the LangGraph ReAct Adzuna agent
(``adzuna.adzuna_agent.run_scheduled_job``) against the live recommender's
Chroma collection + embedder, then hot-updates the recommender's in-memory
BM25 / vacancy lookup so new postings are immediately searchable without
restarting the API.

Design constraints
------------------
- A refresh failure (any layer: env, Adzuna 4xx, LLM rate-limit, Chroma
  write error, BM25 rebuild crash) **must never** affect live request
  serving.  Every cycle is wrapped in a top-level try/except; on failure
  the previous in-memory state is preserved and the next cycle continues
  on schedule.
- No cycle overlap: each ``_refresh_once`` is awaited fully before the
  next sleep begins, so a slow cycle just stretches the gap to the next
  one rather than running two concurrently.
- Auto-disable if Adzuna / Mistral credentials are missing — the cron
  path simply logs and does nothing, instead of spam-erroring.
- All operational state is written to ``log_sink`` so the existing
  ``/api/logs`` endpoint surfaces refresh history.

Environment configuration
-------------------------
VACANCY_REFRESH_ENABLED            "true" / "false"            (default "true")
VACANCY_REFRESH_LOCATIONS          csv of "country:city" pairs
                                   default "gb:london,de:berlin,nl:amsterdam"
VACANCY_REFRESH_ROLES              csv of role keywords
                                   default "data scientist,machine learning engineer,software engineer"
VACANCY_REFRESH_INTERVAL_SECONDS   gap between cycles                (default 21600 = 6h)
VACANCY_REFRESH_INITIAL_DELAY_SECONDS  pre-first-cycle pause          (default 30)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

from services import log_sink

if TYPE_CHECKING:
    from services.recommender import RecommenderService

_log = logging.getLogger("sara.refresh")

# ── Defaults ─────────────────────────────────────────────────────────────────
# Hand-picked top European tech hubs covered by the Adzuna API.  The user
# originally asked for London / Belgrade / Moscow but Adzuna does not expose
# Serbia (rs) or Russia (ru) — see SUPPORTED_COUNTRIES in
# sara_retrieve_rerank/adzuna.py — so we fall back to the three biggest
# European hubs that ARE supported.  Override via VACANCY_REFRESH_LOCATIONS.

_DEFAULT_LOCATIONS = "gb:london,de:berlin,nl:amsterdam"
_DEFAULT_ROLES     = "data scientist,machine learning engineer,software engineer"
_DEFAULT_INTERVAL_SECONDS      = 6 * 60 * 60   # 6 hours
_DEFAULT_INITIAL_DELAY_SECONDS = 30

# Adzuna's IT category — every default role is IT.  If the operator adds
# non-IT roles via env, the agent's existing FALLBACK / POSITION_TO_QUERY
# tables would normally pick a better category, but the cron path uses a
# FixedQueryBuilder that always hits "it-jobs".  This trade-off is
# intentional: we want predictable category coverage on the cron schedule.
_DEFAULT_CATEGORY = "it-jobs"


# ══════════════════════════════════════════════════════════════════════════════

def _parse_locations(raw: str) -> list[tuple[str, str]]:
    """Parse "gb:london,de:berlin" → [("gb","london"), ("de","berlin")].

    Entries without a colon are treated as country="gb" (Adzuna default).
    Returns [] for empty / whitespace-only input.
    """
    out: list[tuple[str, str]] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            country, city = chunk.split(":", 1)
            country = country.strip().lower() or "gb"
            city    = city.strip()
        else:
            country, city = "gb", chunk
        out.append((country, city))
    return out


def _parse_roles(raw: str) -> list[str]:
    return [r.strip() for r in raw.split(",") if r.strip()]


def _build_default_queries() -> list:
    """Build the static query list for the cron path from env vars."""
    # Local import: adzuna depends on langchain_core / chromadb, which we
    # do not want to import at backend startup unless the refresh loop is
    # actually wired in by the lifespan.
    from adzuna.config import AdzunaQuery

    locations = _parse_locations(os.environ.get("VACANCY_REFRESH_LOCATIONS", _DEFAULT_LOCATIONS))
    roles     = _parse_roles(os.environ.get("VACANCY_REFRESH_ROLES", _DEFAULT_ROLES))
    return [
        AdzunaQuery(what=role, where=city, category=_DEFAULT_CATEGORY, country=country)
        for country, city in locations
        for role in roles
    ]


def _credentials_present() -> tuple[bool, str]:
    """Check that Adzuna + LLM credentials needed by the agent are set.

    The agent needs ADZUNA_APP_ID + ADZUNA_APP_KEY to hit the HTTP API and
    one of MISTRAL_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY for the
    LLM that drives the ReAct loop and the field extractor.

    Returns (ok, reason).
    """
    if not os.environ.get("ADZUNA_APP_ID") or not os.environ.get("ADZUNA_APP_KEY"):
        return False, "ADZUNA_APP_ID / ADZUNA_APP_KEY not configured"
    if not any(
        os.environ.get(k)
        for k in ("MISTRAL_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")
    ):
        return False, "no LLM provider key set (MISTRAL_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)"
    return True, ""


# ══════════════════════════════════════════════════════════════════════════════

class VacancyRefreshService:
    """Owns the async background task that drives periodic vacancy refresh.

    Lifecycle
    ---------
    ``await start()``  — schedule the loop; never raises
    ``await stop()``   — cancel + await the loop cleanly on shutdown
    ``await trigger_once()`` — run one cycle synchronously (for the admin
                                manual-trigger endpoint and for tests)

    The instance is created once in the FastAPI lifespan and shared with
    routes that want to trigger an immediate refresh.
    """

    def __init__(self, recommender: "RecommenderService"):
        self._recommender = recommender
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        # Single-flight guard: prevents the admin endpoint from triggering
        # a manual refresh while the scheduled loop is already mid-cycle.
        self._cycle_lock = asyncio.Lock()

        self._enabled = os.environ.get("VACANCY_REFRESH_ENABLED", "true").lower() != "false"
        self._interval = float(
            os.environ.get("VACANCY_REFRESH_INTERVAL_SECONDS", str(_DEFAULT_INTERVAL_SECONDS))
        )
        self._initial_delay = float(
            os.environ.get(
                "VACANCY_REFRESH_INITIAL_DELAY_SECONDS",
                str(_DEFAULT_INITIAL_DELAY_SECONDS),
            )
        )

    # ── Public lifecycle ──────────────────────────────────────────────────────

    async def start(self) -> None:
        """Kick off the background loop. Idempotent. Never raises."""
        try:
            if self._task is not None and not self._task.done():
                _log.info("vacancy refresh: already running")
                return
            if not self._enabled:
                _log.info("vacancy refresh: disabled via VACANCY_REFRESH_ENABLED=false")
                self._log_event("disabled", reason="VACANCY_REFRESH_ENABLED=false")
                return
            ok, reason = _credentials_present()
            if not ok:
                _log.info("vacancy refresh: skipped (%s)", reason)
                self._log_event("disabled", reason=reason)
                return

            self._stopping.clear()
            self._task = asyncio.create_task(self._run_loop(), name="vacancy-refresh-loop")
            _log.info(
                "vacancy refresh: scheduled  interval=%.0fs  initial_delay=%.0fs",
                self._interval, self._initial_delay,
            )
            self._log_event(
                "scheduled",
                interval_seconds=self._interval,
                initial_delay_seconds=self._initial_delay,
            )
        except Exception as exc:  # noqa: BLE001 — startup must never crash the app
            _log.exception("vacancy refresh: failed to schedule loop: %s", exc)
            self._log_event("schedule_error", error=str(exc))

    async def stop(self) -> None:
        """Cancel the loop and wait for it to exit. Never raises."""
        if self._task is None:
            return
        self._stopping.set()
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass
        finally:
            self._task = None
            _log.info("vacancy refresh: stopped")

    async def trigger_once(self) -> dict:
        """Run a single refresh cycle and return a result dict.

        Used by the admin manual-trigger endpoint and by tests.
        Respects the single-flight cycle lock — a manual trigger fired
        while the scheduled loop is mid-cycle waits for it to finish
        rather than running concurrently.
        """
        async with self._cycle_lock:
            return await self._refresh_once()

    # ── Internal loop ─────────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Sleep-then-refresh forever until cancelled."""
        try:
            # Initial delay lets the API finish warming up (DB init,
            # Chroma load, BM25 build) before we add load on top.
            await asyncio.wait_for(self._stopping.wait(), timeout=self._initial_delay)
            return  # stopping was set during initial delay
        except asyncio.TimeoutError:
            pass

        while not self._stopping.is_set():
            async with self._cycle_lock:
                await self._refresh_once()

            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=self._interval)
                return
            except asyncio.TimeoutError:
                continue

    async def _refresh_once(self) -> dict:
        """Run the agent once and merge its results. Returns a result dict.

        Catches every exception so a single cycle's failure never escapes
        into the loop or up to the lifespan.
        """
        from adzuna import FixedQueryBuilder
        from adzuna.adzuna_agent import run_scheduled_job

        started = time.time()
        try:
            queries = _build_default_queries()
        except Exception as exc:  # noqa: BLE001
            _log.exception("vacancy refresh: query plan build failed: %s", exc)
            self._log_event("error", stage="build_queries", error=str(exc))
            return {"status": "error", "error": str(exc)}

        if not queries:
            _log.info("vacancy refresh: empty query plan — skip cycle")
            self._log_event("skipped", reason="empty query plan")
            return {"status": "skipped", "reason": "empty query plan"}

        self._log_event("cycle_start", n_queries=len(queries))
        _log.info("vacancy refresh: starting cycle with %d queries", len(queries))

        try:
            result = await asyncio.to_thread(
                run_scheduled_job,
                embedder=self._recommender.embedder,
                chroma_collection=self._recommender.chroma_collection,
                query_builder=FixedQueryBuilder(queries),
            )
        except Exception as exc:  # noqa: BLE001
            _log.exception("vacancy refresh: agent crashed: %s", exc)
            self._log_event(
                "error",
                stage="agent_invoke",
                error=str(exc),
                duration_seconds=time.time() - started,
            )
            return {"status": "error", "error": str(exc)}

        # The agent itself returns {"status": "ok"|"error", ...} — error means
        # the agent caught its own exception internally; we just log and skip
        # the merge step.  In-memory state is untouched.
        if result.get("status") != "ok":
            _log.warning("vacancy refresh: agent returned non-ok: %s", result.get("error"))
            self._log_event(
                "agent_error",
                error=result.get("error"),
                duration_seconds=time.time() - started,
            )
            return result

        processed = result.get("processed_vacancies") or []
        added = 0
        if processed:
            try:
                added = await asyncio.to_thread(self._recommender.add_vacancies, processed)
            except Exception as exc:  # noqa: BLE001
                # RecommenderService.add_vacancies already swallows internally,
                # but belt-and-braces: a thread-boundary exception must never
                # take down the loop.
                _log.exception("vacancy refresh: add_vacancies wrapper failed: %s", exc)
                self._log_event(
                    "merge_error",
                    error=str(exc),
                    processed=len(processed),
                    duration_seconds=time.time() - started,
                )
                return {"status": "error", "error": str(exc)}

        duration = time.time() - started
        _log.info(
            "vacancy refresh: cycle ok — processed=%d  appended=%d  duration=%.1fs",
            len(processed), added, duration,
        )
        self._log_event(
            "cycle_done",
            processed=len(processed),
            appended=added,
            duration_seconds=duration,
            summary=result.get("summary", ""),
        )
        return {
            "status": "ok",
            "processed": len(processed),
            "appended": added,
            "duration_seconds": duration,
            "summary": result.get("summary", ""),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_event(self, event: str, **fields) -> None:
        """Write a structured event to log_sink. Never raises."""
        try:
            log_sink.append(
                ts=time.time(),
                level="info",
                logger="sara.refresh",
                component="vacancy_refresh",
                event=event,
                **fields,
            )
        except Exception:  # noqa: BLE001 — log writes are best-effort
            pass
