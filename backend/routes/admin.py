"""Admin endpoints: manual triggers for background jobs.

Routes here are mounted under ``/api/admin/*`` and guarded by the existing
``_AuthMiddleware`` in ``main.py`` — any request reaching these handlers
has already passed the session-cookie check.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.post("/admin/vacancy-refresh")
async def trigger_vacancy_refresh(request: Request) -> dict:
    """Run one Adzuna vacancy refresh cycle synchronously.

    Wraps ``VacancyRefreshService.trigger_once()`` — same code path as the
    scheduled loop, but blocks the HTTP request until the cycle completes
    so the caller sees the result (processed / appended counts, summary,
    or error).

    Returns 503 if the refresh service is disabled (missing credentials
    or VACANCY_REFRESH_ENABLED=false at startup) — the handle is only
    attached when ``VacancyRefreshService.start()`` actually scheduled
    the loop.
    """
    service = getattr(request.app.state, "vacancy_refresh", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Vacancy refresh service is not enabled. "
                "Check ADZUNA_APP_ID / ADZUNA_APP_KEY / LLM key / "
                "VACANCY_REFRESH_ENABLED env vars."
            ),
        )
    return await service.trigger_once()
