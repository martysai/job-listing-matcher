"""Launch the FastAPI backend with the Adzuna HTTP layer monkeypatched.

This is a thin wrapper that re-uses the production app from ``backend/main.py``
unchanged, then installs an in-memory stub for ``adzuna.scraper.scrape_adzuna_batch``
and ``adzuna.tools.scrape_adzuna_batch`` before uvicorn boots.  The result is
a live demo of the scheduled refresh loop:

    * the loop fires on the configured cadence (VACANCY_REFRESH_INTERVAL_SECONDS)
    * every cycle the ReAct agent calls the mocked HTTP and gets one synthetic
      vacancy, which is then extracted via the real LLM and merged into the
      live Chroma + BM25 index
    * structured events land in data/logs/app.jsonl exactly as they would in
      production

Use this when you don't have real Adzuna credentials but want to verify the
plumbing is correct end-to-end.

Usage:
    python scripts/run_backend_with_mock_adzuna.py --port 8000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "backend" / "services"))
os.chdir(REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

# ── Make the agent's credential gate pass ────────────────────────────────────
# The real HTTP request never fires because we monkeypatch the scrape function
# below, but adzuna.config snapshots these env vars at import-time and the
# agent refuses to run if either is empty.
if not os.environ.get("ADZUNA_APP_ID"):
    os.environ["ADZUNA_APP_ID"] = "mock-app-id"
if not os.environ.get("ADZUNA_APP_KEY"):
    os.environ["ADZUNA_APP_KEY"] = "mock-app-key"


MARKER_PREFIX = "[MOCK]"


def _build_mock_batch(queries) -> list[dict]:
    """Return one synthetic vacancy per refresh cycle.

    Each cycle generates a unique id (and unique marker title segment) based
    on the wall clock so we can see new rows accumulate in Chroma over time.
    """
    import time
    cycle_id = int(time.time())
    role = "Machine Learning Engineer"
    if queries:
        first = queries[0]
        # `AdzunaQuery` exposes `what`; fall back to a default if mocked
        role = getattr(first, "what", role) or role
    return [
        {
            "id": f"mock-{cycle_id}",
            "title": f"{MARKER_PREFIX} {role.title()} (cycle {cycle_id})",
            "description": (
                "Synthetic vacancy produced by run_backend_with_mock_adzuna.py "
                "for local end-to-end testing of the scheduled refresh loop. "
                f"Role: {role}. Stack: Python, PyTorch, Kubernetes, AWS. "
                "Strong background in machine learning systems required."
            ),
            "company": {"display_name": "Mock Refresh Demo Co"},
            "location": {
                "display_name": "London, UK",
                "area": ["UK", "Greater London", "London"],
            },
            "category": {"label": "IT Jobs", "tag": "it-jobs"},
            "contract_type": "permanent",
            "salary_min": 90000,
            "salary_max": 130000,
            "created": "2026-05-23T08:00:00Z",
            "redirect_url": f"https://example.com/jobs/mock-{cycle_id}",
        }
    ]


def _install_mock_adzuna_http() -> None:
    async def fake_scrape(queries, country="gb", delay_seconds=0.0, max_vacancies=None):
        batch = _build_mock_batch(queries)
        print(
            f"[mock-adzuna] scrape_adzuna_batch -> {len(batch)} fake vacancy/ies "
            f"(country={country}, n_queries={len(queries)})",
            flush=True,
        )
        return batch

    import adzuna.scraper as scraper_mod
    import adzuna.tools as tools_mod
    scraper_mod.scrape_adzuna_batch = fake_scrape
    tools_mod.scrape_adzuna_batch = fake_scrape
    print("[mock-adzuna] installed in adzuna.scraper and adzuna.tools", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    _install_mock_adzuna_http()

    # Import after monkeypatching so any references the app captures resolve
    # to the patched symbols. (The refresh service lazy-imports adzuna inside
    # _refresh_once, so this is fine.)
    import uvicorn
    from main import app

    print(
        f"[mock-adzuna] starting backend at http://{args.host}:{args.port} "
        f"(refresh interval={os.environ.get('VACANCY_REFRESH_INTERVAL_SECONDS', 'default')}s)",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
