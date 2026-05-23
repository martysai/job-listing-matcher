"""End-to-end local sanity check for the Adzuna cron refresh integration.

Boots the live RecommenderService (real HuggingFace embedder, real Chroma,
real BM25 over the seed corpus), monkeypatches only the Adzuna HTTP layer
to return one synthetic vacancy with a unique marker title, then drives
the full ``VacancyRefreshService.trigger_once()`` pipeline:

    LangGraph ReAct agent → @tool scrape_vacancies (mocked Adzuna HTTP)
                          → @tool check_scrape_quality
                          → @tool extract_fields (real Mistral LLM)
                          → @tool index_to_vector_store (real Chroma upsert)
    → RecommenderService.add_vacancies (atomic BM25 swap, lookup update)
    → RecommenderService.search(...)
    → assert the unique marker title appears in the search results

Prints the log_sink events the loop wrote so they match what the running
backend would emit on its scheduled cycle.

Usage:
    .venv/Scripts/python scripts/e2e_mock_refresh.py        # full run
    .venv/Scripts/python scripts/e2e_mock_refresh.py --quiet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "backend" / "services"))
os.chdir(REPO_ROOT)

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

# The agent's credential gate refuses to run with an empty ADZUNA_APP_ID,
# but the mock bypasses the actual HTTP call — so set placeholders so
# the credential check passes. Use direct assignment because .env loads
# ADZUNA_APP_ID as an empty string (present but falsy), which makes
# os.environ.setdefault a no-op.
if not os.environ.get("ADZUNA_APP_ID"):
    os.environ["ADZUNA_APP_ID"] = "e2e-mock-app-id"
if not os.environ.get("ADZUNA_APP_KEY"):
    os.environ["ADZUNA_APP_KEY"] = "e2e-mock-app-key"


# ── Unique marker so we can prove retrieval after merge ───────────────────────
# Includes nonsense bigrams ("Quantum Sandwich") that should never match any
# seed vacancy; if recommender.search returns this row, the merge worked.
MARKER_TITLE = "[E2E-MOCK] Quantum Sandwich Engineer"
MOCK_VACANCY_ID = "e2e-mock-001"


def _build_mock_adzuna_response() -> list[dict]:
    """One synthetic raw Adzuna dict shaped like the real /search payload."""
    return [
        {
            "id": MOCK_VACANCY_ID,
            "title": MARKER_TITLE,
            "description": (
                "We are looking for a senior Quantum Sandwich Engineer to "
                "work on machine learning systems for distributed sandwich "
                "assembly. Requirements: Python, PyTorch, Kubernetes. "
                "Strong background in deep learning required."
            ),
            "company": {"display_name": "Acme E2E Labs"},
            "location": {
                "display_name": "London, UK",
                "area": ["UK", "Greater London", "London"],
            },
            "category": {"label": "IT Jobs", "tag": "it-jobs"},
            "contract_type": "permanent",
            "salary_min": 90000,
            "salary_max": 130000,
            "created": "2026-05-23T08:00:00Z",
            "redirect_url": "https://example.com/jobs/e2e-mock-001",
        }
    ]


def _install_mock_adzuna_http():
    """Replace adzuna.scraper.scrape_adzuna_batch with an in-memory stub.

    We patch at the symbol the agent's @tool actually calls (the import
    in adzuna.tools), so the mock takes effect regardless of how the
    function is invoked.
    """
    mock_response = _build_mock_adzuna_response()

    async def fake_scrape(queries, country="gb", delay_seconds=0.0, max_vacancies=None):
        # Mimic real signature; ignore everything except returning the
        # synthetic dict.  Print so the test output makes it obvious the
        # mock fired.
        print(f"  [mock] scrape_adzuna_batch called with {len(queries)} queries — returning {len(mock_response)} fake vacancies")
        return list(mock_response)

    import adzuna.scraper as scraper_mod
    import adzuna.tools as tools_mod
    scraper_mod.scrape_adzuna_batch = fake_scrape
    tools_mod.scrape_adzuna_batch = fake_scrape


# ── Log-sink replay ──────────────────────────────────────────────────────────

def _print_recent_refresh_events(since_ts: float) -> None:
    from services.log_sink import LOG_PATH
    if not LOG_PATH.exists():
        print("  (no log_sink file yet)")
        return
    print(f"  log_sink events from {LOG_PATH} (component=vacancy_refresh, ts > {since_ts:.0f}):")
    with LOG_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("component") != "vacancy_refresh":
                continue
            if (rec.get("ts") or 0) < since_ts:
                continue
            keys = {k: v for k, v in rec.items() if k not in {"level", "logger", "component"}}
            print(f"    {json.dumps(keys, ensure_ascii=False)}")


# ── Main ─────────────────────────────────────────────────────────────────────

async def _run(quiet: bool = False) -> int:
    print("== Local E2E mock refresh ==")
    print(f"  repo: {REPO_ROOT}")
    print(f"  VACANCIES_PATH: {os.environ.get('VACANCIES_PATH')}")
    print(f"  CHROMA_DIR: {os.environ.get('CHROMA_DIR')}")
    print(f"  refresh interval: {os.environ.get('VACANCY_REFRESH_INTERVAL_SECONDS', 'default')}s")

    # 1. Boot the live recommender — same singleton the FastAPI lifespan uses.
    t0 = time.time()
    print("\n[1/5] Building RecommenderService (HuggingFace embedder + Chroma + BM25)...")
    from services.recommender import get_recommender
    rec = get_recommender()
    print(f"      ready in {time.time() - t0:.1f}s  ({len(rec.pipeline.vacancies)} vacancies in seed corpus)")

    # 2. Confirm baseline search works on the seed corpus.
    print("\n[2/5] Baseline search (should NOT contain the marker yet):")
    baseline = await rec.search(
        profile={"job_description": {"desired_positions": ["machine learning engineer"]}},
        top_k=5,
    )
    for row in baseline[:3]:
        print(f"      - {row['id']:>12s}  {row['title']:<40s}  score={row['match_score']:.3f}")
    if any(MARKER_TITLE in (row.get("title") or "") for row in baseline):
        print("      !! marker already in baseline — corpus contaminated")
        return 1

    # 3. Install Adzuna HTTP mock + drive one refresh cycle.
    print("\n[3/5] Installing mock Adzuna HTTP and triggering one refresh cycle...")
    _install_mock_adzuna_http()
    from services.vacancy_refresh import VacancyRefreshService
    svc = VacancyRefreshService(rec)
    refresh_started = time.time()
    result = await svc.trigger_once()
    duration = time.time() - refresh_started
    print(f"      result: {json.dumps({k: v for k, v in result.items() if k != 'summary'}, ensure_ascii=False)}")
    if result.get("summary"):
        print(f"      agent summary: {result['summary'][:300]}")
    print(f"      cycle wall-clock: {duration:.1f}s")

    if result.get("status") != "ok":
        print(f"\n  FAIL: refresh did not succeed. error: {result.get('error')}")
        _print_recent_refresh_events(refresh_started)
        return 2
    if result.get("appended", 0) < 1:
        print(f"\n  FAIL: refresh succeeded but no vacancies were appended (appended={result.get('appended')})")
        _print_recent_refresh_events(refresh_started)
        return 3

    # 4. Verify the marker shows up in the live recommender search results.
    print("\n[4/5] Searching for the marker title via the live recommender...")
    after = await rec.search(
        profile={
            "job_description": {
                "desired_positions": ["Quantum Sandwich Engineer"],
                "desired_tech_stack": ["python", "pytorch"],
                "preferred_locations": ["London"],
            },
            "candidate_description": {"skills": ["python", "pytorch"]},
        },
        top_k=5,
    )
    for row in after[:5]:
        marker = " <-- MARKER" if MARKER_TITLE in (row.get("title") or "") else ""
        print(f"      - {row['id']:>14s}  {row['title']:<40s}  score={row['match_score']:.3f}{marker}")

    found = any(MARKER_TITLE in (row.get("title") or "") for row in after)
    if not found:
        # The reranker / fusion may have ranked the mock outside the top-5
        # for a generic query, so retry with a more specific query that
        # exercises BM25 on the unique tokens.
        print("\n      (marker not in top-5 for generic query, retrying with literal title)")
        after2 = await rec.search(
            profile={"job_description": {"desired_positions": [MARKER_TITLE]}},
            top_k=5,
        )
        for row in after2[:5]:
            marker = " <-- MARKER" if MARKER_TITLE in (row.get("title") or "") else ""
            print(f"      - {row['id']:>14s}  {row['title']:<40s}  score={row['match_score']:.3f}{marker}")
        found = any(MARKER_TITLE in (row.get("title") or "") for row in after2)

    # 5. Replay the structured events the loop wrote to log_sink.
    print("\n[5/5] log_sink events emitted during this cycle:")
    _print_recent_refresh_events(refresh_started)

    if found:
        print("\n  PASS: end-to-end mock refresh integration is wired correctly.")
        return 0
    print("\n  FAIL: marker vacancy did not surface in recommender.search results.")
    return 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    code = asyncio.run(_run(quiet=args.quiet))
    sys.exit(code)


if __name__ == "__main__":
    main()
