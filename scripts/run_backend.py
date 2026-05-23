"""Launch the FastAPI backend with REAL Adzuna HTTP (no mocks).

Wrapper that mirrors run_backend_with_mock_adzuna.py's sys.path + cwd setup
but skips the HTTP monkey-patch, so the scheduled refresh loop calls the
real Adzuna API (requires ADZUNA_APP_ID + ADZUNA_APP_KEY in .env).

Why this wrapper rather than ``uvicorn main:app`` directly?
    The recommender imports ``sara_retrieve_rerank`` via the namespace
    package layout under ``backend/services``. Without an explicit
    sys.path insert, Python may resolve that import against another
    checkout (e.g. the parallel agent's worktree). The recommender
    also reads relative paths like VACANCIES_PATH against cwd, so we
    chdir to the worktree root.

Usage:
    python scripts/run_backend.py --port 8001
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    import uvicorn
    from main import app

    print(
        f"[backend] starting at http://{args.host}:{args.port}  "
        f"refresh_interval={os.environ.get('VACANCY_REFRESH_INTERVAL_SECONDS', 'default')}s  "
        f"adzuna_app_id={os.environ.get('ADZUNA_APP_ID', '')[:8] or '(missing)'}",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
