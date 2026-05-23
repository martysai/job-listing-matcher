"""Standalone Adzuna refresh daemon — survives backend restarts.

Runs the same ``VacancyRefreshService`` as the in-backend lifespan path,
but as an independent OS process so the API server can restart, crash, or
be redeployed without interrupting the vacancy refresh cadence.

Architecture
------------
- The daemon and the API server share the **same on-disk Chroma store**
  (default ``data/chroma``) and the same Adzuna JSONL audit log.  Chroma
  supports multiple-reader / single-writer access; the daemon is the
  single writer, the API server reads.
- To avoid double-writes, the API server should run with
  ``VACANCY_REFRESH_ENABLED=false`` (so its own internal loop stays
  disabled).  The daemon owns the cron path.
- New vacancies become visible to the API server the next time its
  pipeline reloads from disk (typically on restart). Mid-run hot updates
  to the live BM25 index require the in-process refresh loop and are
  out of scope for the daemon path.

Usage
-----
    python scripts/run_refresh_daemon.py

Honours every VACANCY_REFRESH_* env knob from ``.env`` plus
``ADZUNA_MAX_VACANCIES`` and ``VACANCY_EXTRACTOR_DELAY`` for throughput
tuning.  Stop with Ctrl-C.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "backend" / "services"))
os.chdir(REPO_ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

# Force-enable the loop regardless of what the API server's .env says;
# the daemon's whole purpose is to run the loop.
os.environ["VACANCY_REFRESH_ENABLED"] = "true"

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("adzuna.daemon")


async def main() -> int:
    from services.recommender import get_recommender
    from services.vacancy_refresh import VacancyRefreshService

    started = time.time()
    log.info("daemon: booting recommender (this loads Chroma + BM25)…")
    recommender = get_recommender()
    log.info("daemon: recommender ready in %.1fs", time.time() - started)

    service = VacancyRefreshService(recommender=recommender)
    await service.start()
    log.info(
        "daemon: refresh loop started — interval=%ss initial_delay=%ss",
        os.environ.get("VACANCY_REFRESH_INTERVAL_SECONDS", "21600"),
        os.environ.get("VACANCY_REFRESH_INITIAL_DELAY_SECONDS", "30"),
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _request_stop(signum: int) -> None:
        log.info("daemon: signal %d received, shutting down…", signum)
        stop_event.set()

    for sig_name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _request_stop, sig)
        except (NotImplementedError, RuntimeError):
            signal.signal(sig, lambda s, _f: _request_stop(s))

    try:
        await stop_event.wait()
    finally:
        log.info("daemon: stopping refresh loop…")
        await service.stop()
        log.info("daemon: stopped.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
