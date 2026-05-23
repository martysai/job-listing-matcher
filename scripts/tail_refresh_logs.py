"""Live tail of the vacancy_refresh events emitted to data/logs/app.jsonl.

Pretty-prints each new event as a single line so you can watch the
scheduled refresh loop fire in real time alongside the running backend.

Usage:
    python scripts/tail_refresh_logs.py
    python scripts/tail_refresh_logs.py --all       # don't filter by component
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = REPO_ROOT / "data" / "logs" / "app.jsonl"


def _fmt(rec: dict) -> str:
    ts = rec.get("ts")
    when = datetime.fromtimestamp(ts).strftime("%H:%M:%S") if isinstance(ts, (int, float)) else "?"
    event = rec.get("event") or rec.get("message") or "?"
    extras = []
    for key in ("processed", "appended", "n_queries", "duration_seconds", "reason", "error"):
        if key in rec and rec[key] not in (None, ""):
            val = rec[key]
            if isinstance(val, float):
                val = f"{val:.2f}"
            if isinstance(val, str) and len(val) > 120:
                val = val[:117] + "..."
            extras.append(f"{key}={val}")
    suffix = "  " + " ".join(extras) if extras else ""
    return f"{when}  {event:<14s}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=str(DEFAULT_LOG), help="path to app.jsonl")
    parser.add_argument("--all", action="store_true", help="don't filter by component=vacancy_refresh")
    parser.add_argument("--from-start", action="store_true", help="print existing events before tailing")
    args = parser.parse_args()

    log_path = Path(args.path)
    print(f"tail: {log_path}  (filter: {'*' if args.all else 'component=vacancy_refresh'})", flush=True)

    while not log_path.exists():
        print("(waiting for log file to appear...)", flush=True)
        time.sleep(2)

    with log_path.open(encoding="utf-8") as fh:
        if not args.from_start:
            fh.seek(0, os.SEEK_END)
        while True:
            line = fh.readline()
            if not line:
                time.sleep(0.5)
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not args.all and rec.get("component") != "vacancy_refresh":
                continue
            print(_fmt(rec), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
