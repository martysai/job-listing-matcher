"""Unified append-only JSONL log sink with DuckDB read access.

All application log streams converge on a single JSONL file (LOG_PATH):
  - sara.agent operational events  → via JsonlHandler attached in main.py
  - sara.llm LLM call audits       → via propagation when path=None
  - chat route events               → via append() directly

The read path (routes/logs.py) queries the file through DuckDB in-process.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import IO, Any

LOG_PATH: Path = Path(os.environ.get("LOG_PATH", "data/logs/app.jsonl"))

_lock: threading.Lock = threading.Lock()
_file: IO[str] | None = None


def _get_file() -> IO[str]:
    global _file
    if _file is None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _file = open(LOG_PATH, "a", encoding="utf-8")  # noqa: WPS515
    return _file


def append(**fields: Any) -> None:
    """Thread-safe append of one JSON record to LOG_PATH. Never raises."""
    try:
        line = json.dumps(fields, ensure_ascii=False, default=str)
        with _lock:
            f = _get_file()
            f.write(line + "\n")
            f.flush()
    except Exception as exc:
        print(f"[log_sink] write failed: {exc}", file=sys.stderr)


def query(sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    """Run a DuckDB SQL query and return rows as dicts.

    The SQL may reference the log file via read_ndjson_auto(?) with LOG_PATH
    as the first positional parameter.  Returns [] if the file does not exist.
    """
    if not LOG_PATH.exists():
        return []
    try:
        import duckdb  # deferred — only the read path needs it

        con = duckdb.connect(":memory:")
        result = con.execute(sql, params or [])
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]
    except Exception as exc:
        print(f"[log_sink] query failed: {exc}", file=sys.stderr)
        return []


class JsonlHandler(logging.Handler):
    """Logging handler that writes structured JSONL records via append().

    If the formatted message is itself a JSON object it is merged into the
    top-level record so LLM call records (which wrap their payload as JSON)
    are stored with their fields at the top level rather than nested under
    ``message``.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            base: dict[str, Any] = {
                "ts": record.created,
                "level": record.levelname.lower(),
                "logger": record.name,
            }
            try:
                parsed = json.loads(msg)
                if isinstance(parsed, dict):
                    base.update(parsed)
                    append(**base)
                    return
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            base["message"] = msg
            append(**base)
        except Exception:
            self.handleError(record)
