"""Hierarchical, DuckDB-backed read access to the unified JSONL log.

Three drill-down levels:
  GET /logs/workflows           coarse buckets (chat, scraper, extraction, system)
  GET /logs/sessions?workflow=  sessions within a bucket (chat sessions, scraper runs…)
  GET /logs/events?workflow=…   individual records of one session, with full raw JSON

Each line is read as a single raw JSON object (read_json_objects), so records with
missing keys yield NULL instead of a binder error on sparse logs. Every endpoint is
bounded by a LIMIT; sessionization is additionally capped to a recent-rows window.
"""

import json
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

from services.log_sink import LOG_PATH, query

router = APIRouter()

# Read each JSONL line as one raw JSON object; absent keys extract to NULL.
_SRC = "read_json_objects(?, format='newline_delimited', ignore_errors=true)"


def _s(field: str) -> str:
    """SQL fragment extracting a top-level string field from the `json` column."""
    return f"json_extract_string(json, '$.{field}')"


def _num(field: str) -> str:
    """SQL fragment extracting a top-level numeric field as DOUBLE (NULL if absent)."""
    return f"TRY_CAST(json_extract_string(json, '$.{field}') AS DOUBLE)"


_TS = _num("ts")

# Maps each record to a coarse workflow bucket for the top-level view.
_WORKFLOW = (
    "CASE "
    f"WHEN {_s('component')} = 'chat' OR {_s('logger')} = 'conversation' THEN 'chat' "
    f"WHEN {_s('logger')} = 'sara.agent' THEN 'scraper' "
    f"WHEN {_s('logger')} = 'sara.llm' THEN 'extraction' "
    "ELSE 'system' END"
)
_IS_ERR = (
    f"CASE WHEN {_s('level')} = 'error' OR {_s('error')} IS NOT NULL "
    "THEN 1 ELSE 0 END"
)

_WORKFLOWS = {"chat", "scraper", "extraction", "system"}

# Upper bound on rows pulled into the sessionization window per request.
_SESSION_SCAN_CAP = 5000


class Workflow(BaseModel):
    workflow: str
    count: int
    first_ts: float | None = None
    last_ts: float | None = None
    error_count: int = 0


class Session(BaseModel):
    session_id: str | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    count: int = 0
    error_count: int = 0
    summary: str | None = None


class Event(BaseModel):
    ts: float | None = None
    level: str | None = None
    logger: str | None = None
    component: str | None = None
    event: str | None = None
    session_id: str | None = None
    summary: str | None = None
    raw: dict[str, Any] | None = None


@router.get("/logs/workflows", response_model=list[Workflow])
async def list_workflows():
    """Top level: one row per workflow bucket with counts and time span."""
    sql = (
        f"SELECT {_WORKFLOW} AS workflow, count(*) AS count, "
        f"min({_TS}) AS first_ts, max({_TS}) AS last_ts, "
        f"sum({_IS_ERR}) AS error_count "
        f"FROM {_SRC} WHERE {_TS} IS NOT NULL "
        f"GROUP BY 1 ORDER BY last_ts DESC NULLS LAST"
    )
    return query(sql, [str(LOG_PATH)])


@router.get("/logs/sessions", response_model=list[Session])
async def list_sessions(
    workflow: str = Query(...),
    limit: int = Query(50, ge=1, le=200),
    gap_seconds: float = Query(900, ge=1),
):
    """Middle level: group a bucket's records into sessions.

    A new session starts when ``session_id`` changes (chat sessions) or when the
    gap to the previous record exceeds ``gap_seconds`` (scraper / extraction runs,
    which carry no explicit id).
    """
    if workflow not in _WORKFLOWS:
        return []
    sql = f"""
    WITH base AS (
      SELECT {_TS} AS ts, {_s('session_id')} AS session_id,
             {_s('event')} AS event, {_s('message')} AS message,
             {_s('component')} AS component, {_IS_ERR} AS is_err,
             {_WORKFLOW} AS workflow
      FROM {_SRC} WHERE {_TS} IS NOT NULL
    ),
    filt AS (
      SELECT * FROM base WHERE workflow = ? ORDER BY ts DESC LIMIT ?
    ),
    marked AS (
      SELECT *, CASE
                  WHEN session_id IS DISTINCT FROM lag(session_id) OVER w
                       OR lag(ts) OVER w IS NULL
                       OR ts - lag(ts) OVER w > ?
                  THEN 1 ELSE 0 END AS is_new
      FROM filt WINDOW w AS (ORDER BY ts)
    ),
    numbered AS (
      SELECT *, sum(is_new) OVER (ORDER BY ts) AS sess FROM marked
    )
    SELECT any_value(session_id) AS session_id,
           min(ts) AS start_ts, max(ts) AS end_ts,
           count(*) AS count, sum(is_err) AS error_count,
           arg_min(coalesce(event, message, component), ts) AS summary
    FROM numbered GROUP BY sess ORDER BY start_ts DESC LIMIT ?
    """
    return query(sql, [str(LOG_PATH), workflow, _SESSION_SCAN_CAP, gap_seconds, limit])


@router.get("/logs/events", response_model=list[Event])
async def list_events(
    workflow: str = Query(...),
    session_id: str | None = Query(None),
    start: float | None = Query(None),
    end: float | None = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Leaf level: individual records of one session, oldest first.

    Identify a session by ``session_id`` (chat) and/or a ``start``/``end`` ts range
    (synthetic scraper / extraction runs). ``raw`` is the full original record.
    """
    if workflow not in _WORKFLOWS:
        return []
    conditions = [f"{_WORKFLOW} = ?"]
    params: list[Any] = [str(LOG_PATH), workflow]
    if session_id is not None:
        conditions.append(f"{_s('session_id')} = ?")
        params.append(session_id)
    if start is not None:
        conditions.append(f"{_TS} >= ?")
        params.append(start)
    if end is not None:
        conditions.append(f"{_TS} <= ?")
        params.append(end)
    where = " AND ".join(conditions)
    params += [limit, offset]
    sql = f"""
    SELECT {_TS} AS ts, {_s('level')} AS level, {_s('logger')} AS logger,
           {_s('component')} AS component, {_s('event')} AS event,
           {_s('session_id')} AS session_id,
           coalesce({_s('event')}, {_s('message')},
                    left({_s('prompt')}, 200), {_s('component')}) AS summary,
           CAST(json AS VARCHAR) AS raw
    FROM {_SRC}
    WHERE {where} AND {_TS} IS NOT NULL
    ORDER BY ts ASC LIMIT ? OFFSET ?
    """
    rows = query(sql, params)
    for row in rows:
        raw = row.get("raw")
        if isinstance(raw, str):
            try:
                row["raw"] = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                row["raw"] = {"_unparsed": raw}
    return rows
