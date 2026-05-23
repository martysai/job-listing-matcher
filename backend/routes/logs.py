from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

from services.log_sink import LOG_PATH, query

router = APIRouter()


class LogEntry(BaseModel):
    ts: float | None = None
    level: str | None = None
    logger: str | None = None
    component: str | None = None
    event: str | None = None
    session_id: str | None = None
    message: str | None = None
    payload: Any = None
    latency_ms: float | None = None
    error: str | None = None


@router.get("/logs", response_model=list[LogEntry])
async def get_logs(
    session_id: str | None = Query(None),
    event: str | None = Query(None),
    component: str | None = Query(None),
    limit: int = Query(50, le=500),
    offset: int = Query(0),
):
    if not LOG_PATH.exists():
        return []

    conditions: list[str] = []
    params: list = [str(LOG_PATH)]

    if session_id:
        conditions.append("session_id = ?")
        params.append(session_id)
    if event:
        conditions.append("event = ?")
        params.append(event)
    if component:
        conditions.append("component = ?")
        params.append(component)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params += [limit, offset]

    sql = (
        f"SELECT ts, level, logger, component, event, session_id, "
        f"message, payload, latency_ms, error "
        f"FROM read_ndjson_auto(?, union_by_name=true, ignore_errors=true) "
        f"{where} "
        f"ORDER BY ts DESC NULLS LAST "
        f"LIMIT ? OFFSET ?"
    )

    return query(sql, params)
