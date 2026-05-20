import json

from fastapi import APIRouter, Query
from pydantic import BaseModel

from services.database import get_db

router = APIRouter()


class LogEntry(BaseModel):
    id: int
    session_id: str | None
    event: str
    payload: dict | None
    created_at: str


@router.get("/logs", response_model=list[LogEntry])
async def get_logs(
    session_id: str | None = Query(None),
    event: str | None = Query(None),
    limit: int = Query(50, le=500),
    offset: int = Query(0),
):
    conditions: list[str] = []
    params: list = []

    if session_id:
        conditions.append("session_id = ?")
        params.append(session_id)
    if event:
        conditions.append("event = ?")
        params.append(event)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params += [limit, offset]

    async with get_db() as db:
        cursor = await db.execute(
            f"SELECT id, session_id, event, payload, created_at "
            f"FROM logs {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            params,
        )
        rows = await cursor.fetchall()

    return [
        {
            "id": r["id"],
            "session_id": r["session_id"],
            "event": r["event"],
            "payload": json.loads(r["payload"]) if r["payload"] else None,
            "created_at": r["created_at"],
        }
        for r in rows
    ]
