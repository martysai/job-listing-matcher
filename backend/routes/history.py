from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from services.database import get_db

router = APIRouter()


class MessageOut(BaseModel):
    role: str
    content: str


@router.get("/chat/history/{session_id}", response_model=list[MessageOut])
async def get_history(session_id: str):
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        rows = await cursor.fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


@router.post("/chat/reset/{session_id}", status_code=204)
async def reset_session(session_id: str):
    now = datetime.now(timezone.utc).isoformat()
    async with get_db() as db:
        await db.execute(
            "UPDATE sessions SET reset_at = ? WHERE id = ?",
            (now, session_id),
        )
        await db.commit()
