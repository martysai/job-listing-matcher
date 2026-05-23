import json
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.conversation import ConversationService
from services.database import get_db

router = APIRouter()
conversation_service = ConversationService()


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    session_id: str


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    new_user_msg = request.messages[-1]

    async def event_generator():
        assistant_chunks: list[str] = []
        now = datetime.now(timezone.utc).isoformat()

        async with get_db() as db:
            await db.execute(
                "INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?, ?)",
                (request.session_id, now),
            )
            await db.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (request.session_id, new_user_msg.role, new_user_msg.content, now),
            )
            await db.commit()

            async for event in conversation_service.stream(
                messages=[m.model_dump() for m in request.messages],
                session_id=request.session_id,
            ):
                if event["type"] == "text":
                    assistant_chunks.append(event["content"])

                yield f"data: {json.dumps(event)}\n\n"

            if assistant_chunks:
                await db.execute(
                    "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (
                        request.session_id,
                        "assistant",
                        "".join(assistant_chunks),
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                await db.commit()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
