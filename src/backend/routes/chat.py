import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.conversation import ConversationService

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
    """
    Stream chat responses as SSE events.

    Event types:
      - data: {"type": "text", "content": "..."}         → streamed text token
      - data: {"type": "ready_to_search", "profile": {}} → bot collected enough info, trigger job search
      - data: {"type": "done"}                            → stream complete
    """

    async def event_generator():
        async for event in conversation_service.stream(
            messages=[m.model_dump() for m in request.messages],
            session_id=request.session_id,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )
