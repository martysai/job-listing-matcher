import asyncio
import os
import queue
import threading
from typing import AsyncGenerator

from mistralai import Mistral

from sara_candidate_poll import parse_job_request

SYSTEM_PROMPT = """You are a friendly job recommendation assistant. Your goal is to understand
what kind of job the user is looking for and collect the following information naturally
through conversation:

1. Desired job title or role
2. Key skills (programming languages, tools, frameworks)
3. Preferred location or remote preference
4. Years of experience
5. Job type preference (full-time, part-time, contract)
6. Salary expectations (optional)
7. Any other important preferences

Be conversational and warm. Ask one or two questions at a time — don't overwhelm the user
with a form-like experience.

Once you have gathered at minimum: role, skills, and location/remote preference, end your
message with the exact marker <COLLECT> on its own line. Only emit it once.
"""

_DEFAULT_MODEL = "mistral-small-latest"
_SENTINEL = object()


def _stream_worker(
    client: Mistral,
    model: str,
    messages: list[dict],
    q: "queue.Queue[object]",
) -> None:
    """Run Mistral streaming in a background thread; push chunks onto q."""
    try:
        with client.chat.stream(model=model, messages=messages) as stream:
            for chunk in stream:
                delta = chunk.data.choices[0].delta.content
                if delta:
                    q.put(delta)
    finally:
        q.put(_SENTINEL)


class ConversationService:
    def __init__(self):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    async def stream(
        self,
        messages: list[dict],
        session_id: str,
    ) -> AsyncGenerator[dict, None]:
        text_buffer = ""
        in_collect = False
        search_emitted = False

        q: queue.Queue = queue.Queue()
        full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

        threading.Thread(
            target=_stream_worker,
            args=(self.client, _DEFAULT_MODEL, full_messages, q),
            daemon=True,
        ).start()

        loop = asyncio.get_event_loop()

        while True:
            chunk = await loop.run_in_executor(None, q.get)
            if chunk is _SENTINEL:
                break

            if in_collect:
                continue

            text_buffer += chunk

            if "<COLLECT>" in text_buffer:
                pre, _ = text_buffer.split("<COLLECT>", 1)
                if pre:
                    yield {"type": "text", "content": pre}
                in_collect = True
                text_buffer = ""
            elif len(text_buffer) > 20:
                safe, text_buffer = text_buffer[:-20], text_buffer[-20:]
                yield {"type": "text", "content": safe}

        if text_buffer.strip():
            yield {"type": "text", "content": text_buffer}

        if in_collect and not search_emitted:
            user_text = "\n".join(
                m["content"] for m in messages if m["role"] == "user"
            )
            parsed = await loop.run_in_executor(None, parse_job_request, user_text)
            yield {"type": "ready_to_search", "profile": parsed.model_dump()}

        yield {"type": "done"}
