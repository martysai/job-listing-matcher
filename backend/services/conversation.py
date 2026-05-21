import asyncio
import os
import queue
import threading
from typing import AsyncGenerator

from mistralai import Mistral

from sara_candidate_poll import parse_job_request

SYSTEM_PROMPT = """You are a friendly job recommendation assistant. Your goal is to understand
what kind of job the user is looking for by collecting the information below naturally
through conversation.

Be conversational and warm. Ask one or two questions at a time — don't overwhelm the user
with a form-like experience. Never present these as a numbered list or a form.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INFORMATION TO COLLECT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

About the candidate (facts):
  - Total years of professional work experience (only if they state it explicitly)
  - Natural languages they speak (e.g. English, Russian, German)
  - Highest completed education level (e.g. bachelor's, master's, PhD)
  - Current skills — technologies, tools, frameworks, and soft skills they already have

What they are looking for (preferences):
  - Desired job title(s) or role(s)
  - Technologies/tools they want to USE at the new job (may overlap with current skills)
  - Subject domains or disciplines they want to work IN (e.g. distributed systems, NLP,
    computer vision) — distinct from specific tools
  - Activities or responsibilities they want to perform (e.g. design architecture,
    mentor junior engineers, lead a team)
  - Preferred industry or type of organisation (e.g. fintech, startup, product company)
  - Preferred work locations (cities, countries, or regions)
  - Locations or regions they want to EXCLUDE (e.g. "not the US", "no relocation to Asia")
  - Salary expectations: amount, currency, period (monthly/annual), and whether gross or net
  - Desired benefits beyond salary (e.g. health insurance, stock options, relocation support)
  - Employment type: full-time, part-time, contract, freelance, etc.
  - Preferred remote policy (remote / hybrid / onsite) and any formats they want to exclude

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPORTANT DISTINCTIONS TO PROBE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Current skills vs. desired stack: "I know Python" is a skill; "I want to keep using
  Python" or "I'd love to pick up Rust" is a preference for the new role. If unclear, ask.
- Preferred vs. excluded locations/remote: ask not only where they want to work, but
  whether there are places or formats they actively want to avoid.
- Salary: if they give a number, gently confirm whether it is monthly or annual, and
  gross or net, if not already stated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLETION SIGNAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Once you have gathered at minimum: desired role, current skills or desired tech stack,
and either a preferred location or a remote-work preference — end your message with the
exact marker <COLLECT> on its own line. Only emit it once.
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
