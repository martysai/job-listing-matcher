import asyncio
import time
from typing import AsyncGenerator

from llm_router import stream_chat as _router_stream_chat

from sara_candidate_poll import parse_job_request
from services import log_sink
from services.recommender import RecommenderService

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
EARLY SEARCH REQUEST
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
If the user explicitly signals they want to see job results immediately — for example: 
"just show me jobs", "search now", "find me something", "let's go", or any clear expression of impatience 
with the conversation — do the following:

1. Acknowledge briefly and warmly (one sentence).
2. Do NOT ask any further questions.
3. Treat all fields not yet collected as empty / unknown.
4. End your message immediately with <SEARCH> on its own line.

Do not apologise for missing information or list what is unknown. Simply proceed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPLETION SIGNAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Once you have gathered at minimum: desired role, current skills or desired tech stack,
and either a preferred location or a remote-work preference — end your message with the
exact marker <SEARCH> on its own line.
"""

_DEFAULT_TIER = "small"
_recommender = RecommenderService()


class ConversationService:
    def __init__(self):
        # No client is held: the router constructs providers lazily so a
        # transient outage on one side cannot break instantiation.
        pass

    async def stream(
        self,
        messages: list[dict],
        session_id: str,
    ) -> AsyncGenerator[dict, None]:
        text_buffer = ""
        in_search = False

        async for chunk in _router_stream_chat(
            messages,
            tier=_DEFAULT_TIER,
            system=SYSTEM_PROMPT,
        ):
            if in_search:
                continue

            text_buffer += chunk

            if "<SEARCH>" in text_buffer:
                pre, _ = text_buffer.split("<SEARCH>", 1)
                if pre:
                    yield {"type": "text", "content": pre}
                in_search = True
                text_buffer = ""
            elif len(text_buffer) > 20:
                safe, text_buffer = text_buffer[:-20], text_buffer[-20:]
                yield {"type": "text", "content": safe}

        if text_buffer.strip():
            yield {"type": "text", "content": text_buffer}

        if in_search:
            yield {"type": "searching"}
            user_text = "\n".join(
                m["content"] for m in messages if m["role"] == "user"
            )
            log_sink.append(
                ts=time.time(),
                level="info",
                logger="conversation",
                component="chat",
                event="parse_start",
                session_id=session_id,
            )
            loop = asyncio.get_running_loop()
            parsed = await loop.run_in_executor(None, parse_job_request, user_text)
            log_sink.append(
                ts=time.time(),
                level="info",
                logger="conversation",
                component="chat",
                event="parse_done",
                session_id=session_id,
                payload=parsed.model_dump(),
            )
            jobs = await _recommender.search(
                profile=parsed.model_dump(exclude_none=True), top_k=10
            )
            yield {"type": "jobs", "jobs": jobs}

        yield {"type": "done"}
