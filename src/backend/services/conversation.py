import json
import os
from typing import AsyncGenerator
import anthropic

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
with a form-like experience. Once you have gathered enough information (at minimum: role,
skills, and location/remote preference), respond with a special signal.

When you have enough information to search for jobs, end your message with this exact JSON
block on its own line (the frontend will detect and strip it):

<SEARCH_READY>
{
  "title": "...",
  "skills": ["...", "..."],
  "location": "...",
  "experience_years": null,
  "job_type": "full-time",
  "salary_min": null,
  "notes": "..."
}
</SEARCH_READY>

Only emit <SEARCH_READY> once. After that, continue chatting normally if the user has
more to say, but don't repeat the signal.
"""


class ConversationService:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    async def stream(
        self,
        messages: list[dict],
        session_id: str,
    ) -> AsyncGenerator[dict, None]:
        """
        Stream events to the client.
        Detects the <SEARCH_READY> block and emits a structured event.
        """
        buffer = ""
        search_emitted = False

        async with self.client.messages.stream(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                buffer += text

                # Check if we're accumulating a <SEARCH_READY> block
                if "<SEARCH_READY>" in buffer:
                    # Split into before and inside/after the block
                    pre, rest = buffer.split("<SEARCH_READY>", 1)

                    # Yield text before the marker
                    if pre.strip():
                        yield {"type": "text", "content": pre}

                    if "</SEARCH_READY>" in rest and not search_emitted:
                        json_str, after = rest.split("</SEARCH_READY>", 1)
                        try:
                            profile = json.loads(json_str.strip())
                            yield {"type": "ready_to_search", "profile": profile}
                            search_emitted = True
                        except json.JSONDecodeError:
                            pass  # Malformed — just skip

                        buffer = after  # Continue with any text after the block
                    # else: still accumulating the closing tag — wait
                else:
                    # No search block in flight — yield text if we have a safe chunk
                    # Hold back last 20 chars to avoid splitting the opening tag
                    if len(buffer) > 20:
                        safe, buffer = buffer[:-20], buffer[-20:]
                        yield {"type": "text", "content": safe}

        # Flush remaining buffer
        if buffer.strip() and "<SEARCH_READY>" not in buffer:
            yield {"type": "text", "content": buffer}

        yield {"type": "done"}
