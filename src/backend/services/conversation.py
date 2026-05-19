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
        # Two separate buffers prevent the pre-text from being re-emitted on
        # every chunk while waiting for </SEARCH_READY> to arrive.
        text_buffer = ""
        search_buffer = ""
        in_search_block = False
        search_emitted = False

        async with self.client.messages.stream(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                if not in_search_block:
                    text_buffer += text

                    if "<SEARCH_READY>" in text_buffer:
                        pre, rest = text_buffer.split("<SEARCH_READY>", 1)
                        if pre:
                            yield {"type": "text", "content": pre}
                        in_search_block = True
                        search_buffer = rest
                        text_buffer = ""
                    else:
                        # Hold back 20 chars to avoid splitting the opening tag
                        if len(text_buffer) > 20:
                            safe, text_buffer = text_buffer[:-20], text_buffer[-20:]
                            yield {"type": "text", "content": safe}
                else:
                    search_buffer += text

                    if "</SEARCH_READY>" in search_buffer and not search_emitted:
                        json_str, after = search_buffer.split("</SEARCH_READY>", 1)
                        try:
                            profile = json.loads(json_str.strip())
                            yield {"type": "ready_to_search", "profile": profile}
                            search_emitted = True
                        except json.JSONDecodeError:
                            pass
                        in_search_block = False
                        text_buffer = after
                        search_buffer = ""

        if text_buffer.strip():
            yield {"type": "text", "content": text_buffer}

        yield {"type": "done"}
