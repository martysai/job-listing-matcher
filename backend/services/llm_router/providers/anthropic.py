"""Anthropic provider adapter (``anthropic`` SDK).

Notes on the SDK surface we touch:

* ``client.messages.stream(...)`` is a context manager whose ``.text_stream``
  yields text deltas.  We mirror the Mistral/OpenAI provider shape by
  yielding bare ``str`` chunks.
* ``client.messages.create(...)`` with a forced ``tool_choice`` is the
  canonical way to get a structured payload out of Claude — the model fills
  the tool's JSON schema, and we hand the args dict to Pydantic for parsing.

Anthropic's API also requires the system prompt to be a top-level parameter
(``system=``) rather than a message with ``role == "system"``.  We split it
out transparently here so callers can keep using the OpenAI-style message
list.
"""

from __future__ import annotations

from typing import Iterable, Type, TypeVar

from pydantic import BaseModel

from ..auth import load_anthropic_key
from ..config import tier as _tier
from .base import Provider

T = TypeVar("T", bound=BaseModel)


_MAX_TOKENS_DEFAULT = 4096
_STRUCTURED_TOOL_NAME = "emit_structured_response"


def _split_system(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Pull any ``system`` role messages out into a single system string."""
    system_chunks: list[str] = []
    rest: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str) and content:
                system_chunks.append(content)
        else:
            rest.append(msg)
    system = "\n\n".join(system_chunks) if system_chunks else None
    return system, rest


class AnthropicProvider(Provider):
    """Wraps ``anthropic.Anthropic`` with router-shaped methods."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        from anthropic import Anthropic  # lazy import

        token = api_key or load_anthropic_key()
        self._client = Anthropic(api_key=token)

    def stream_chat(self, *, tier: str, messages: list[dict]) -> Iterable[str]:
        model = _tier(tier).anthropic
        system, rest = _split_system(messages)
        kwargs: dict = {
            "model": model,
            "messages": rest,
            "max_tokens": _MAX_TOKENS_DEFAULT,
        }
        if system is not None:
            kwargs["system"] = system
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                if text:
                    yield text

    def parse_structured(
        self,
        *,
        tier: str,
        prompt: str,
        response_format: Type[T],
    ) -> T:
        model = _tier(tier).anthropic
        schema = response_format.model_json_schema()
        tool = {
            "name": _STRUCTURED_TOOL_NAME,
            "description": (
                f"Return a {response_format.__name__} value populated from the "
                "user's request."
            ),
            "input_schema": schema,
        }
        response = self._client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS_DEFAULT,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool],
            tool_choice={"type": "tool", "name": _STRUCTURED_TOOL_NAME},
        )

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and getattr(
                block, "name", None
            ) == _STRUCTURED_TOOL_NAME:
                return response_format.model_validate(block.input)

        raise ValueError(
            "Anthropic returned no tool_use block for structured call "
            f"(model={model!r})"
        )
