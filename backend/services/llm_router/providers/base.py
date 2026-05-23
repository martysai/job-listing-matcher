"""Provider interface used by the router.

A provider knows how to:

* Stream tokens for a chat completion (used by the live conversation).
* Return a Pydantic-parsed structured response in one shot (used by the
  candidate-profile parser).

Streaming is exposed as a synchronous iterator of ``str`` chunks because the
underlying SDKs (mistralai, openai) are blocking; the router moves the work
to a thread for async callers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass
class StructuredResult:
    """Wraps a parsed Pydantic model so logging/metadata can be attached later."""

    parsed: BaseModel


class Provider(ABC):
    """Adapter for a single LLM provider."""

    name: str

    @abstractmethod
    def stream_chat(self, *, tier: str, messages: list[dict]) -> Iterable[str]:
        """Yield text chunks for the given chat messages."""

    @abstractmethod
    def parse_structured(
        self,
        *,
        tier: str,
        prompt: str,
        response_format: Type[T],
    ) -> T:
        """Return a Pydantic model parsed from a single-prompt structured call."""
