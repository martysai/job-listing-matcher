"""Mistral provider adapter (``mistralai`` SDK)."""

from __future__ import annotations

import os
from typing import Iterable, Type, TypeVar

from pydantic import BaseModel

from ..config import tier as _tier
from .base import Provider

T = TypeVar("T", bound=BaseModel)


class MistralProvider(Provider):
    """Wraps ``mistralai.client.Mistral`` with router-shaped methods."""

    name = "mistral"

    def __init__(self, api_key: str | None = None):
        from mistralai.client import Mistral  # lazy import

        key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not key:
            raise EnvironmentError(
                "MISTRAL_API_KEY environment variable is not set"
            )
        self._client = Mistral(api_key=key)

    def stream_chat(self, *, tier: str, messages: list[dict]) -> Iterable[str]:
        model = _tier(tier).mistral
        with self._client.chat.stream(model=model, messages=messages) as stream:
            for chunk in stream:
                delta = chunk.data.choices[0].delta.content
                if delta:
                    yield delta

    def parse_structured(
        self,
        *,
        tier: str,
        prompt: str,
        response_format: Type[T],
    ) -> T:
        model = _tier(tier).mistral
        response = self._client.chat.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format,
        )
        return response.choices[0].message.parsed
