"""GitHub Models provider adapter (OpenAI SDK, custom base URL)."""

from __future__ import annotations

from typing import Iterable, Type, TypeVar

from pydantic import BaseModel

from ..auth import load_github_pat
from ..config import GITHUB_MODELS_BASE_URL, tier as _tier
from .base import Provider

T = TypeVar("T", bound=BaseModel)


class GitHubModelsProvider(Provider):
    """OpenAI-SDK-compatible client pointed at ``models.github.ai``."""

    name = "github"

    def __init__(self, api_key: str | None = None):
        from openai import OpenAI  # lazy import

        token = api_key or load_github_pat()
        self._client = OpenAI(
            base_url=GITHUB_MODELS_BASE_URL,
            api_key=token,
        )

    def stream_chat(self, *, tier: str, messages: list[dict]) -> Iterable[str]:
        model = _tier(tier).github
        stream = self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content

    def parse_structured(
        self,
        *,
        tier: str,
        prompt: str,
        response_format: Type[T],
    ) -> T:
        model = _tier(tier).github
        response = self._client.chat.completions.parse(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_format,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError(
                "GitHub Models returned no parsed payload for structured call"
            )
        return parsed
