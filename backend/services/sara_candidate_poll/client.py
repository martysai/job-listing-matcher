"""Structured-parse client backed by the LLM router (Mistral → GitHub Models)."""

from typing import Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from llm_router import parse_structured as _router_parse_structured

load_dotenv()

T = TypeVar("T", bound=BaseModel)

# Kept for backward compatibility with parser.py which imports this symbol.
_DEFAULT_MODEL = "mistral-large-latest"
_DEFAULT_TIER = "large"


def parse_structured(
    full_prompt: str,
    response_format: Type[T],
    model: str = _DEFAULT_MODEL,
) -> T:
    """Send ``full_prompt`` to the LLM router and parse the response.

    The ``model`` argument is kept for backward compatibility but is no longer
    consulted directly — provider+model selection is owned by the router via
    the tier mapping in ``llm_router.config``.  Callers that need a different
    capability tier should call ``llm_router.parse_structured`` directly.
    """
    return _router_parse_structured(
        full_prompt,
        response_format,
        tier=_DEFAULT_TIER,
    )
