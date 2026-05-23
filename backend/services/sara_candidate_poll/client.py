"""Structured-parse client backed by the LLM router (GitHub Models → Anthropic → Mistral)."""

from typing import Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

from llm_router import parse_structured as _router_parse_structured
from llm_router.config import tier_for_model_string

load_dotenv()

T = TypeVar("T", bound=BaseModel)

# Kept for backward compatibility with parser.py which imports this symbol
# (and uses it as the default for its own ``model=`` argument).
_DEFAULT_MODEL = "mistral-large-latest"
_DEFAULT_TIER = "large"


def parse_structured(
    full_prompt: str,
    response_format: Type[T],
    model: str = _DEFAULT_MODEL,
) -> T:
    """Send ``full_prompt`` to the LLM router and parse the response.

    The legacy ``model`` argument is mapped to a router tier via
    ``llm_router.config.tier_for_model_string`` (``"large"`` for anything
    containing ``large``/``opus``/``sonnet``, ``"small"`` otherwise) so
    callers that still pass a CLI ``--model`` keep their capability tier
    intact even though concrete provider+model selection is owned by the
    router.  Callers wanting an explicit tier should call
    ``llm_router.parse_structured`` directly.
    """
    return _router_parse_structured(
        full_prompt,
        response_format,
        tier=tier_for_model_string(model) if model else _DEFAULT_TIER,
    )
