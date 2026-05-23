"""Configuration for the LLM router.

Two tiers cover every call site in this project:

* ``small`` — cheap, fast model for chat streaming, structured extraction,
  rerank scoring.
* ``large`` — higher-capacity model for the ReAct agent and the candidate
  profile parser.

Each tier maps to a concrete model id per provider.  Provider order and
cooldown duration are env-driven so they can be flipped without code edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

PROVIDER_MISTRAL = "mistral"
PROVIDER_GITHUB = "github"

DEFAULT_PROVIDER_ORDER = (PROVIDER_MISTRAL, PROVIDER_GITHUB)


@dataclass(frozen=True)
class TierModels:
    """Model ids for a single tier across providers."""

    mistral: str
    github: str

    # LiteLLM-prefixed equivalents, used by the LiteLLM bridge so callers do
    # not have to think about the ``openai/openai/...`` double-prefix quirk.
    mistral_litellm: str
    github_litellm: str


TIERS: dict[str, TierModels] = {
    "small": TierModels(
        mistral="mistral-small-latest",
        github="openai/gpt-4o-mini",
        mistral_litellm="mistral/mistral-small-latest",
        github_litellm="openai/openai/gpt-4o-mini",
    ),
    "large": TierModels(
        mistral="mistral-large-latest",
        github="openai/gpt-4o",
        mistral_litellm="mistral/mistral-large-latest",
        github_litellm="openai/openai/gpt-4o",
    ),
}


GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"


def provider_order() -> tuple[str, ...]:
    """Return providers in attempt order; honours ``LLM_PROVIDER_ORDER`` env."""
    raw = os.environ.get("LLM_PROVIDER_ORDER")
    if not raw:
        return DEFAULT_PROVIDER_ORDER
    parts = tuple(p.strip().lower() for p in raw.split(",") if p.strip())
    valid = tuple(p for p in parts if p in (PROVIDER_MISTRAL, PROVIDER_GITHUB))
    return valid or DEFAULT_PROVIDER_ORDER


def cooldown_seconds() -> float:
    """How long to skip a provider after it returned a transient error."""
    return float(os.environ.get("LLM_PROVIDER_COOLDOWN_S", "60"))


def tier(name: str) -> TierModels:
    """Look up the tier mapping; raises ``KeyError`` for unknown tiers."""
    try:
        return TIERS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown LLM tier {name!r}; expected one of {sorted(TIERS)}"
        ) from exc
