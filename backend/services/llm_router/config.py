"""Configuration for the LLM router.

Two tiers cover every call site in this project:

* ``small`` — cheap, fast model for chat streaming, structured extraction,
  rerank scoring.
* ``large`` — higher-capacity model for the ReAct agent and the candidate
  profile parser.

Each tier maps to a concrete model id per provider.  Provider order and
cooldown duration are env-driven so they can be flipped without code edits.

Default attempt order is reliability-first across three educational/low-quota
subscriptions: Anthropic (highest quality, primary) → GitHub Models (OpenAI
SDK over models.github.ai, fallback) → Mistral (third).  The Anthropic model
id defaults to the latest stable ``claude-sonnet-4-6`` for both tiers and
can be overridden per tier via env vars (see ``_anthropic_model``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GITHUB = "github"
PROVIDER_MISTRAL = "mistral"

DEFAULT_PROVIDER_ORDER = (PROVIDER_ANTHROPIC, PROVIDER_GITHUB, PROVIDER_MISTRAL)

ALL_PROVIDERS = (PROVIDER_ANTHROPIC, PROVIDER_GITHUB, PROVIDER_MISTRAL)


_ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-6"


def _anthropic_model(tier_name: str) -> str:
    """Resolve the Anthropic model id for a tier.

    Both tiers default to the latest stable Sonnet (``claude-sonnet-4-6``).
    Override per tier with ``ANTHROPIC_SMALL_MODEL`` / ``ANTHROPIC_LARGE_MODEL``
    so users can pin a date-stamped version (e.g. ``claude-sonnet-4-5-20250929``)
    without touching code.
    """
    env_var = f"ANTHROPIC_{tier_name.upper()}_MODEL"
    return os.environ.get(env_var) or _ANTHROPIC_DEFAULT_MODEL


@dataclass(frozen=True)
class TierModels:
    """Model ids for a single tier across providers."""

    anthropic: str
    mistral: str
    github: str

    # LiteLLM-prefixed equivalents, used by the LiteLLM bridge so callers do
    # not have to think about the ``openai/openai/...`` double-prefix quirk.
    anthropic_litellm: str
    mistral_litellm: str
    github_litellm: str


def _tier_models(tier_name: str, *, mistral: str, github: str) -> TierModels:
    anthropic_model = _anthropic_model(tier_name)
    return TierModels(
        anthropic=anthropic_model,
        mistral=mistral,
        github=github,
        anthropic_litellm=f"anthropic/{anthropic_model}",
        mistral_litellm=f"mistral/{mistral}",
        github_litellm=f"openai/{github}",
    )


# The TIERS dict is rebuilt on every import; if you need fresh env overrides
# inside a long-running test, call ``refresh_tiers()``.
TIERS: dict[str, TierModels] = {
    "small": _tier_models("small", mistral="mistral-small-latest", github="openai/gpt-4o-mini"),
    "large": _tier_models("large", mistral="mistral-large-latest", github="openai/gpt-4o"),
}


def refresh_tiers() -> None:
    """Rebuild ``TIERS`` from the current environment.

    Tests that monkey-patch ``ANTHROPIC_SMALL_MODEL`` / ``ANTHROPIC_LARGE_MODEL``
    after import can call this to pick up the change.  Production code never
    needs it because env vars are read at process start.
    """
    TIERS["small"] = _tier_models(
        "small", mistral="mistral-small-latest", github="openai/gpt-4o-mini"
    )
    TIERS["large"] = _tier_models(
        "large", mistral="mistral-large-latest", github="openai/gpt-4o"
    )


GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"


def provider_order() -> tuple[str, ...]:
    """Return providers in attempt order; honours ``LLM_PROVIDER_ORDER`` env.

    Unknown names in the env value are dropped *with a warning* so a typo
    (e.g. ``mistal,github``) is visible in logs instead of silently falling
    back to defaults.  If every entry is unknown, the default order is used.
    """
    raw = os.environ.get("LLM_PROVIDER_ORDER")
    if not raw:
        return DEFAULT_PROVIDER_ORDER
    parts = tuple(p.strip().lower() for p in raw.split(",") if p.strip())
    valid = tuple(p for p in parts if p in ALL_PROVIDERS)
    dropped = tuple(p for p in parts if p not in ALL_PROVIDERS)
    if dropped:
        logger.warning(
            "llm_router: LLM_PROVIDER_ORDER dropped unknown provider name(s) %r "
            "(known providers: %r)",
            dropped,
            ALL_PROVIDERS,
        )
    if not valid:
        logger.warning(
            "llm_router: LLM_PROVIDER_ORDER=%r yielded no known providers; "
            "falling back to default order %r",
            raw,
            DEFAULT_PROVIDER_ORDER,
        )
        return DEFAULT_PROVIDER_ORDER
    return valid


def tier_for_model_string(model_string: str) -> str:
    """Map a free-form LiteLLM model string to a router tier.

    Heuristic: anything containing ``large``, ``opus``, or ``sonnet`` (anywhere,
    case-insensitive) is routed through the ``large`` tier; everything else
    falls into ``small``.  Used by legacy call sites (adzuna agent, adzuna
    extractor, rerank scorer) that still take a CLI ``--model`` argument but
    delegate concrete provider/model selection to ``llm_router.config.TIERS``.
    """
    s = (model_string or "").lower()
    if any(marker in s for marker in ("large", "opus", "sonnet")):
        return "large"
    return "small"


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
