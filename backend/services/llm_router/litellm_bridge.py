"""Bridge for callers that prefer LiteLLM / LangChain ``ChatLiteLLM``.

The Adzuna ReAct agent, the Adzuna SGR extractor, and the rerank scorer all
go through LiteLLM today.  Rather than reimplement them on the provider
interface, we hand them a primary+fallback model assembly that LiteLLM /
LangChain understands natively:

* ``litellm_completion(tier=…)`` — calls ``litellm.completion`` with the
  primary model, then retries on the GitHub Models model when LiteLLM raises
  a transient error.
* ``make_chat_model(tier=…)`` — returns a LangChain runnable that already has
  ``.with_fallbacks([...])`` attached so callers (e.g. ``ChatLiteLLM`` users)
  can keep their existing ``llm.invoke(prompt)`` code untouched.

Both honour ``LLM_PROVIDER_ORDER`` so flipping primary↔fallback is a one-env
change.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from .auth import (
    AnthropicAuthError,
    GitHubModelsAuthError,
    load_anthropic_key,
    load_github_pat,
)
from .config import (
    GITHUB_MODELS_BASE_URL,
    PROVIDER_ANTHROPIC,
    PROVIDER_GITHUB,
    PROVIDER_MISTRAL,
    provider_order,
    tier as _tier,
)
from .errors import classify

logger = logging.getLogger(__name__)


# ── shared model-string assembly ─────────────────────────────────────────

# Header names we'll redact when an exception text echoes the request.
_SECRET_KW_NAMES = ("api_key", "Authorization", "x-api-key", "anthropic-api-key")


def _scrub_secret(text: str, secret: str) -> str:
    if not secret:
        return text
    return text.replace(secret, "***")


def _scrub_summary(summary: str, secrets: Iterable[str]) -> str:
    out = summary
    for s in secrets:
        out = _scrub_secret(out, s)
    return out


def _model_for(provider: str, tier: str) -> str:
    cfg = _tier(tier)
    if provider == PROVIDER_ANTHROPIC:
        return cfg.anthropic_litellm
    if provider == PROVIDER_MISTRAL:
        return cfg.mistral_litellm
    if provider == PROVIDER_GITHUB:
        return cfg.github_litellm
    raise ValueError(f"Unknown provider {provider!r}")


def _kwargs_for(provider: str) -> dict[str, Any]:
    """Provider-specific LiteLLM kwargs (api_base/api_key)."""
    if provider == PROVIDER_GITHUB:
        try:
            token = load_github_pat()
        except GitHubModelsAuthError as exc:
            # Surface as a transient-shaped error so the primary chain still
            # gets a chance; the router/bridge will raise if both fail.
            logger.warning("llm_router: GitHub Models auth unavailable: %s", exc)
            raise
        return {
            "api_base": GITHUB_MODELS_BASE_URL,
            "api_key": token,
        }
    if provider == PROVIDER_ANTHROPIC:
        try:
            token = load_anthropic_key()
        except AnthropicAuthError as exc:
            logger.warning("llm_router: Anthropic auth unavailable: %s", exc)
            raise
        return {"api_key": token}
    return {}


def _ordered_providers() -> list[str]:
    return list(provider_order())


# ── LiteLLM completion with failover ─────────────────────────────────────

def litellm_completion(
    *,
    tier: str,
    messages: list[dict],
    **extra: Any,
) -> Any:
    """Call ``litellm.completion`` with primary→fallback failover.

    Extra kwargs (``temperature``, ``response_format``, etc.) are forwarded
    unchanged to every attempt.
    """
    from litellm import completion  # lazy import to keep startup fast

    errors: list[tuple[str, BaseException]] = []
    used_secrets: list[str] = []
    for provider in _ordered_providers():
        try:
            model = _model_for(provider, tier)
            provider_kwargs = _kwargs_for(provider)
        except (GitHubModelsAuthError, AnthropicAuthError, ValueError) as exc:
            errors.append((provider, exc))
            continue

        secret = provider_kwargs.get("api_key")
        if secret:
            used_secrets.append(secret)

        try:
            return completion(
                model=model,
                messages=messages,
                **provider_kwargs,
                **extra,
            )
        except Exception as exc:
            verdict = classify(exc)
            errors.append((provider, exc))
            if verdict == "fatal":
                raise
            logger.warning(
                "llm_router: litellm provider %r transient, failing over: %s: %s",
                provider,
                type(exc).__name__,
                exc,
            )
            continue

    summary = "; ".join(f"{n}: {type(e).__name__}: {e}" for n, e in errors)
    raise RuntimeError(
        f"All LiteLLM providers failed: {_scrub_summary(summary, used_secrets)}"
    )


# ── LangChain ChatLiteLLM with .with_fallbacks([...]) ────────────────────

def make_chat_model(
    *,
    tier: str,
    temperature: float = 0,
    api_base: str | None = None,
    **extra: Any,
) -> Any:
    """Return a LangChain runnable with built-in primary→fallback failover.

    Used by the Adzuna agent (ReAct loop) and the rerank scorer.  The
    returned object behaves like a normal ``ChatLiteLLM``:
    ``llm.invoke(prompt)`` and ``llm.with_config(...)`` keep working, but a
    transient failure on the primary provider transparently retries on the
    fallback.

    ``api_base`` (when supplied) is applied to the Mistral primary only;
    other providers use their own native endpoints.

    If the *primary* provider's credentials are missing (e.g. GitHub PAT
    file absent, Anthropic key file absent), it is silently demoted to the
    fallback list and the next eligible provider becomes primary — so the
    router's "skip a broken provider" contract holds even when LiteLLM is
    the call surface.  ``RuntimeError`` is raised only if *no* provider can
    be constructed.
    """
    from langchain_litellm import ChatLiteLLM

    providers = _ordered_providers()
    if not providers:
        raise RuntimeError("No LLM providers configured")

    built: list[Any] = []
    skip_reasons: list[str] = []
    for provider_name in providers:
        try:
            kwargs: dict[str, Any] = {
                "model": _model_for(provider_name, tier),
                "temperature": temperature,
            }
            if api_base and provider_name == PROVIDER_MISTRAL:
                kwargs["api_base"] = api_base
            kwargs.update(_kwargs_for(provider_name))
            kwargs.update(extra)
            built.append(ChatLiteLLM(**kwargs))
        except (GitHubModelsAuthError, AnthropicAuthError) as exc:
            logger.warning(
                "llm_router: skipping provider %r (auth unavailable): %s",
                provider_name,
                exc,
            )
            skip_reasons.append(f"{provider_name}: {exc}")

    if not built:
        raise RuntimeError(
            "make_chat_model: no usable providers (auth failures: "
            f"{'; '.join(skip_reasons) or 'none'})"
        )

    primary = built[0]
    fallbacks = built[1:]
    if not fallbacks:
        return primary
    return primary.with_fallbacks(fallbacks)
