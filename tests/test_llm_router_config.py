"""Config tests: TIERS layout, default provider order, env-driven overrides."""

from __future__ import annotations

import pytest

from llm_router import config


def test_default_provider_order_is_anthropic_first(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER_ORDER", raising=False)
    assert config.provider_order() == ("anthropic", "github", "mistral")


def test_provider_order_env_override_full(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistral,github,anthropic")
    assert config.provider_order() == ("mistral", "github", "anthropic")


def test_provider_order_env_override_partial(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "github,mistral")
    assert config.provider_order() == ("github", "mistral")


def test_provider_order_drops_unknown_names(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "anthropic,bogus,mistral")
    assert config.provider_order() == ("anthropic", "mistral")


def test_provider_order_falls_back_to_default_when_all_unknown(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "bogus,more-bogus")
    assert config.provider_order() == config.DEFAULT_PROVIDER_ORDER


def test_tier_has_anthropic_fields() -> None:
    small = config.tier("small")
    assert small.anthropic.startswith("claude-")
    assert small.anthropic_litellm == f"anthropic/{small.anthropic}"

    large = config.tier("large")
    assert large.anthropic.startswith("claude-")
    assert large.anthropic_litellm == f"anthropic/{large.anthropic}"


def test_tier_defaults_to_sonnet_4_6() -> None:
    # Both tiers default to the same Sonnet model — Anthropic is the
    # reliability primary, not a quality differentiator.
    assert config.tier("small").anthropic == "claude-sonnet-4-6"
    assert config.tier("large").anthropic == "claude-sonnet-4-6"


def test_anthropic_model_env_override_per_tier(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_SMALL_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("ANTHROPIC_LARGE_MODEL", "claude-opus-4-1-20250805")
    config.refresh_tiers()
    try:
        assert config.tier("small").anthropic == "claude-haiku-4-5"
        assert config.tier("small").anthropic_litellm == "anthropic/claude-haiku-4-5"
        assert config.tier("large").anthropic == "claude-opus-4-1-20250805"
        assert (
            config.tier("large").anthropic_litellm
            == "anthropic/claude-opus-4-1-20250805"
        )
    finally:
        monkeypatch.delenv("ANTHROPIC_SMALL_MODEL", raising=False)
        monkeypatch.delenv("ANTHROPIC_LARGE_MODEL", raising=False)
        config.refresh_tiers()


def test_tier_lookup_unknown() -> None:
    with pytest.raises(KeyError):
        config.tier("xxl")


def test_all_providers_constant() -> None:
    assert set(config.ALL_PROVIDERS) == {"anthropic", "github", "mistral"}


# ── tier_for_model_string ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model_string,expected",
    [
        ("mistral/mistral-small-latest", "small"),
        ("openai/openai/gpt-4o-mini", "small"),
        ("openai/openai/gpt-4o", "small"),  # no "large"/"opus"/"sonnet" marker
        ("mistral/mistral-large-latest", "large"),
        ("anthropic/claude-sonnet-4-6", "large"),
        ("anthropic/claude-opus-4-1-20250805", "large"),
        ("anthropic/Claude-Sonnet-4-6", "large"),  # case-insensitive
        ("", "small"),
        (None, "small"),  # tolerates None
    ],
)
def test_tier_for_model_string(model_string, expected) -> None:
    assert config.tier_for_model_string(model_string) == expected


# ── provider_order logging on unknown names ──────────────────────────────


def test_provider_order_logs_warning_on_typo(monkeypatch, caplog) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "mistal,github,anthropic")
    with caplog.at_level("WARNING", logger="llm_router.config"):
        result = config.provider_order()
    assert result == ("github", "anthropic")
    assert any(
        "dropped unknown provider name" in rec.message and "mistal" in rec.message
        for rec in caplog.records
    )


def test_provider_order_logs_warning_when_all_unknown(monkeypatch, caplog) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "foo,bar")
    with caplog.at_level("WARNING", logger="llm_router.config"):
        result = config.provider_order()
    assert result == config.DEFAULT_PROVIDER_ORDER
    assert any("yielded no known providers" in rec.message for rec in caplog.records)


def test_provider_order_no_warning_when_clean(monkeypatch, caplog) -> None:
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "anthropic,mistral")
    with caplog.at_level("WARNING", logger="llm_router.config"):
        result = config.provider_order()
    assert result == ("anthropic", "mistral")
    assert not any("dropped" in rec.message for rec in caplog.records)
