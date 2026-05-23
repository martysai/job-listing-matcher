"""Tests for the sara_candidate_poll client + parser tier mapping.

We swap ``llm_router.parse_structured`` to verify the legacy ``model=``
argument is honoured by mapping it to a router tier — the regression we
want to lock in is that ``model="…large…"`` doesn't silently degrade to
the small tier (the original bug the review caller flagged).
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

import llm_router as _router_pkg
from sara_candidate_poll import client as candidate_client


class _Echo(BaseModel):
    text: str


@pytest.fixture()
def capture_parse(monkeypatch):
    calls: list[dict] = []

    def fake(prompt, response_format, *, tier):
        calls.append({"prompt": prompt, "tier": tier, "response_format": response_format})
        return response_format(text=prompt)

    # The client imports `parse_structured as _router_parse_structured` at
    # import time, so patch that bound name to intercept calls.
    monkeypatch.setattr(candidate_client, "_router_parse_structured", fake)
    return calls


def test_default_model_routes_to_large(capture_parse) -> None:
    candidate_client.parse_structured("hi", _Echo)
    assert capture_parse[0]["tier"] == "large"


def test_explicit_large_model_routes_to_large(capture_parse) -> None:
    candidate_client.parse_structured("hi", _Echo, model="mistral-large-latest")
    assert capture_parse[0]["tier"] == "large"


def test_sonnet_model_routes_to_large(capture_parse) -> None:
    candidate_client.parse_structured("hi", _Echo, model="claude-sonnet-4-6")
    assert capture_parse[0]["tier"] == "large"


def test_opus_model_routes_to_large(capture_parse) -> None:
    candidate_client.parse_structured("hi", _Echo, model="claude-opus-4-1-20250805")
    assert capture_parse[0]["tier"] == "large"


def test_small_model_routes_to_small(capture_parse) -> None:
    candidate_client.parse_structured("hi", _Echo, model="mistral-small-latest")
    assert capture_parse[0]["tier"] == "small"


def test_empty_model_defaults_to_large(capture_parse) -> None:
    candidate_client.parse_structured("hi", _Echo, model="")
    assert capture_parse[0]["tier"] == "large"
