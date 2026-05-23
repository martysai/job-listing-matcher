"""Error classification tests for ``llm_router.errors.classify``.

We do not import provider SDKs here; classification is duck-typed by class
name, attributes, and HTTP status, so synthetic exception classes are
sufficient and keep the tests fast and hermetic.
"""

from __future__ import annotations

import pytest

from llm_router.errors import classify


def _make(name: str, *, status: int | None = None) -> Exception:
    cls = type(name, (Exception,), {})
    exc = cls("synthetic")
    if status is not None:
        exc.status_code = status  # type: ignore[attr-defined]
    return exc


@pytest.mark.parametrize(
    "name",
    [
        "RateLimitError",
        "TooManyRequestsError",
        "Timeout",
        "TimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "ServiceUnavailableError",
        "ReadTimeout",
        "SDKError",
    ],
)
def test_transient_by_name(name: str) -> None:
    assert classify(_make(name)) == "transient"


@pytest.mark.parametrize(
    "name",
    [
        "AuthenticationError",
        "PermissionDeniedError",
        "BadRequestError",
        "NotFoundError",
        "UnprocessableEntityError",
        "InvalidRequestError",
        "ContentPolicyViolationError",
    ],
)
def test_fatal_by_name(name: str) -> None:
    assert classify(_make(name)) == "fatal"


@pytest.mark.parametrize("status", [408, 413, 425, 429, 500, 502, 503, 504, 522, 524])
def test_transient_by_status(status: int) -> None:
    exc = _make("WeirdError", status=status)
    assert classify(exc) == "transient"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_fatal_by_status(status: int) -> None:
    exc = _make("WeirdError", status=status)
    assert classify(exc) == "fatal"


def test_apistatuserror_413_request_too_large_is_transient() -> None:
    # GitHub Models caps gpt-4o at 8k input tokens and rejects oversized
    # requests with HTTP 413 ``tokens_limit_reached``.  Anthropic Sonnet
    # 4.6 has 200k context, so failover is the right behaviour rather than
    # propagating the failure to the caller.
    exc = _make("APIStatusError", status=413)
    assert classify(exc) == "transient"


def test_apierror_with_fatal_4xx_status_is_fatal() -> None:
    # An "APIError" subclass that carries a 401 should still be fatal — we
    # never want to retry an auth failure on the fallback provider.
    exc = _make("APIError", status=401)
    assert classify(exc) == "fatal"


def test_builtin_timeout_is_transient() -> None:
    assert classify(TimeoutError("slow")) == "transient"


def test_builtin_connection_error_is_transient() -> None:
    assert classify(ConnectionError("broken pipe")) == "transient"


def test_unknown_exception_defaults_to_fatal() -> None:
    # Conservative default: only retry when we have a positive signal.
    assert classify(ValueError("bad input")) == "fatal"


def test_response_object_status_is_inspected() -> None:
    exc = Exception("wrapped")

    class _Resp:
        status_code = 503

    exc.response = _Resp()  # type: ignore[attr-defined]
    assert classify(exc) == "transient"
