"""Provider-agnostic error classification for failover decisions.

The router fails over from one provider to the next only when the underlying
exception is transient (rate limit, server error, network blip).  Authentication
failures and request-shape errors are fatal and propagate immediately, because
retrying them on another provider would either fail the same way or hide a real
bug.
"""

from __future__ import annotations

from typing import Literal

Verdict = Literal["transient", "fatal"]


_TRANSIENT_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504, 522, 524}


def _status_code(exc: BaseException) -> int | None:
    """Best-effort extraction of an HTTP status code from disparate SDK errors."""
    for attr in ("status_code", "http_status", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def classify(exc: BaseException) -> Verdict:
    """Classify an exception as ``"transient"`` (failover) or ``"fatal"`` (raise).

    Detection is duck-typed by class name and attributes so the module does not
    have to import optional SDKs (``openai``, ``mistralai``, ``litellm``,
    ``httpx``) at module-import time.
    """
    name = type(exc).__name__

    # Fatal first: anything auth-shaped or shape-of-request shaped.
    fatal_markers = (
        "AuthenticationError",
        "PermissionDeniedError",
        "BadRequestError",
        "NotFoundError",
        "UnprocessableEntityError",
        "InvalidRequestError",
        "ContentPolicyViolationError",
    )
    if any(marker in name for marker in fatal_markers):
        return "fatal"

    # Strongly transient by class name.
    transient_markers = (
        "RateLimitError",
        "TooManyRequestsError",
        "Timeout",
        "TimeoutError",
        "APIConnectionError",
        "ConnectionError",
        "ServiceUnavailableError",
        "InternalServerError",
        "APIError",
        "ServerError",
        "RemoteProtocolError",
        "ReadError",
        "WriteError",
        "ConnectError",
        "PoolTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "ConnectTimeout",
        "SDKError",
    )
    if any(marker in name for marker in transient_markers):
        status = _status_code(exc)
        if status is not None and 400 <= status < 500 and status not in _TRANSIENT_STATUS_CODES:
            return "fatal"
        return "transient"

    # Fallback: check HTTP status if the exception carries one.
    status = _status_code(exc)
    if status is not None:
        if status in _TRANSIENT_STATUS_CODES or status >= 500:
            return "transient"
        if 400 <= status < 500:
            return "fatal"

    # Built-in network errors are transient.
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return "transient"

    return "fatal"
