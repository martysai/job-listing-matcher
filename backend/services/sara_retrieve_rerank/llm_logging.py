"""Structured JSONL logging for LLM-based components.

Logs prompt, response, latency, and error for every wrapped LLM call. The
wrapper is opt-in and side-effect-free unless explicitly applied to an invoke
function, so existing call paths and tests are unaffected.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEFAULT_LOGGER_NAME = "sara.llm"


def setup_llm_jsonl_logger(
    path: str | Path | None = None,
    *,
    logger_name: str = DEFAULT_LOGGER_NAME,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a JSONL logger for LLM calls.

    When ``path`` is None the logger propagates to its parent so records
    reach whatever handler the application has attached (e.g. JsonlHandler
    on the root ``sara`` logger).  When ``path`` is given a FileHandler is
    added and propagation is disabled — used by tests and standalone scripts.

    Idempotent: repeated calls with the same ``path`` do not stack handlers.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if path is None:
        logger.propagate = True
        return logger

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.propagate = False

    target = str(output_path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == output_path.resolve():
            return logger

    handler = logging.FileHandler(target, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def wrap_invoke_with_logging(
    invoke: Callable[[Any, str], Any],
    *,
    logger: logging.Logger | None = None,
    component: str = "llm",
    extra: Callable[[Any, str], dict[str, Any]] | None = None,
) -> Callable[[Any, str], Any]:
    """Wrap an LLM `invoke(llm, prompt)` function with JSONL audit logging.

    The wrapper does not change return values or exceptions, only logs them.
    `extra` can produce additional structured fields (e.g. candidate / vacancy
    IDs) per call.
    """
    log = logger or logging.getLogger(DEFAULT_LOGGER_NAME)

    def logging_invoke(llm: Any, prompt: str) -> Any:
        start = time.perf_counter()
        record: dict[str, Any] = {
            "timestamp": time.time(),
            "component": component,
            "prompt": prompt,
        }
        if extra is not None:
            try:
                record.update(extra(llm, prompt))
            except Exception:
                # Telemetry must never break the caller.
                pass
        try:
            response = invoke(llm, prompt)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["latency_ms"] = (time.perf_counter() - start) * 1000.0
            log.info(json.dumps(record, ensure_ascii=False, default=str))
            raise

        record["response"] = str(getattr(response, "content", response))
        record["latency_ms"] = (time.perf_counter() - start) * 1000.0
        log.info(json.dumps(record, ensure_ascii=False, default=str))
        return response

    return logging_invoke
