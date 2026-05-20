"""Tool 1.5 — LLM structured field extraction for Adzuna vacancies.

Sends vacancy title + description to the LLM (Mistral SGR) and parses the
response into a VacancyExtracted object whose field names and allowed values
mirror the enum constants in schema_features.py.

Every LLM call is written to a JSONL audit log via llm_logging.py so
extraction quality can be monitored per vacancy_id.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from adzuna.config import (
    EXTRACTOR_DELAY_SECONDS,
    EXTRACTOR_MODEL,
)
from sara_retrieve_rerank.llm_logging import (
    setup_llm_jsonl_logger,
    wrap_invoke_with_logging,
)
from sara_retrieve_rerank.schema_features import (
    COMPANY_TYPES,
    EMPLOYEE_TYPES,
    ENGLISH_LEVELS,
    GRADES,
    WORK_FORMATS,
)


# ── Pydantic schema for SGR ───────────────────────────────────────────────────

class VacancyExtracted(BaseModel):
    """Structured fields extracted from a vacancy title + description.

    Enum values are taken directly from schema_features.py constants so the
    merged vacancy dict is accepted by vacancy_schema_features() and
    pair_schema_features() without additional normalisation.
    """

    grades:          list[str]       = []
    """Seniority levels required. Allowed: """ + ", ".join(GRADES) + "."

    english_level:   str | None      = None
    """Minimum English level. Allowed: """ + ", ".join(ENGLISH_LEVELS) + " or null."

    work_format:     list[str]       = []
    """Work arrangements. Allowed: """ + ", ".join(WORK_FORMATS) + "."

    employee_type:   list[str]       = []
    """Contract types. Allowed: """ + ", ".join(EMPLOYEE_TYPES) + "."

    company_type:    str | None      = None
    """Company type. Allowed: """ + ", ".join(COMPANY_TYPES) + " or null."

    tags:            list[str]       = []
    """Technologies and tools mentioned in the vacancy."""

    specializations: list[str]       = []
    """Domain specialisations (e.g. fintech, computer vision, b2b saas)."""

    company_domains: list[str]       = []
    """Industry sectors the company operates in."""


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT: str = f"""
You are a structured vacancy parser.  Given a job title and description,
extract the following fields and return a valid JSON object.

Allowed values (use only these exact strings):
  grades:        {list(GRADES)}
  english_level: {list(ENGLISH_LEVELS)} or null
  work_format:   {list(WORK_FORMATS)}
  employee_type: {list(EMPLOYEE_TYPES)}
  company_type:  {list(COMPANY_TYPES)} or null
  tags:          free list of technology / skill names
  specializations: free list of domain names (fintech, computer vision, …)
  company_domains: free list of industry sectors

Return ONLY a valid JSON object.  No extra text, no markdown.
""".strip()


def _build_prompt(vacancy: dict) -> str:
    """Compose the user-turn message for one vacancy."""
    return (
        f"Title: {vacancy.get('title', '')}\n\n"
        f"Description:\n{vacancy.get('tldr_sanitized', '')[:1500]}"
    )


# ── Logging wrapper ───────────────────────────────────────────────────────────

class _LLMConfig:
    """Carries model name + current vacancy_id into the logging wrapper's
    extra callback without using global mutable state."""

    def __init__(self, model: str, vacancy_id: str = ""):
        self.model      = model
        self.vacancy_id = vacancy_id


def _sgr_invoke(llm: _LLMConfig, prompt: str) -> Any:
    """Raw litellm SGR call; returns the completion response object."""
    from litellm import completion  # lazy import to keep startup fast

    return completion(
        model    = llm.model,
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        response_format = VacancyExtracted,
        temperature     = 0,
    )


def _make_logged_invoke(logger: logging.Logger) -> Any:
    """Wrap _sgr_invoke with the project's JSONL audit logger.

    The extra callback injects vacancy_id into each log record so
    extraction failures can be traced to specific vacancies.
    """
    return wrap_invoke_with_logging(
        _sgr_invoke,
        logger    = logger,
        component = "vacancy_extractor",
        extra     = lambda llm, _prompt: {"vacancy_id": llm.vacancy_id},
    )


# ── Public tool ───────────────────────────────────────────────────────────────

def extract_vacancy_fields_batch(
    vacancy_dicts: list[dict],
    *,
    model:         str  = EXTRACTOR_MODEL,
    delay_seconds: float = EXTRACTOR_DELAY_SECONDS,
) -> list[VacancyExtracted]:
    """Tool 1.5 — extract structural fields from a batch of partial vacancy dicts.

    For each vacancy, sends title + description to the LLM and parses the
    SGR response into VacancyExtracted.  Failures are caught and logged;
    the output list always has the same length as the input so callers can
    zip input and output safely.

    Parameters
    ----------
    vacancy_dicts:
        Partial dicts from scraper.adzuna_to_vacancy_dict().
    model:
        litellm model string.
    delay_seconds:
        Pause between calls to avoid rate-limit errors.
    """
    logger        = setup_llm_jsonl_logger()
    logged_invoke = _make_logged_invoke(logger)
    results: list[VacancyExtracted] = []
    agent_log = logging.getLogger("sara.agent")
    total = len(vacancy_dicts)
    n_ok = 0
    n_failed = 0

    agent_log.info(
        "Field extraction started — %d vacancies  model=%s  delay=%.1fs",
        total, model, delay_seconds,
    )

    for idx, vacancy in enumerate(vacancy_dicts, start=1):
        vacancy_id = str(vacancy.get("dataset_id", ""))
        llm_config = _LLMConfig(model=model, vacancy_id=vacancy_id)
        prompt     = _build_prompt(vacancy)

        try:
            response  = logged_invoke(llm_config, prompt)
            extracted = VacancyExtracted.model_validate_json(
                response.choices[0].message.content
            )
            n_ok += 1
            agent_log.debug("[%d/%d] extracted  id=%s", idx, total, vacancy_id)

        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            is_rate_limit = "429" in str(exc) or "RateLimit" in str(exc) or "capacity" in str(exc).lower()
            if is_rate_limit:
                agent_log.warning(
                    "[%d/%d] Rate limit (429) for %s — increase VACANCY_EXTRACTOR_DELAY "
                    "(currently %.1f s).  Using empty fields.",
                    idx, total, vacancy_id, delay_seconds,
                )
            else:
                agent_log.warning(
                    "[%d/%d] Extraction failed for %s: %s — using empty fields",
                    idx, total, vacancy_id, exc,
                )
            extracted = VacancyExtracted()

        results.append(extracted)
        time.sleep(delay_seconds)

    agent_log.info(
        "Field extraction finished — %d/%d ok  %d failed",
        n_ok, total, n_failed,
    )
    return results
