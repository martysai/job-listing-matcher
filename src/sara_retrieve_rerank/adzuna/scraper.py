"""Tool 1 — Adzuna HTTP scraper and vacancy dict mapper.

scrape_adzuna_batch
    Async: executes a list of AdzunaQuery objects, deduplicates results by
    vacancy id, and returns raw Adzuna JSON dicts.

adzuna_to_vacancy_dict
    Maps one raw Adzuna dict to the field layout expected by:
      - documents.vacancy_to_text()       (embedding text)
      - documents.vacancy_to_metadata()   (Chroma metadata)
      - schema_features.vacancy_schema_features()  (reranker features)
    Fields unavailable from Adzuna are set to empty; extractor.py fills them.

merge_extracted
    Merges LLM-extracted fields (from extractor.py) back into the partial dict.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from sara_retrieve_rerank.adzuna.config import (
    ADZUNA_APP_ID,
    ADZUNA_APP_KEY,
    ADZUNA_BASE_URL,
    ADZUNA_CONTRACT_TO_WORK_FORMAT,
    ADZUNA_CONTRACT_TO_WORK_TYPE,
    ADZUNA_COUNTRY,
    AdzunaQuery,
    SCRAPE_DELAY_SECONDS,
)
from sara_retrieve_rerank.preprocessing import clean_html

if TYPE_CHECKING:
    from sara_retrieve_rerank.adzuna.extractor import VacancyExtracted


async def scrape_adzuna_batch(
    queries: list[AdzunaQuery],
    country: str = ADZUNA_COUNTRY,
    delay_seconds: float = SCRAPE_DELAY_SECONDS,
) -> list[dict]:
    """Execute all queries sequentially and return deduplicated raw vacancy dicts.

    Skips queries that return 0 results immediately (saves API quota).
    Continues on HTTP errors and timeouts, logging a warning per failure.
    """
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError("pip install aiohttp") from exc

    all_vacancies: list[dict] = []
    seen_ids: set[str] = set()
    log = logging.getLogger("sara.agent")

    async with aiohttp.ClientSession() as session:
        for i, query in enumerate(queries):
            url    = ADZUNA_BASE_URL.format(country=country, page=1)
            params = {**query.to_params(), "app_id": ADZUNA_APP_ID, "app_key": ADZUNA_APP_KEY}

            try:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status != 200:
                        log.warning("Query %d HTTP %d — skip", i, resp.status)
                        continue
                    results = (await resp.json()).get("results", [])
                    if not results:
                        continue
                    new = [v for v in results if str(v["id"]) not in seen_ids]
                    seen_ids.update(str(v["id"]) for v in new)
                    all_vacancies.extend(new)
                    log.info(
                        "Query %3d +%4d  (total %d)  what=%r  where=%r",
                        i, len(new), len(all_vacancies), query.what, query.where,
                    )
            except asyncio.TimeoutError:
                log.warning("Query %d: timeout — skip", i)
            except Exception as exc:  # noqa: BLE001
                log.warning("Query %d: %s — skip", i, exc)

            await asyncio.sleep(delay_seconds)

    return all_vacancies


def adzuna_to_vacancy_dict(raw: dict) -> dict:
    """Partially map a raw Adzuna API response to the project vacancy schema.

    The returned dict uses the same field names as vacancies_safe_ml_dataset
    so it can be passed directly to vacancy_to_text(), vacancy_to_metadata(),
    and vacancy_schema_features() without further adaptation.

    Fields that Adzuna does not expose (grades, english_level, company_type,
    tags, specializations, company_domains) are set to empty values.
    Call merge_extracted() after extractor.extract_vacancy_fields_batch()
    to fill them in.
    """
    contract = raw.get("contract_type", "")
    loc      = raw.get("location") or {}
    area     = loc.get("area") or {}

    return {
        "dataset_id":      str(raw["id"]),
        "source":          "adzuna",
        # ── embedding text fields (vacancy_to_text) ───────────────────────────
        "title":           raw.get("title", ""),
        "original_title":  raw.get("title", ""),   # no separate original in Adzuna
        "work_format":     ADZUNA_CONTRACT_TO_WORK_FORMAT.get(contract, []),
        "work_type":       ADZUNA_CONTRACT_TO_WORK_TYPE.get(contract, ""),
        "english_level":   "",    # → extractor
        "employee_type":   [],    # → extractor
        "company_type":    "",    # → extractor
        "tags":            [],    # → extractor
        "grades":          [],    # → extractor
        "specializations": [],    # → extractor
        "company_domains": [],    # → extractor
        "title_aliases":   [],
        "cities":   [loc.get("display_name")]  if loc.get("display_name")  else [],
        "regions":  [area.get("display_name")] if area.get("display_name") else [],
        "tldr_sanitized":  clean_html(raw.get("description", "")),
        # ── reranker salary features (schema_features._to_usd) ────────────────
        "salary": {
            "min":           raw.get("salary_min"),
            "max":           raw.get("salary_max"),
            "currency":      None,   # Adzuna does not reliably expose currency
            "salary_in_usd": None,
        },
        # Adzuna has no quality score; 0.0 is the neutral default.
        # There is no has_vacancy_score indicator in schema_features, so zeros
        # do not mislead the reranker.
        "vacancy_score": 0.0,
    }


def merge_extracted(partial: dict, extracted: "VacancyExtracted") -> dict:
    """Merge LLM-extracted structured fields into the partial vacancy dict.

    work_format from the LLM is applied only when the mapper could not
    infer it (empty list), to avoid an LLM hallucination overwriting a
    reliable contract_type signal.
    """
    merged = dict(partial)
    merged["grades"]          = extracted.grades
    merged["english_level"]   = extracted.english_level or ""
    merged["employee_type"]   = extracted.employee_type
    merged["company_type"]    = extracted.company_type or ""
    merged["tags"]            = extracted.tags
    merged["specializations"] = extracted.specializations
    merged["company_domains"] = extracted.company_domains
    if not merged["work_format"] and extracted.work_format:
        merged["work_format"] = extracted.work_format
    return merged
